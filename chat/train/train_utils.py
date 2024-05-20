# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

import time
import os
from tqdm import tqdm
import sys
from copy import deepcopy

from contextlib import suppress
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from einops import rearrange


class Dict2Class:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


class SysLogger(object):
    def __init__(self, filename="../log/log.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message+'\n')
        self.log.write(message)


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    trainloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    tb
):
    # setup loaders
    num_batches_per_epoch = len(trainloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    print('num_batches_per_epoch={}, total_training_steps={}'.format(num_batches_per_epoch, total_training_steps))

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
    assert media_token_id == tokenizer.encode("<audio>")[-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, batch in tqdm(
        enumerate(trainloader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    ):

        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### FORWARD PASS ####
        audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)  # (B, N_WINDOWS, WINDOW_LENGTH)
        audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)  # (B, N_WINDOWS)
        input_ids = batch["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)  # (B, N_TOKENS)
        attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)  # (B, N_TOKENS)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, :2] = -100
        labels[labels == tokenizer.encode("<audio>")[-1]] = -100

        # mask all prompts except for between <SEP> and <|endofchunk|>
        sep_locations = labels == tokenizer.sep_token_id
        eoc_locations = labels == endofchunk_token_id

        if not all(sep_locations.sum(dim=1) == eoc_locations.sum(dim=1)):
            print("Warning: <SEP>-<EoC> pairing mismatch at step {} due to max_token limit.".format(num_steps))

        for i in range(labels.shape[0]):
            shouldmask = True
            for j in range(labels.shape[1]):
                if shouldmask and (labels[i][j] != tokenizer.eos_token_id):
                    masked_value = -100
                else:
                    masked_value = labels[i][j]

                if labels[i][j] == tokenizer.sep_token_id:
                    shouldmask = False
                elif labels[i][j] == endofchunk_token_id:
                    shouldmask = True
                
                labels[i][j] = masked_value
            
            if labels[i][-1] not in [-100, tokenizer.eos_token_id, tokenizer.pad_token_id, endofchunk_token_id]:
                for j in range(labels.shape[1]-1, -1, -1):
                    if labels[i][j] not in [-100, tokenizer.eos_token_id, endofchunk_token_id]:
                        labels[i][j] = -100
                    else:
                        break

        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                audio_x=audio_clips,
                audio_x_mask=audio_embed_mask,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output.loss

        divided_loss = loss / args.gradient_accumulation_steps
        train_loss = divided_loss * args.loss_multiplier
        train_loss.backward()

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <audio> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0:
                samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )
                log_dict = {
                    "data_time": data_time_m.avg,
                    "step_time": step_time_m.avg,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": loss.item()
                }

                if ((num_steps + 1) % args.logging_steps == 0):
                    for key in log_dict:
                        tb.add_scalar("Train/{}".format(key), log_dict[key], global_step)

                step_time_m.reset()
                data_time_m.reset()

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0):
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}\n"
            )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <audio> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        checkpoint_dir = os.path.join(args.expdir, args.run_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {checkpoint_dir}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{checkpoint_dir}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0 and epoch % 20 != 0:
                try:
                    os.remove(f"{checkpoint_dir}/checkpoint_{epoch-1}.pt")
                except:
                    pass
