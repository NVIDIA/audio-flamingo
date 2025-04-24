# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

""" Main training script """

import argparse
import functools
import glob
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import random
import shutil
import sys 
sys.path.append('../')
import yaml
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
torch.cuda.empty_cache() 

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data.data import get_audiotext_dataloader  # AudioTextData, DataCollator
from distributed import init_distributed_device, world_info_from_env
from train_utils import (
    train_one_epoch,
    get_mp_policy_dtype,
    save_checkpoint,
    Dict2Class,
    get_autocast, 
    get_cast_dtype
)
from valid_utils import validation_losses
from src.factory import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/config.yaml', help='yaml config path')
    parsed_args = parser.parse_args()

    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config["clap_config"]
    args = Dict2Class(config['train_config'])

    if 'sft_config' in config:
        sft_config = config['sft_config']
        unfreeze_full_lm = sft_config['unfreeze_full_lm']
    else:
        sft_config = None
        unfreeze_full_lm = False

    # get paths done 
    exp_path = os.path.join(args.expdir, args.run_name)
    os.makedirs(exp_path, exist_ok=True)
    print('exp_path:', exp_path)
    shutil.copy(parsed_args.config, os.path.join(exp_path, 'config.yaml'))
    data_config["dataset_blending_output"] = os.path.join(exp_path, data_config["dataset_blending_output"])

    # Validate args
    if args.fsdp and not args.fsdp_use_orig_params:
        print(
            "Warning: FSDP is running without fsdp_use_orig_params flag. "
            + "This is not recommended because it means we will use uniform weight decay"
            + " and train all embeddings, not just the newly added ones. "
            + "Note: OPT models are not compatible with fsdp_use_orig_params flag."
        )

    if args.fsdp and args.fsdp_sharding_strategy == "hybrid":
        print(
            "Warning: As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
            + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
            + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
            + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
        )

    # Set up distributed training
    print('initializing distributed environment')
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    print('creating model')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        unfreeze_full_lm=unfreeze_full_lm
    )
    random_seed(args.seed, args.rank)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")

    # Load model checkpoint on CPU
    checkpoint_list = glob.glob(f"{args.expdir}/{args.run_name}/checkpoint_*.pt")
    if len(checkpoint_list) == 0:
        print(f"Found no checkpoints for run {args.run_name}.")
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = sorted(
            checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        print(
            f"Found checkpoint {resume_from_checkpoint} for run {args.run_name}."
        )

    # load pretrained model
    resume_from_epoch = 0
    if (resume_from_checkpoint is None) and (sft_config is not None):
        # just started SFT
        pretrained_path = os.path.join(
            sft_config['pretrained_path'],
            sft_config['pretrained_ckpt']
        )
        if args.rank == 0:
            print(f"Loading checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)
            del checkpoint["model_state_dict"]
            del msd


    elif resume_from_checkpoint is not None:
        # continue training (either pretraining or STF)
        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)
            del checkpoint["model_state_dict"]
            del msd
    
    else:
        pass

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )

        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper),
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # apply weight decay only to params in the xattn layers
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        # unclear if we should be using no weight decay or small weight decay for all parameters
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # load optimizer checkpoint
    if resume_from_checkpoint is not None:
        osd = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
        optimizer.load_state_dict(osd)
        del checkpoint["optimizer_state_dict"]
        del osd

    # Initialize data loaders
    AudioTextDataInfo = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, args.batch_size, split='train',
        epoch=0, force_reblend=True
    )

    total_training_steps = (
        len(AudioTextDataInfo.dataset) // (args.batch_size * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
        tb = SummaryWriter(os.path.join(exp_path, 'tensorboard'))
    else:
        tb = None

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        del checkpoint["lr_scheduler_state_dict"]

    # Start training!
    ddp_model.train()

    print('start training from epoch {}'.format(resume_from_epoch))
    for epoch in range(resume_from_epoch, args.num_epochs):
        # force reblending dataset for every epoch
        if epoch > 0:
            AudioTextDataInfo = get_audiotext_dataloader(
                data_config, clap_config, tokenizer, args.batch_size, split='train',
                epoch=epoch, force_reblend=True
            )
        AudioTextDataInfo.set_epoch(epoch)
        trainloader = AudioTextDataInfo.dataloader
        
        # train one epoch
        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            trainloader=trainloader,
            device_id=device_id,
            tb=tb
        )

        # save checkpoint
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)
        time.sleep(1.0)

        # validation 
        if epoch % 5 == 0:
            torch.distributed.barrier()
            try:
                with torch.no_grad():
                    valid_losses = validation_losses(
                        model=ddp_model, 
                        data_config=data_config, 
                        clap_config=clap_config, 
                        tokenizer=tokenizer, 
                        batch_size=args.batch_size, 
                        autocast=get_autocast(args.precision, cache_enabled=(not args.fsdp)), 
                        cast_dtype=get_cast_dtype(args.precision),
                        device_id=device_id
                    )

                if args.rank == 0:
                    for key in valid_losses:
                        tb.add_scalar("Valid/{}".format(key), valid_losses[key], (epoch+1)*len(trainloader))
            
            except Exception as error:
                print("An exception occurred:", error)
                
            torch.distributed.barrier()
        
    # save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)
    if args.rank == 0:
        tb.close()


if __name__ == "__main__":
    main()
