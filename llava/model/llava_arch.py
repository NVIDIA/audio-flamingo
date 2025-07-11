# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import logging
import os
import os.path as osp
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict, deque
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from transformers import AutoConfig, GenerationConfig, LogitsProcessor, PreTrainedModel
from transformers.modeling_utils import ContextManagers, no_init_weights

from llava.constants import DEFAULT_SOUND_TOKEN,DEFAULT_SPEECH_TOKEN, IGNORE_INDEX, NUM_EXTRA_TOKENS
from llava.mm_utils import process_image, process_images, process_sounds,process_sound_masks
from llava.model.configuration_llava import LlavaConfig, ResponseFormat
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_sound_tower
from llava.model.multimodal_projector.builder import build_speech_mm_projector, build_sound_mm_projector
from llava.model.utils import get_model_config
from llava.train.sequence_parallel import get_pg_manager
from llava.utils import distributed
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation


class LlavaMetaModel(ABC):
    def _init_llm(self, llm_cfg, config, *args, **kwargs):
        llm, tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        return llm, tokenizer

    def init_vlm(self, config: PreTrainedModel = None, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower") or hasattr(self, "speech_tower") or hasattr(self, "sound_tower") or hasattr(self, "mm_projector") or hasattr(self, "speech_mm_projector") or hasattr(self, "sound_mm_projector"):
            # already initialized, skipped
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        print(cfgs)
        if len(cfgs) == 7:
            llm_cfg, vision_tower_cfg, speech_tower_cfg,sound_tower_cfg, mm_projector_cfg, speech_mm_projector_cfg,sound_mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `speech_mm_projector_cfg` `sound_mm_projector_cfg` `vision_tower_cfg` `speech_tower_cfg` `sound_tower_cfg` not found in the config.")

        self.llm, self.tokenizer = self._init_llm(llm_cfg, config, *args, **kwargs)
        
        self.sound_tower = build_sound_tower(sound_tower_cfg, config)
        self.sound_mm_projector = build_sound_mm_projector(sound_mm_projector_cfg, config)
        
        if isinstance(self.config, dict):
            self.vocab_size = config.llm_cfg["vocab_size"] + NUM_EXTRA_TOKENS
        else:
            self.vocab_size = self.tokenizer.vocab_size + NUM_EXTRA_TOKENS
            logging.info(
                f"[XGrammar] config is not a dict, loading vocab size from tokenizer {self.tokenizer.vocab_size} + {NUM_EXTRA_TOKENS} => {self.vocab_size}"
            )

        # XGrammar tokenizer and grammar compiler
        # lazy init only when specified json output during inference
        self.grammar_compiler = None

        self.encoders = {}
        for name in ["sound"]:
            config = getattr(self.config, f"{name}_encoder")
            if isinstance(config, str):
                config = json.loads(config)
            self.encoders[name] = instantiate(config, parent=self)

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.speech_tower is not None or self.mm_projector is not None or self.speech_mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass

    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, LlavaConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(
                f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, LlavaConfig)}"
            )

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 7:
            llm_cfg, vision_tower_cfg, speech_tower_cfg,sound_tower_cfg, mm_projector_cfg, speech_mm_projector_cfg,sound_mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `speech_mm_projector_cfg` `sound_mm_projector_cfg` `vision_tower_cfg` `speech_tower_cfg` `sound_tower_cfg` not found in the config.")

        init_context = [
            no_init_weights(_enable=True),
        ]

        with ContextManagers(init_context):
            vlm = cls(config, *args, **kwargs)

        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower") or hasattr(vlm, "speech_tower") or hasattr(vlm, "sound_tower") or hasattr(vlm, "mm_projector") or hasattr(vlm, "speech_mm_projector") or hasattr(vlm, "sound_mm_projector"):
            if vlm.is_loaded:
                return vlm

        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.sound_tower = build_sound_tower(sound_tower_cfg, config)
        vlm.sound_mm_projector = build_sound_mm_projector(sound_mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        # FIXME(ligeng, yunhao): llm should never be none here.
        assert (
            vlm.llm is not None or vlm.vision_tower is not None or vlm.speech_tower is not None or vlm.mm_projector is not None or vlm.speech_mm_projector is not None
        ), "At least one of the components must be instantiated."
        return vlm

    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()

        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.llm.config

        
        if self.get_sound_tower():
            print(f"saving sound_tower to {osp.join(output_dir, 'sound_tower')}")
            self.sound_tower.config._name_or_path = osp.join(output_dir, "sound_tower")
            sound_tower_state_dict = OrderedDict(
                {k.split("sound_tower.sound_tower.")[-1]: v for k, v in state_dict.items() if "sound_tower" in k}
            )
            self.sound_tower.sound_tower.save_pretrained(
                os.path.join(output_dir, "sound_tower"),
                state_dict=sound_tower_state_dict,
            )
            self.config.sound_tower_cfg = self.sound_tower.config
        
        if self.get_sound_mm_projector():
            print(f"saving sound_mm_projector to {osp.join(output_dir, 'sound_mm_projector')}")
            self.sound_mm_projector.config._name_or_path = osp.join(output_dir, "sound_mm_projector")
            sound_mm_projector_state_dict = OrderedDict(
                {k.split("sound_mm_projector.")[-1]: v for k, v in state_dict.items() if "sound_mm_projector" in k}
            )
            self.sound_mm_projector.save_pretrained(
                os.path.join(output_dir, "sound_mm_projector"),
                state_dict=sound_mm_projector_state_dict,
            )
            self.config.sound_mm_projector_cfg = self.sound_mm_projector.config

        ## update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head
    
    def get_sound_tower(self):
        sound_tower = getattr(self, "sound_tower", None)
        if type(sound_tower) is list:
            sound_tower = sound_tower[0]
        return sound_tower


    def get_sound_mm_projector(self):
        sound_mm_projector = getattr(self, "sound_mm_projector", None)
        if type(sound_mm_projector) is list:
            sound_mm_projector = sound_mm_projector[0]
        return sound_mm_projector

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
            self.config.speech_tower_cfg = self.speech_tower.config
        if getattr(self.config, "sound_tower_cfg", None) is None:
            self.config.sound_tower_cfg = self.sound_tower.config
        if getattr(self.config, "sound_mm_projector_cfg", None) is None:
            self.config.sound_mm_projector_cfg = self.sound_mm_projector.config

    def freezed_module_patch(self):
        """
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        """
        if self.training:
            if self.get_llm() and not getattr(self.config, "tune_language_model", False):
                pass

            if self.get_sound_tower() and not getattr(self.config, "tune_sound_tower", False):
                self.get_sound_tower().eval()
            if self.get_sound_mm_projector() and not getattr(self.config, "tune_sound_mm_projector", False):
                self.get_sound_mm_projector().eval()

    
    def encode_sound(self, sounds, masks=None):
   
        sound_features = self.get_sound_tower()(sounds, masks)
        sound_features = self.get_sound_mm_projector()(sound_features)
        return sound_features

    ## @yunhao: is there a better way to handle function call and attributes for llm?
    ## support beam search
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class LlavaMetaForCausalLM(ABC):
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        media_meta: Dict[str, Dict[str, Any]]= None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            for name in media:
                self.encoders[name].end_tokens = None

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        media_embeds = self.__embed_media_tokens(media, media_config, media_meta)
        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0
        # Remove padding
        batch_size = labels.shape[0]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # -------------------------------- #
        num_audio_tokens = torch.stack(media_meta["sound_embed_masks"], dim=0).sum(-1)
        num_audio_tokens = torch.tensor([round(int(x) / 10) * 10 for x in num_audio_tokens])            
        num_audios = len(media_embeds['sound']) # length of queue is the number of audios we have in total
        max_audio_tokens, embed_dim = media_embeds['sound'][0].shape
       

        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)

        audio_embeds = []
        while media_embeds['sound']:       
            audio_embeds.append(media_embeds['sound'].popleft())
        audio_embeds = torch.stack(audio_embeds,dim=0)
        
        masked_audio_features = audio_embeds[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self.tokenizer.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.tokenizer.media_token_ids['sound'] #hard coded to just work with 'sound'
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = text_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.tokenizer.media_token_ids['sound']) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=text_embeds.dtype, device=text_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=text_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=text_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = text_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, IGNORE_INDEX, dtype=torch.long) #.to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=text_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        # # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(final_embedding, final_labels)
        return self.__batchify_sequence(inputs, labels)
      

    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        media_meta: Dict[str, Dict[str, Any]]= None,
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            if self.training:
                # Gather metainfo of media objects from all ranks
                info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                infos = list(chain(*distributed.all_gather(info)))

                # The entire batch does not contain any media objects of this type.
                if not infos:
                    continue

                # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                if media.get(name) is None or len(media[name]) == 0:
                    dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.device)
                    embeds["dummy"].extend(self.encoders[name]([dummy], media_config[name]))
                    continue
            embeds[name] = deque(self.encoders[name](media[name], media_config[name], media_meta['sound_feature_masks'])) # hard coded
        return embeds

    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and any(len(input) > self.tokenizer.model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({self.tokenizer.model_max_length}).")
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    def repack_multimodal_data(self, inputs_embeds, attention_mask, position_ids, labels):
        # Handle sequence parallelism
        PROCESS_GROUP_MANAGER = get_pg_manager()

        # We do re-sharding instead of packing here to ensure the sequence length is the same across all ranks.
        if PROCESS_GROUP_MANAGER is not None:
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank
            sp_group = PROCESS_GROUP_MANAGER.sp_pg
            ring_degree = PROCESS_GROUP_MANAGER.ring_degree
            ring_rank = PROCESS_GROUP_MANAGER.ring_rank
            ring_type = PROCESS_GROUP_MANAGER.ring_type
            ulysses_degree = PROCESS_GROUP_MANAGER.ulysses_degree
            ulysses_rank = PROCESS_GROUP_MANAGER.ulysses_rank

            bs, shard_seqlen = position_ids.shape
            sp_seq_len = [torch.zeros(1, dtype=torch.int64, device=position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_seq_len, torch.tensor(shard_seqlen, device=position_ids.device), group=sp_group)
            sp_seq_len_cat = torch.cat(sp_seq_len, dim=0)

            if sp_rank == 0:
                original_start_id = 0
            else:
                original_start_id = torch.sum(sp_seq_len_cat[:sp_rank]).item()
            original_end_id = torch.sum(sp_seq_len_cat[: sp_rank + 1]).item()

            # Gather attention_mask, position_ids, labels and input_embeds
            all_inputs_embeds = torch.zeros(
                bs,
                torch.sum(sp_seq_len_cat),
                inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ).contiguous()
            all_inputs_embeds[:, original_start_id:original_end_id, :] += inputs_embeds
            dist.barrier(group=sp_group)
            dist.all_reduce(all_inputs_embeds, group=sp_group)
            dist.barrier(group=sp_group)

            attention_mask_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(sp_degree)
            ]
            position_ids_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=position_ids.dtype, device=position_ids.device)
                for i in range(sp_degree)
            ]
            labels_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=labels.dtype, device=labels.device) for i in range(sp_degree)
            ]

            dist.all_gather(attention_mask_list, attention_mask, group=sp_group)
            dist.all_gather(position_ids_list, position_ids, group=sp_group)
            dist.all_gather(labels_list, labels, group=sp_group)

            effective_seqlen_list = [attention_mask_list[i].sum(dim=-1) for i in range(sp_degree)]
            effective_seqlen = torch.stack(effective_seqlen_list, dim=-1)
            effective_seqlen_batch_list = torch.unbind(effective_seqlen, dim=0)

            global_attention_mask_list = []
            global_position_ids_list = []
            global_labels_list = []
            global_inputs_embeds_list = []
            for i in range(bs):
                global_attention_mask_batch_list = []
                global_position_ids_batch_list = []
                global_labels_batch_list = []
                global_inputs_embeds_batch_list = []
                for j in range(sp_degree):
                    eff_len = effective_seqlen_batch_list[i][j]
                    prev_len = torch.sum(sp_seq_len_cat[:j]).item() if j > 0 else 0

                    global_attention_mask_batch_list.append(attention_mask_list[j][i, :eff_len])
                    global_position_ids_batch_list.append(position_ids_list[j][i, :eff_len])
                    global_labels_batch_list.append(labels_list[j][i, :eff_len])
                    global_inputs_embeds_batch_list.append(all_inputs_embeds[i, prev_len : prev_len + eff_len, :])
                global_attention_mask_list.append(torch.cat(global_attention_mask_batch_list, dim=0))
                global_position_ids_list.append(torch.cat(global_position_ids_batch_list, dim=0))
                global_labels_list.append(torch.cat(global_labels_batch_list, dim=0))
                global_inputs_embeds_list.append(torch.cat(global_inputs_embeds_batch_list, dim=0))

                global_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    global_attention_mask_list, batch_first=True, padding_value=False
                )
                global_position_ids = torch.nn.utils.rnn.pad_sequence(
                    global_position_ids_list, batch_first=True, padding_value=-1
                )
                global_labels = torch.nn.utils.rnn.pad_sequence(
                    global_labels_list, batch_first=True, padding_value=IGNORE_INDEX
                )
                global_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
                    global_inputs_embeds_list, batch_first=True, padding_value=0
                )

            # Re-shard the inputs
            if ring_degree > 1:
                total_effective_seqlen = torch.sum(effective_seqlen, dim=1)
                new_seqlen_per_rank = total_effective_seqlen // sp_degree
                assert torch.all(
                    total_effective_seqlen % sp_degree == 0
                ), "total_effective_seqlen must be divisible by sp_degree"

                max_new_seqlen = torch.max(new_seqlen_per_rank).item()

                new_attention_mask = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_attention_mask.dtype, device=global_attention_mask.device
                )
                new_position_ids = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_position_ids.dtype, device=global_position_ids.device
                )
                new_labels = torch.full(
                    (bs, max_new_seqlen), IGNORE_INDEX, dtype=global_labels.dtype, device=global_labels.device
                )
                new_inputs_embeds = torch.zeros(
                    (bs, max_new_seqlen, global_inputs_embeds.shape[-1]),
                    dtype=global_inputs_embeds.dtype,
                    device=global_inputs_embeds.device,
                )

                if ring_type == "ring_varlen":
                    for i in range(bs):
                        start_idx = new_seqlen_per_rank[i] * sp_rank
                        end_idx = start_idx + new_seqlen_per_rank[i]
                        new_attention_mask[i, : new_seqlen_per_rank[i]] = global_attention_mask[i, start_idx:end_idx]
                        new_position_ids[i, : new_seqlen_per_rank[i]] = global_position_ids[i, start_idx:end_idx]
                        new_labels[i, : new_seqlen_per_rank[i]] = global_labels[i, start_idx:end_idx]
                        new_inputs_embeds[i, : new_seqlen_per_rank[i], :] = global_inputs_embeds[
                            i, start_idx:end_idx, :
                        ]
                elif ring_type == "zigzag_ring_varlen":
                    chunk_size = total_effective_seqlen // (2 * sp_degree)
                    for i in range(bs):
                        # Zigzag pattern indices
                        if sp_degree == ring_degree:
                            forward_rank_idx = sp_rank
                            backward_rank_idx = 2 * sp_degree - sp_rank - 1
                        else:
                            ulysses_offset = ulysses_rank * ring_degree * 2
                            forward_rank_idx = ring_rank + ulysses_offset
                            backward_rank_idx = sp_degree - ring_rank - 1 + ulysses_offset

                        # Calculate start and end indices for the forward and backward zigzag
                        start_idx_fwd = forward_rank_idx * chunk_size[i]
                        end_idx_fwd = start_idx_fwd + chunk_size[i]

                        start_idx_bwd = backward_rank_idx * chunk_size[i]
                        end_idx_bwd = start_idx_bwd + chunk_size[i]

                        # Fill new tensors with zigzag data
                        new_attention_mask[i, : chunk_size[i]] = global_attention_mask[i, start_idx_fwd:end_idx_fwd]
                        new_attention_mask[i, chunk_size[i] : 2 * chunk_size[i]] = global_attention_mask[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_position_ids[i, : chunk_size[i]] = global_position_ids[i, start_idx_fwd:end_idx_fwd]
                        new_position_ids[i, chunk_size[i] : 2 * chunk_size[i]] = global_position_ids[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_labels[i, : chunk_size[i]] = global_labels[i, start_idx_fwd:end_idx_fwd]
                        new_labels[i, chunk_size[i] : 2 * chunk_size[i]] = global_labels[i, start_idx_bwd:end_idx_bwd]

                        new_inputs_embeds[i, : chunk_size[i], :] = global_inputs_embeds[i, start_idx_fwd:end_idx_fwd, :]
                        new_inputs_embeds[i, chunk_size[i] : 2 * chunk_size[i], :] = global_inputs_embeds[
                            i, start_idx_bwd:end_idx_bwd, :
                        ]
                else:
                    raise ValueError(f"Invalid ring_type: {ring_type}")
            else:
                global_seq_len = global_attention_mask.shape[-1]
                seq_len_sharded = global_seq_len // sp_degree
                start_idx_reshard = seq_len_sharded * sp_rank
                end_idx_reshard = start_idx_reshard + seq_len_sharded if sp_rank < sp_degree - 1 else global_seq_len

                new_attention_mask = torch.narrow(
                    global_attention_mask, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_position_ids = torch.narrow(
                    global_position_ids, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_labels = torch.narrow(global_labels, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard)
                new_inputs_embeds = torch.narrow(
                    global_inputs_embeds, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )

            return new_inputs_embeds, new_attention_mask, new_position_ids, new_labels

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        seqlens = [attention_mask[k].sum().item() for k in range(batch_size)]

        # Pack all sequences together
        inputs_embeds_p = [inputs_embeds[k][attention_mask[k]] for k in range(batch_size)]
        attention_mask_p = [torch.ones(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        position_ids_p = [torch.arange(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        labels_p = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Add one dummy token at the end of the packed sequence to ensure that `_get_unpacked_data` will be called
        inputs_embeds_p.append(torch.zeros(1, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=device))
        attention_mask_p.append(torch.tensor([0], dtype=torch.int, device=device))
        position_ids_p.append(torch.tensor([0], dtype=torch.int, device=device))
        labels_p.append(torch.tensor([IGNORE_INDEX], dtype=torch.int, device=device))

        # Mask the first token of each sequence to avoid contamination
        for label in labels_p:
            label[0] = IGNORE_INDEX

        # Batch the data
        inputs_embeds_p = torch.cat(inputs_embeds_p, dim=0).unsqueeze(0)
        attention_mask_p = torch.cat(attention_mask_p, dim=0).unsqueeze(0)
        position_ids_p = torch.cat(position_ids_p, dim=0).unsqueeze(0)
        labels_p = torch.cat(labels_p, dim=0).unsqueeze(0)

        if hasattr(
            self, "pad_to_multiple_of"
        ):  # related to quantization, please refer to ModelArguments for more information.
            assert len(labels_p.shape) == 2
            batch_size, max_length, cur_length = labels_p.shape[0], labels_p.shape[1], labels_p.shape[1]
            hidden_size = inputs_embeds_p.shape[-1]

            if max_length % self.pad_to_multiple_of != 0:
                max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
                difference = max_length - cur_length

                inputs_embeds_p = torch.cat(
                    (
                        inputs_embeds_p,
                        torch.full((batch_size, difference, hidden_size), self.llm.pad_token_id).to(inputs_embeds_p),
                    ),
                    dim=1,
                )
                labels_p = torch.cat((labels_p, torch.full((batch_size, difference), IGNORE_INDEX).to(labels_p)), dim=1)
                attention_mask_p = torch.cat(
                    (
                        attention_mask_p,
                        torch.zeros((batch_size, difference), dtype=torch.bool).to(attention_mask_p),
                    ),
                    dim=1,
                )
                position_ids_p = torch.cat(
                    (position_ids_p, torch.full((batch_size, difference), -1).to(position_ids_p)), dim=1
                )

        return inputs_embeds_p, attention_mask_p, position_ids_p, labels_p

    def get_xgr_logits_processor(self, response_format: ResponseFormat) -> List[LogitsProcessor]:
        # Convert response format to logits processor
        import xgrammar as xgr

        logging.info("[XGrammar] Compiling grammar for contrained output")

        if self.grammar_compiler is None:
            # logging.info(f"[XGrammar] {self.tokenizer}, {self.tokenizer.vocab_size}, {self.vocab_size}")
            self.grammar_compiler = xgr.GrammarCompiler(
                xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
            )

        if response_format.type == "json_schema":
            compiled_grammar = self.grammar_compiler.compile_json_schema(
                response_format.json_schema.schema_,
                indent=2,
            )
        else:
            compiled_grammar = self.grammar_compiler.compile_builtin_json_grammar()

        return [xgr.contrib.hf.LogitsProcessor(compiled_grammar)]

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        media_meta: Dict[str, Dict[str, Any]]= None,
        **generation_kwargs,
    ):
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask, media_meta)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        if response_format:
            xgr_logits_processor = self.get_xgr_logits_processor(response_format)
        else:
            xgr_logits_processor = None

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media, media_meta = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "sound":
                sounds = process_sounds(media["sound"]).half()         
                media[name] = [sound for sound in sounds]
                sound_feature_masks = process_sound_masks(media_meta["sound_feature_masks"]).half()   
                media_meta["sound_feature_masks"] = [sound_mask for sound_mask in sound_feature_masks]
                sound_embed_masks = process_sound_masks(media_meta["sound_embed_masks"]).half()   
                media_meta["sound_embed_masks"] = [sound_mask for sound_mask in sound_embed_masks]
            else:
                raise ValueError(f"Unsupported media type: {name}")
           

        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

        # Generate the response
        try:
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                media_meta=media_meta,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,  # structured generation
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            logging.warning("Generation failed with sampling, retrying with greedy decoding.")
            generation_config.do_sample = False
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                media_meta=media_meta,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,
            )

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return response

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        return generation_config
