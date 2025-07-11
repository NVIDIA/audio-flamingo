# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import PreTrainedTokenizer

from llava.constants import IGNORE_INDEX
from llava.utils.logging import logger

__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Gather everything from the batch
        input_ids, labels, media, block_sizes = [], [], {name: [] for name in self.tokenizer.media_tokens}, []

        media_meta = {}

        media_meta["sound_feature_masks"] = []
        media_meta["sound_embed_masks"] = []
        media_meta["frame_times"] = []
        for instance in instances:
            if isinstance(instance["input_ids"], torch.Tensor):
                input_ids.append(instance["input_ids"])
                labels.append(instance["labels"])
                for name in media:
                    objs = instance.get(name)
                    objs = objs if objs is not None else []
                    media[name].append([obj for obj in objs])
                if instance.get("sound") is not None:
                    for name_k in media_meta:
                        if "sound" in name_k:
                            objs = instance.get(name_k)
                            media_meta[name_k].append([obj for obj in objs])
                if instance.get("video") is not None or instance.get("image") is not None:
                    for name_k in media_meta:
                        if "frame" in name_k:
                            objs = instance.get(name_k)
                            media_meta[name_k].append([obj for obj in objs])
                if "block_sizes" in instance:
                    block_sizes.append(instance["block_sizes"])
                else:
                    block_sizes.append(
                        [None for _ in range(len(instance.get("image")))] if instance.get("image") is not None else []
                    )
            else:
                input_ids.extend(instance["input_ids"])
                labels.extend(instance["labels"])
                for name in media:
                    objs = instance.get(name)
                    objs = objs if objs is not None else [[] for _ in range(len(instance["input_ids"]))]
                    media[name].extend(objs)
                if instance.get("sound") is not None:
                    for name_k in media_meta:
                        if "sound" in name_k:
                            objs = instance.get(name_k)
                            media_meta[name_k].extend(objs)
                if instance.get("video") is not None or instance.get("image") is not None:
                    for name_k in media_meta:
                        if "frame" in name_k:
                            objs = instance.get(name_k)
                            media_meta[name_k].append([obj for obj in objs])
                if "block_sizes" in instance:
                    block_sizes.extend(instance["block_sizes"])
                else:
                    block_sizes.extend(
                        [[None for _ in range(len(objs))] for objs in instance.get("image")]
                        if instance.get("image") is not None
                        else [[] for _ in range(len(instance["input_ids"]))]
                    )

        batch_size = len(input_ids)
        

        # Check if the number of media objects (or the number of block sizes) matches the number of media tokens
        for name in media:
            for k in range(batch_size):
                if name == "image" and not all([_ is None for _ in block_sizes[k]]):
                    actual = len(block_sizes[k])
                else:
                    actual = len(media[name][k])
                expected = (input_ids[k] == self.tokenizer.media_token_ids[name]).sum().item()
                if actual != expected:
                    raise ValueError(
                        f"Number mismatch between {name} objects and {name} tokens. "
                        f"There are {expected} {name} tokens but {actual} {name} objects."
                    )
        
        # Batchify the inputs
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Truncate media objects if necessary
        for name in media:
            objects = []
            for k in range(batch_size):
                if name == "image" and not all([_ is None for _ in block_sizes[k]]):
                    actual = len(media[name][k])
                    num_large_scale_blocks = sum([x * y for x, y in block_sizes[k]])
                    num_small_scale_blocks = actual - num_large_scale_blocks
                    num_small_scale_blocks_each_img = num_small_scale_blocks // len(block_sizes[k])
                    expected_full_image = (input_ids[k] == self.tokenizer.media_token_ids[name]).sum().item()
                    expected = (
                        sum([x * y for x, y in block_sizes[k][:expected_full_image]])
                        + num_small_scale_blocks_each_img * expected_full_image
                    )
                    if actual > expected:
                        logger.warning(f"Truncating the number of {name} objects from {actual} to {expected}")
                        media[name][k] = media[name][k][:expected]
                    objects.extend(media[name][k])
                    block_sizes[k] = block_sizes[k][:expected_full_image]
                else:
                    actual = len(media[name][k])
                    expected = (input_ids[k] == self.tokenizer.media_token_ids[name]).sum().item()
                    if actual > expected:
                        logger.warning(f"Truncating the number of {name} objects from {actual} to {expected}")
                        media[name][k] = media[name][k][:expected]
                    objects.extend(media[name][k])
                    if name == "image":
                        block_sizes[k] = block_sizes[k][:expected]
            media[name] = objects

        for name in media_meta:
            objects = []
            for k in range(batch_size):
                try:
                    objects.extend(media_meta[name][k])
                except:
                    continue
            media_meta[name] = objects
     
        # Flatten block sizes from [[bls_im1_instance1, bls_im2_instance1], [bls_im1_instance2, bls_im2_instance2], ...] to [bls_im1_instance1, bls_im2_instance1, bls_im1_instance2, bls_im2_instance2, ...]
        block_sizes = sum(block_sizes, [])
        return {
            "input_ids": input_ids,
            "media": media,
            "media_config": {"image": {"block_sizes": block_sizes}, "video": {}, "speech": {}, "sound": {}},
            "labels": labels,
            "attention_mask": attention_mask,
            "media_meta": media_meta,
        }
