# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llava.mm_utils import dynamic_process_images_and_prompt, dynamic_s2_process_images_and_prompt, process_images
from llava.train.args import DataArguments
from llava.utils.logging import logger
from llava.utils.media import extract_media
from llava.utils.tokenizer import preprocess_conversation

__all__ = ["BaseDataset"]

def _process_speech(speech: List[Any], data_args: DataArguments) -> torch.Tensor:
    return torch.tensor(speech)

def _process_sound(sound: List[Any], data_args: DataArguments) -> torch.Tensor:
    return torch.tensor(sound)

def _process_sound_masks(sound_masks: List[Any], data_args: DataArguments) -> torch.Tensor:
    return torch.tensor(sound_masks)


class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        no_system_prompt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.no_system_prompt = no_system_prompt
        self.instances = []
        self.enable_dynamic_res = False
        self.enable_dynamic_res_s2 = False
        # global_batch_size: int,
        self.global_batch_size = kwargs.get("global_batch_size", 1)

        # by default, dataset cls will resample on failure
        self.resample_on_failure = kwargs.get("resample_on_failure", True)

        # by default, dataset cls will resample on failure
        self.resample_on_failure = kwargs.get("resample_on_failure", True)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        try:
            # Process instance to conversation
            conversation = self.process(instance)

            # Extract media from conversation
            media, media_meta = extract_media(conversation, self.data_args)

            if "speech" in media:
                processed_speech = _process_speech(media["speech"], self.data_args)
            if "sound" in media:
                processed_sound = _process_sound(media["sound"], self.data_args)
                processed_sound_feature_masks = _process_sound_masks(media_meta["sound_feature_masks"], self.data_args)
                processed_sound_embed_masks = _process_sound_masks(media_meta["sound_embed_masks"], self.data_args)
            # Prepare "input_ids" and "labels" for training
            data = preprocess_conversation(conversation, self.tokenizer, no_system_prompt=self.no_system_prompt)

            if "speech" in media:
                data["speech"] = processed_speech
            if "sound" in media:
                data["sound"] = processed_sound
                data["sound_feature_masks"] = processed_sound_feature_masks
                data["sound_embed_masks"] = processed_sound_embed_masks

        except Exception as e:
            if not self.resample_on_failure:
                raise e
            else:
                logger.exception(f"Error processing instance '{instance}': '{e}'. Resampling.")
                return self.__getitem__(random.randint(0, len(self.instances) - 1))

        return data

    def __len__(self) -> int:
        return len(self.instances)
