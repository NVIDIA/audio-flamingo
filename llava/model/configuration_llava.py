# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

from pydantic import BaseModel, Field
from transformers import PretrainedConfig


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        speech_hidden_size=None,
        sound_hidden_size=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        sound_mm_projector_lr=None,
        sound_tower_lr=None,
        num_time_tokens=None,
        time_token_format=None,
        sound_encoder: str = '{"_target_": "llava.model.encoders.BasicSoundEncoder"}',
        **kwargs,
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.resume_path = resume_path

        self.hidden_size = hidden_size
        self.sound_hidden_size = sound_hidden_size
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.sound_mm_projector_lr = sound_mm_projector_lr
        self.sound_tower_lr = sound_tower_lr
        self.num_time_tokens = num_time_tokens
        self.time_token_format = time_token_format

        self.sound_encoder = sound_encoder


class JsonSchemaResponseFormat(BaseModel):
    schema_: str = Field(alias="schema")


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None
