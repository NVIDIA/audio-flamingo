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
        vision_tower_cfg=None,
        speech_tower_cfg=None,
        sound_tower_cfg=None,
        mm_projector_cfg=None,
        speech_mm_projector_cfg=None,
        sound_mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        speech_hidden_size=None,
        sound_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        fps=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_projector_lr=None,
        speech_mm_projector_lr=None,
        sound_mm_projector_lr=None,
        vision_tower_lr=None,
        speech_tower_lr=None,
        sound_tower_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        s2=None,
        dynamic_s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        s2_resize_output_to_scale_idx=0,
        min_tiles: Optional[int] = 1,
        max_tiles: Optional[int] = 12,
        video_max_tiles: Optional[int] = 1,
        num_time_tokens=None,
        time_token_format=None,
        image_encoder: str = '{"_target_": "llava.model.encoders.BasicImageEncoder"}',
        video_encoder: str = '{"_target_": "llava.model.encoders.BasicVideoEncoder"}',
        speech_encoder: str = '{"_target_": "llava.model.encoders.BasicSpeechEncoder"}',
        sound_encoder: str = '{"_target_": "llava.model.encoders.BasicSoundEncoder"}',
        **kwargs,
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.speech_tower_cfg = speech_tower_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.speech_mm_projector_cfg = speech_mm_projector_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.resume_path = resume_path

        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.speech_hidden_size = speech_hidden_size
        self.sound_hidden_size = sound_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.fps = fps
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.speech_mm_projector_lr = speech_mm_projector_lr
        self.sound_mm_projector_lr = sound_mm_projector_lr
        self.vision_tower_lr = vision_tower_lr
        self.speech_tower_lr = speech_tower_lr
        self.sound_tower_lr = sound_tower_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.dynamic_s2 = dynamic_s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size
        self.s2_resize_output_to_scale_idx = s2_resize_output_to_scale_idx
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.video_max_tiles = video_max_tiles
        self.num_time_tokens = num_time_tokens
        self.time_token_format = time_token_format

        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.speech_encoder = speech_encoder
        self.sound_encoder = sound_encoder


class JsonSchemaResponseFormat(BaseModel):
    schema_: str = Field(alias="schema")


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None
