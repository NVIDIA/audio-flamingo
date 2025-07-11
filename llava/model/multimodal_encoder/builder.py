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

# This file is modified from https://github.com/haotian-liu/LLaVA/

import os

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from .whisper_encoder import WhisperSpeechTower
from .afwhisper_audio_encoder import AFWhisperSoundTower

def build_speech_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_name_or_path is None:
        return None
    speech_tower = WhisperSpeechTower(model_name_or_path, config)
    config.speech_hidden_size = speech_tower.config.hidden_size
    return speech_tower

def build_sound_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_name_or_path is None:
        return None
    sound_tower = AFWhisperSoundTower(model_name_or_path, config)
    config.sound_hidden_size = 1280
    return sound_tower

