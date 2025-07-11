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

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F



class SpeechTower(nn.Module):
    def __init__(self, speech_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.speech_tower_name = speech_tower
        self.cfg_only = None

    def forward(self, speeches):
        if type(speeches) is list:
            speech_features = []
            for speech in speeches:
                speech_feature = self.speech_tower.encoder(speech)
                speech_feature = speech_feature.last_hidden_state
                speech_feature = speech_feature.to(speech.dtype)
                speech_features.append(speech_feature)
        else:
            speech_features = self.speech_tower.encoder(speeches)
            speech_features = speech_features.last_hidden_state
            speech_features = speech_features.to(speeches.dtype)

        return speech_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.speech_tower.dtype

    @property
    def device(self):
        return self.speech_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.speech_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size


