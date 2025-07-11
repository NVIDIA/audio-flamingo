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
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
DEFAULT_SOUND_TOKEN = "<sound>"
DEFAULT_SPEECH_TOKEN = "<speech>"
SENTINEL_TOKEN = "<vila/sentinel>"

MEDIA_TOKENS = {
    "speech": "<speech>",
    "sound": "<sound>",
}


"""
151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151646: AddedToken("[BOS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151647: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151648: AddedToken("<vila/sentinel>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151649: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151650: AddedToken("<vila/video>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151651: AddedToken("<sound>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151652: AddedToken("<speech>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),

"""
NUM_EXTRA_TOKENS = 10
