# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import os

from llava.utils import io

__all__ = ["EVAL_ROOT", "TASKS"]


EVAL_ROOT = "scripts/eval"
TASKS = io.load(os.path.join(os.path.dirname(__file__), "registry_audio.yaml"))
