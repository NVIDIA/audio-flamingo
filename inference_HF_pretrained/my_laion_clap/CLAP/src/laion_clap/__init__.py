# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/LAION-AI/CLAP under the CC0-1.0 license.
#   LICENSE is in incl_licenses directory.

import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
from .hook import CLAP_Module