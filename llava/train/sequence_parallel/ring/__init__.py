# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

# Adopted from https://github.com/zhuzilin/ring-flash-attention.
# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889


from .ring_flash_attn import ring_flash_attn_func, ring_flash_attn_kvpacked_func, ring_flash_attn_qkvpacked_func
from .ring_flash_attn_varlen import (
    ring_flash_attn_varlen_func,
    ring_flash_attn_varlen_kvpacked_func,
    ring_flash_attn_varlen_qkvpacked_func,
)
from .stripe_flash_attn import stripe_flash_attn_func, stripe_flash_attn_kvpacked_func, stripe_flash_attn_qkvpacked_func
from .zigzag_ring_flash_attn import (
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_kvpacked_func,
    zigzag_ring_flash_attn_qkvpacked_func,
)
from .zigzag_ring_flash_attn_varlen import (
    zigzag_ring_flash_attn_varlen_func,
    zigzag_ring_flash_attn_varlen_qkvpacked_func,
)
