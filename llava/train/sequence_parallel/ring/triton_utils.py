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

# Adopted from https://github.com/zhuzilin/ring-flash-attention.
# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889

import torch
import triton
import triton.language as tl


@triton.jit
def flatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_batch,
    stride_lse_nheads,
    stride_lse_seqlen,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_batch * stride_lse_batch + pid_head * stride_lse_nheads
    OUT = OUT + pid_head * stride_out_nheads + start_idx * stride_out_seqlen

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def flatten_varlen_lse(lse, cu_seqlens):
    """
    Arguments:
        lse: (batch_size, nheads, max_seqlen)
        cu_seqlens: (batch_size + 1,)
    Return:
        flatten_lse: (nheads, total_seqlen)
    """
    total_seqlen = cu_seqlens[-1]
    batch_size, nheads, max_seqlen = lse.shape
    output = torch.empty((nheads, total_seqlen), dtype=lse.dtype, device=lse.device)

    grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads)
    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        flatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            BLOCK_M,
        )
    return output


@triton.jit
def unflatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_batch,
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_seqlen,
    stride_lse_nheads,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_head * stride_lse_nheads + start_idx * stride_lse_seqlen
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    """
    Arguments:
        lse: (total_seqlen, nheads, 1)
        cu_seqlens: (batch_size + 1,)
        max_seqlen: int
    Return:
        unflatten_lse: (batch_size, nheads, max_seqlen)
    """
    lse = lse.unsqueeze(dim=-1)
    batch_size = len(cu_seqlens) - 1
    nheads = lse.shape[1]
    output = torch.empty(
        (batch_size, nheads, max_seqlen),
        dtype=lse.dtype,
        device=lse.device,
    )

    grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads)
    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        unflatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            BLOCK_M,
        )
    return output
