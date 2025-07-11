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

import torch

# 4 block
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, get_configs_io_block

"""SiLU Activation Backward"""
"""Input uses 1 * 16 group quantization"""
"""Grad uses 1 * 16 group quantization"""
"""Output uses per-tensor quantization, but should be quantized outside this function"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""


@triton.autotune(
    configs=[] + get_configs_io_block(),
    key=[
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["BLOCK_N"] // args["QB"],
    }
)
@triton.jit
def _fp8_silu_backward_legacy_kernel(
    output_ptr,
    output_scale_ptr,  # output
    input_ptr,
    input_scale_ptr,  # input
    grad_ptr,
    grad_scale_ptr,  # input
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input_stride_0,
    input_stride_1,  # input stride
    s_input_stride_0,
    s_input_stride_1,  # scale of input stride
    grad_stride_0,
    grad_stride_1,  # input stride
    s_grad_stride_0,
    s_grad_stride_1,  # scale of input stride
    output_stride_0,
    output_stride_1,  # output stride
    s_output_stride_0,
    s_output_stride_1,  # scale of output stride
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):  # CUDA block size

    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # input ptr
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr,
        shape=(M, SN),
        strides=(s_input_stride_0, s_input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr)
    scale_input = tl.load(scale_input_ptr)

    input = input.to(tl.float32)
    scale_input = scale_input.to(tl.float32)

    # Dequantize and silu calculation
    scale_input = tl.reshape(scale_input, (BLOCK_M, BLOCK_SN, 1))
    input = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))
    input = input * scale_input

    # pointers of gradient
    grad_block_ptr = tl.make_block_ptr(
        base=grad_ptr,
        shape=(M, N),
        strides=(grad_stride_0, grad_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # grad ptr
    scale_grad_ptr = tl.make_block_ptr(
        base=grad_scale_ptr,
        shape=(M, SN),
        strides=(s_grad_stride_0, s_grad_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    grad = tl.load(grad_block_ptr)
    scale_grad = tl.load(scale_grad_ptr)

    grad = grad.to(tl.float32)
    scale_grad = scale_grad.to(tl.float32)

    # Dequantize and silu calculation
    scale_grad = tl.reshape(scale_grad, (BLOCK_M, BLOCK_SN, 1))
    grad = tl.reshape(grad, (BLOCK_M, BLOCK_SN, QB))
    grad = grad * scale_grad

    # Actual Calculation of SiLU's backward
    sigmoid = 1 / (1.0 + libdevice.exp(-input))
    silu_output = sigmoid + input * sigmoid * (1 - sigmoid)
    silu_output = silu_output * grad

    # Quantize Scale calculation
    abs_output = tl.abs(silu_output)
    max_val = tl.max(abs_output, axis=2) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN, 1))

    # Quantize
    # silu_output = tl.fdiv(silu_output, scale_output)
    silu_output = silu_output.to(output_ptr.type.element_ty)

    scale_output = scale_output.to(output_scale_ptr.type.element_ty)
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN))
    silu_output = tl.reshape(silu_output, (BLOCK_M, BLOCK_N))

    # debug
    # silu_output = input
    # scale_output = scale_input

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(M, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, silu_output, boundary_check=(0, 1))
    tl.store(scale_output_ptr, scale_output, boundary_check=(0, 1))


def fp8_silu_backward_legacy(x, s_x, g, s_g, QB, stochastic=False):  # Stochastic Rounding is left outside this function
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        assert len(s_x.shape) == 3
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        s_x = s_x.reshape(-1, s_x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        s_g = s_g.reshape(-1, s_g.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    _, SN = s_x.shape  # assume the shape of quantization block size is always 1 * G

    y = torch.empty_like(g, dtype=torch.bfloat16)
    s_y = torch.empty_like(s_g, dtype=s_g.dtype)
    fp8MaxValue = FP8_MAX_VALUE[g.dtype]  # E4M3 and E5M2 have different max value

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_silu_backward_legacy_kernel[grid](
        y,
        s_y,
        x,
        s_x,
        g,
        s_g,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        x.stride(0),
        x.stride(1),
        s_x.stride(0),
        s_x.stride(1),
        g.stride(0),
        g.stride(1),
        s_g.stride(0),
        s_g.stride(1),
        y.stride(0),
        y.stride(1),
        s_y.stride(0),
        s_y.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])
        s_y = s_y.reshape(BS, -1, s_y.shape[-1])

    return y, s_y
