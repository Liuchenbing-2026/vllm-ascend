#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""LoRA bgmv/sgmv ops: Triton kernels dispatched through torch custom ops.

Importing lora_ops_triton registers the kernels in the ``vllm_ascend_triton``
namespace (torch.library.custom_op), so torch._dynamo treats them like the
stock ``torch.ops._C_ascend.*`` ops.  All runtime checks and AscendC fallbacks
live inside the eager impls; the wrappers below keep the stock
vllm_ascend.lora.lora_ops API and stay free of data-dependent Python branches
so the serving path can be traced with fullgraph=True.
"""
import torch

from vllm_ascend.lora import lora_ops_triton  # noqa: F401  (registers custom ops)


def bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0):
    torch.ops.vllm_ascend_triton.bgmv_shrink(
        inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)
    return output_tensor


def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs=True):
    return bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                             0, output_tensor.size(1), add_inputs)


def bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, add_inputs=True):
    torch.ops.vllm_ascend_triton.bgmv_expand_slice(
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size)
    return output_tensor


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                lora_indices_tensor, batches, max_seq_length, token_nums, scaling):
    torch.ops.vllm_ascend_triton.sgmv_shrink(
        inputs, lora_a_weights, output_tensor, seq_len_tensor, lora_indices_tensor, scaling)
    return output_tensor


def sgmv_expand(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                lora_indices_tensor, batches, max_seq_length, token_nums, add_inputs=False):
    return sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                             lora_indices_tensor, batches, max_seq_length, token_nums,
                             0, output_tensor.size(1), add_inputs)


def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                      lora_indices_tensor, batches, max_seq_length, token_nums,
                      slice_offset, slice_size, add_inputs=False):
    torch.ops.vllm_ascend_triton.sgmv_expand_slice(
        inputs, lora_b_weights, output_tensor, seq_len_tensor, lora_indices_tensor,
        slice_offset, slice_size)
    return output_tensor
