# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
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
import math
from importlib import import_module

import torch
import torch.distributed
import torch.distributed as dist
import torch_npu

from vllm_ascend.quantization.quant_type import QuantType

COMM_STREAM = None

_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def load_cann_mega_moe_ops():
    ops_module = import_module("cann_ops_transformer.ops")
    get_symm_buffer_for_mega_moe = ops_module.get_symm_buffer_for_mega_moe
    mega_moe = ops_module.mega_moe
    return get_symm_buffer_for_mega_moe, mega_moe


def _get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    # Returns (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type).
    # The current custom op package still requires explicit INT4 for W4A8
    # packed weights; otherwise it derives W4A8's packed N as an INT8 N and
    # rejects weight2.
    #
    # dispatch_quant_out_dtype: the doc types this as torch.dtype (torch.int8 /
    # torch.float8_e4m3fn). We pass the ACL enum ints (258 / 24) because W8A8
    # was validated end-to-end this way in PD; switching W4A8 to torch.int8 did
    # NOT fix the W4A8 accuracy issue and slowed graph capture (see bug_a3.md),
    # so keep the working values until the W4A8 accuracy root cause is found on
    # the operator side.
    if quant_type == QuantType.W8A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8)
    if quant_type == QuantType.W4A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4)
    raise RuntimeError(
        "MegaMoe integration supports W8A8/W4A8 INT on A2/A3 and MXFP on FP8-capable "
        "MegaMoe platforms. "
        f"Unsupported quant type: {quant_type}."
    )


def get_cann_megamoe_dummy_token_capacity(num_experts: int, num_topk: int) -> int:
    """Reserve enough global rows to route to every expert once."""
    if num_topk < 1:
        raise ValueError(f"CANN MegaMoe requires num_topk to be positive, got {num_topk}.")
    if num_experts < num_topk:
        raise ValueError(
            "CANN MegaMoe dummy routing requires num_experts >= num_topk, got "
            f"num_experts={num_experts}, num_topk={num_topk}."
        )
    return math.ceil(num_experts / num_topk)


def get_cann_megamoe_buffer_params(
    base_num_max_tokens_per_rank: int,
    ep_world_size: int,
    num_experts: int,
    num_topk: int,
) -> tuple[int, int, int, int]:
    """Return max tokens, local experts, dummy rows, and receive bound."""
    dummy_token_capacity = get_cann_megamoe_dummy_token_capacity(num_experts, num_topk)
    num_max_tokens_per_rank = base_num_max_tokens_per_rank + dummy_token_capacity
    if not 1 <= num_max_tokens_per_rank <= 4096:
        raise ValueError(f"CANN MegaMoe requires num_max_tokens_per_rank in [1, 4096], got {num_max_tokens_per_rank}.")
    if ep_world_size not in {2, 4, 8, 16, 32}:
        raise ValueError(f"CANN MegaMoe only supports EP sizes 2, 4, 8, 16, and 32, got {ep_world_size}.")
    if num_experts % ep_world_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_world_size={ep_world_size}.")
    if not 1 <= num_topk <= 16:
        raise ValueError(f"CANN MegaMoe requires num_topk in [1, 16], got {num_topk}.")
    num_experts_per_rank = num_experts // ep_world_size
    max_recv_token_num = num_max_tokens_per_rank * ep_world_size * min(num_topk, num_experts_per_rank)
    return num_max_tokens_per_rank, num_experts_per_rank, dummy_token_capacity, max_recv_token_num


def append_cann_megamoe_dummy_tokens(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_active_mask: torch.Tensor | None,
    num_experts: int,
    ep_rank_id: int,
    ep_world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Append equal-size A2 sentinels that cover all experts globally."""
    if ep_world_size < 1 or not 0 <= ep_rank_id < ep_world_size:
        raise ValueError(
            "CANN MegaMoe dummy routing requires a valid EP rank: "
            f"ep_rank_id={ep_rank_id}, ep_world_size={ep_world_size}."
        )
    num_topk = int(topk_ids.shape[-1])
    global_dummy_capacity = get_cann_megamoe_dummy_token_capacity(num_experts, num_topk)
    local_dummy_capacity = math.ceil(global_dummy_capacity / ep_world_size)
    global_dummy_rows = ep_rank_id * local_dummy_capacity + torch.arange(
        local_dummy_capacity,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    dummy_topk_ids = global_dummy_rows[:, None] * num_topk + torch.arange(
        num_topk,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    dummy_topk_ids = dummy_topk_ids.remainder(num_experts)
    dummy_hidden_states = torch.ones(
        (local_dummy_capacity, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    dummy_topk_weights = torch.full(
        (local_dummy_capacity, num_topk),
        1.0 / num_topk,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )

    original_num_tokens = int(hidden_states.shape[0])
    hidden_states = torch.cat((hidden_states, dummy_hidden_states), dim=0)
    topk_ids = torch.cat((topk_ids, dummy_topk_ids), dim=0)
    topk_weights = torch.cat((topk_weights, dummy_topk_weights), dim=0)
    if x_active_mask is None:
        x_active_mask = torch.ones(original_num_tokens, dtype=torch.int8, device=hidden_states.device)
    dummy_mask = torch.ones(local_dummy_capacity, dtype=x_active_mask.dtype, device=x_active_mask.device)
    x_active_mask = torch.cat((x_active_mask, dummy_mask), dim=0)
    return hidden_states, topk_ids, topk_weights, x_active_mask, original_num_tokens
