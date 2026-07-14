# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    get_cann_megamoe_dummy_token_capacity,
    resolve_cann_megamoe_max_recv_tokens,
)

_DISPATCH_FFN_COMBINE_MODE = 1
_CANN_MEGAMOE_MODE = 2
_CANN_MEGAMOE_MODULE_NAME = "cann_ops_transformer.ops"
_CANN_MEGAMOE_DISPATCH_QUANT_MODE = 2


def _as_tensor_list(value: torch.Tensor | list[torch.Tensor], name: str) -> list[torch.Tensor]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} cannot be empty for CANN MegaMoe.")
        return value
    return [value]


def _normalize_cann_megamoe_activation(activation: str) -> str:
    activation_value = str(getattr(activation, "value", activation)).lower()
    if activation_value in {"silu", "swiglu"}:
        return "swiglu"
    raise ValueError(f"CANN MegaMoe only supports SwiGLU, got activation={activation!r}.")


def _append_cann_megamoe_dummy_tokens(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_active_mask: torch.Tensor | None,
    num_experts: int,
    ep_rank_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Route zero-weight rows from EP rank 0 so every expert is active."""
    num_topk = int(topk_ids.shape[-1])
    dummy_token_capacity = get_cann_megamoe_dummy_token_capacity(num_experts, num_topk)
    total_dummy_routes = dummy_token_capacity * num_topk
    dummy_topk_ids = torch.arange(total_dummy_routes, dtype=topk_ids.dtype, device=topk_ids.device)
    dummy_topk_ids = dummy_topk_ids.remainder(num_experts).view(dummy_token_capacity, num_topk)
    dummy_hidden_states = torch.zeros(
        (dummy_token_capacity, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    dummy_topk_weights = torch.zeros(
        (dummy_token_capacity, num_topk),
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )

    original_num_tokens = int(hidden_states.shape[0])
    hidden_states = torch.cat((hidden_states, dummy_hidden_states), dim=0)
    topk_ids = torch.cat((topk_ids, dummy_topk_ids), dim=0)
    topk_weights = torch.cat((topk_weights, dummy_topk_weights), dim=0)
    if x_active_mask is None:
        x_active_mask = torch.ones(original_num_tokens, dtype=torch.int8, device=hidden_states.device)
    dummy_mask = torch.full(
        (dummy_token_capacity,),
        1 if ep_rank_id == 0 else 0,
        dtype=x_active_mask.dtype,
        device=x_active_mask.device,
    )
    x_active_mask = torch.cat((x_active_mask, dummy_mask), dim=0)
    return hidden_states, topk_ids, topk_weights, x_active_mask, original_num_tokens

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}


def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
    return _MoECommMethods.get(moe_comm_type)


def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
        _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
        _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)
    else:
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None
    swiglu_limit: float = 0.0


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.uint8,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )

        mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        self._cann_megamoe_ops = None
        self._cann_symm_buffers = {}
        self._cann_megamoe_call_index = 0
        self._cann_megamoe_last_contract_signature = None
        self._cann_megamoe_contract_check = os.getenv("VLLM_ASCEND_MEGAMOE_CONTRACT_CHECK", "0") == "1"
        self._cann_megamoe_sync_after_call = os.getenv("VLLM_ASCEND_MEGAMOE_SYNC_AFTER_CALL", "0") == "1"
        enable_fused_mc2 = get_ascend_config().enable_fused_mc2
        if enable_fused_mc2 == _DISPATCH_FFN_COMBINE_MODE:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None
        if enable_fused_mc2 == _CANN_MEGAMOE_MODE and get_ascend_device_type() == AscendDeviceType.A2:
            self._load_cann_megamoe_ops()

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _load_cann_megamoe_ops(self):
        if self._cann_megamoe_ops is None:
            try:
                module = importlib.import_module(_CANN_MEGAMOE_MODULE_NAME)
                self._cann_megamoe_ops = (
                    module.get_mega_moe_ccl_buffer_size,
                    module.get_symm_buffer_for_mega_moe,
                    module.mega_moe,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import CANN MegaMoe APIs. Ensure the image contains "
                    "cann_ops_transformer and the CANN environment is sourced."
                ) from exc
        return self._cann_megamoe_ops

    def _get_cann_symm_buffer(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
        weight1: list[torch.Tensor],
        weight2: list[torch.Tensor],
    ):
        get_buffer_size, get_symm_buffer, _ = self._load_cann_megamoe_ops()
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)

        if fused_experts_input.dynamic_eplb or fused_experts_input.routing.global_redundant_expert_num:
            raise RuntimeError("CANN MegaMoe mode 2 does not support EPLB or redundant experts yet.")

        group = get_mc2_group().device_group
        ep_world_size = int(self.token_dispatcher.ep_world_size)
        num_experts = int(self.moe_config.num_experts)
        if num_experts % ep_world_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by ep_world_size={ep_world_size}.")
        num_local_experts = num_experts // ep_world_size
        if len(weight1) != num_local_experts or len(weight2) != num_local_experts:
            raise ValueError(
                "CANN MegaMoe requires one weight tensor per local expert: "
                f"expected={num_local_experts}, w1={len(weight1)}, w2={len(weight2)}."
            )

        base_num_max_tokens_per_rank = int(self.token_dispatcher.max_num_tokens_per_rank)
        num_topk = int(topk_ids.shape[-1])
        num_max_tokens_per_rank = base_num_max_tokens_per_rank + get_cann_megamoe_dummy_token_capacity(
            num_experts,
            num_topk,
        )
        if num_max_tokens_per_rank < 1 or num_max_tokens_per_rank > 4096:
            raise ValueError(
                "CANN MegaMoe requires num_max_tokens_per_rank in [1, 4096], got "
                f"{num_max_tokens_per_rank}."
            )

        hidden = int(self.moe_config.hidden_dim)
        intermediate_hidden = int(self.moe_config.intermediate_size_per_partition)
        max_recv_token_num = resolve_cann_megamoe_max_recv_tokens(
            num_max_tokens_per_rank,
            ep_world_size,
            num_topk,
            num_local_experts,
        )
        required_buffer_mb = int(
            get_buffer_size(
                ep_world_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                max_recv_token_num=max_recv_token_num,
                dispatch_quant_mode=_CANN_MEGAMOE_DISPATCH_QUANT_MODE,
                dispatch_quant_out_dtype=torch.int8,
            )
        )
        if required_buffer_mb <= 0:
            raise RuntimeError(f"get_mega_moe_ccl_buffer_size returned invalid size {required_buffer_mb} MB.")

        key = (
            id(group),
            num_experts,
            num_max_tokens_per_rank,
            max_recv_token_num,
            num_topk,
            hidden,
            intermediate_hidden,
            required_buffer_mb,
        )
        if key not in self._cann_symm_buffers:
            logger.info(
                "CANN MegaMoe sym-buffer alloc: ep_rank=%s ep_world=%d "
                "experts=%d max_tokens_per_rank=%d dummy_tokens=%d "
                "max_recv_tokens=%d buffer=%d MB",
                self.token_dispatcher.ep_rank_id,
                ep_world_size,
                num_experts,
                num_max_tokens_per_rank,
                get_cann_megamoe_dummy_token_capacity(num_experts, num_topk),
                max_recv_token_num,
                required_buffer_mb,
            )
            sym_buffer = get_symm_buffer(
                group,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                max_recv_token_num=max_recv_token_num,
                dispatch_quant_mode=_CANN_MEGAMOE_DISPATCH_QUANT_MODE,
                dispatch_quant_out_dtype=torch.int8,
            )
            actual_buffer_bytes = int(getattr(sym_buffer, "ccl_buffer_size", 0))
            required_buffer_bytes = required_buffer_mb * 1024 * 1024
            if actual_buffer_bytes < required_buffer_bytes:
                raise RuntimeError(
                    "CANN MegaMoe HCCL buffer is smaller than required: "
                    f"actual={actual_buffer_bytes} bytes, required={required_buffer_bytes} bytes."
                )
            self._cann_symm_buffers[key] = sym_buffer
        return self._cann_symm_buffers[key]

    def _check_cann_megamoe_contract(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x_active_mask: torch.Tensor,
    ) -> tuple[int, bool]:
        """Validate the documented A2 cross-rank MegaMoe call contract."""
        call_index = self._cann_megamoe_call_index
        self._cann_megamoe_call_index += 1
        if not self._cann_megamoe_contract_check:
            return call_index, False

        if hidden_states.dim() != 2:
            raise ValueError(f"CANN MegaMoe x must be 2D, got shape={tuple(hidden_states.shape)}.")
        num_tokens, hidden = map(int, hidden_states.shape)
        if topk_ids.dim() != 2 or topk_weights.dim() != 2:
            raise ValueError(
                "CANN MegaMoe topk tensors must be 2D: "
                f"ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}."
            )
        if topk_ids.shape != topk_weights.shape or int(topk_ids.shape[0]) != num_tokens:
            raise ValueError(
                "CANN MegaMoe x/topk shapes disagree: "
                f"x={tuple(hidden_states.shape)}, ids={tuple(topk_ids.shape)}, "
                f"weights={tuple(topk_weights.shape)}."
            )
        if x_active_mask.dtype != torch.int8 or tuple(x_active_mask.shape) != (num_tokens,):
            raise ValueError(
                "CANN MegaMoe x_active_mask must be INT8 with shape (num_tokens,): "
                f"dtype={x_active_mask.dtype}, shape={tuple(x_active_mask.shape)}, num_tokens={num_tokens}."
            )

        num_topk = int(topk_ids.shape[1])
        num_experts = int(self.moe_config.num_experts)
        invalid_ids = int(((topk_ids < 0) | (topk_ids >= num_experts)).sum().item())
        sorted_ids = torch.sort(topk_ids, dim=1).values
        duplicate_rows = int((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(dim=1).sum().item())
        if invalid_ids or duplicate_rows:
            raise ValueError(
                "CANN MegaMoe topk_ids violate the documented range/uniqueness contract: "
                f"invalid_ids={invalid_ids}, duplicate_rows={duplicate_rows}, experts={num_experts}."
            )

        active_tokens = int(x_active_mask.sum(dtype=torch.int64).item())
        layer_index = int(getattr(_EXTRA_CTX, "moe_layer_index", -1))
        ep_rank = int(self.token_dispatcher.ep_rank_id)
        ep_world = int(self.token_dispatcher.ep_world_size)
        local_contract = torch.tensor(
            [call_index, layer_index, num_tokens, hidden, num_topk, int(x_active_mask.numel())],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        gathered = torch.empty(local_contract.numel() * ep_world, dtype=torch.int64, device=hidden_states.device)

        signature = (num_tokens, active_tokens)
        should_trace = call_index < 4 or call_index % 64 == 0 or signature != self._cann_megamoe_last_contract_signature
        if should_trace:
            logger.warning(
                "CANN MegaMoe contract before: ep_rank=%d call=%d layer=%d tokens=%d active=%d "
                "hidden=%d topk=%d",
                ep_rank,
                call_index,
                layer_index,
                num_tokens,
                active_tokens,
                hidden,
                num_topk,
            )

        torch.distributed.all_gather_into_tensor(
            gathered,
            local_contract,
            group=get_mc2_group().device_group,
        )
        contracts = gathered.view(ep_world, -1)
        reference = contracts[0]
        mismatch = (contracts != reference).any(dim=1)
        if bool(mismatch.any().item()):
            contract_rows = contracts.cpu().tolist()
            raise RuntimeError(
                "CANN MegaMoe A2 requires identical call order and num_tokens on every EP rank; "
                f"contracts={contract_rows}. Columns are "
                "[call_index, layer_index, num_tokens, hidden, num_topk, mask_length]."
            )

        self._cann_megamoe_last_contract_signature = signature
        return call_index, should_trace

    def _apply_cann_megamoe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
    ):
        if fused_experts_input.quant.quant_type != QuantType.W8A8:
            raise RuntimeError(
                "CANN MegaMoe mode 2 currently supports only W8A8 routed experts, got "
                f"{fused_experts_input.quant.quant_type}."
            )
        if fused_experts_input.hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "CANN MegaMoe A8W8 requires BF16 hidden states, got "
                f"{fused_experts_input.hidden_states.dtype}."
            )
        if fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None:
            raise ValueError("CANN MegaMoe W8A8 requires both w1_scale and w2_scale.")

        weight1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
        weight2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
        weight_scales1 = [
            tensor.squeeze(0).contiguous() if tensor.dim() == 2 and tensor.shape[0] == 1 else tensor.contiguous()
            for tensor in _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale")
        ]
        weight_scales2 = [
            tensor.squeeze(0).contiguous() if tensor.dim() == 2 and tensor.shape[0] == 1 else tensor.contiguous()
            for tensor in _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale")
        ]
        if len(weight_scales1) != len(weight1) or len(weight_scales2) != len(weight2):
            raise ValueError(
                "CANN MegaMoe requires one scale tensor per expert weight: "
                f"w1={len(weight1)}, w1_scale={len(weight_scales1)}, "
                f"w2={len(weight2)}, w2_scale={len(weight_scales2)}."
            )
        if any(weight.dtype != torch.int8 for weight in (*weight1, *weight2)):
            raise ValueError("CANN MegaMoe W8A8 requires INT8 expert weights.")
        valid_scale_dtypes = {torch.int64, torch.uint64}
        if any(scale.dtype not in valid_scale_dtypes for scale in (*weight_scales1, *weight_scales2)):
            raise ValueError("CANN MegaMoe W8A8 requires UINT64-compatible expert scales.")

        sym_buffer = self._get_cann_symm_buffer(fused_experts_input, topk_ids, weight1, weight2)
        _, _, mega_moe = self._load_cann_megamoe_ops()
        x_active_mask = None
        if fused_experts_input.routing.mc2_mask is not None:
            raw_mask = fused_experts_input.routing.mc2_mask
            x_active_mask = (
                raw_mask.contiguous() if raw_mask.dtype == torch.int8 else raw_mask.to(torch.int8).contiguous()
            )

        hidden_states, topk_ids, topk_weights, x_active_mask, original_num_tokens = (
            _append_cann_megamoe_dummy_tokens(
                fused_experts_input.hidden_states,
                topk_ids,
                fused_experts_input.topk_weights,
                x_active_mask,
                int(self.moe_config.num_experts),
                int(self.token_dispatcher.ep_rank_id),
            )
        )
        call_index, should_trace = self._check_cann_megamoe_contract(
            hidden_states,
            topk_ids,
            topk_weights,
            x_active_mask,
        )
        activation_clamp = fused_experts_input.swiglu_limit if fused_experts_input.swiglu_limit > 0 else None
        output, expert_tokens = mega_moe(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            weight1,
            weight2,
            sym_buffer,
            l1_weights_sf=weight_scales1,
            l2_weights_sf=weight_scales2,
            x_active_mask=x_active_mask,
            activation=_normalize_cann_megamoe_activation(fused_experts_input.activation),
            activation_clamp=activation_clamp,
        )
        if self._cann_megamoe_sync_after_call:
            torch.npu.synchronize()
        if should_trace:
            logger.warning(
                "CANN MegaMoe contract after: ep_rank=%d call=%d output_tokens=%d",
                self.token_dispatcher.ep_rank_id,
                call_index,
                int(output.shape[0]),
            )
        return output[:original_num_tokens], expert_tokens

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
            "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."
        )

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        # Apply log2phy if needed
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if get_ascend_config().enable_fused_mc2 == _DISPATCH_FFN_COMBINE_MODE:
            assert not (
                fused_experts_input.weights.w1_scale_bias is None or fused_experts_input.weights.w2_scale_bias is None
            ), "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."

            out = torch.empty_like(fused_experts_input.hidden_states)
            torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                x=fused_experts_input.hidden_states,
                weight1=fused_experts_input.weights.w1,
                weight2=fused_experts_input.weights.w2,
                expert_idx=topk_ids,
                scale1=fused_experts_input.weights.w1_scale,
                scale2=fused_experts_input.weights.w2_scale,
                bias1=fused_experts_input.weights.w1_scale_bias,
                bias2=fused_experts_input.weights.w2_scale_bias,
                probs=fused_experts_input.topk_weights.to(torch.float32),
                group=self.token_dispatcher.moe_all_to_all_group_name,
                max_output_size=get_ascend_config().mega_moe_max_tokens,
                swiglu_limit=fused_experts_input.swiglu_limit,
                x_active_mask=fused_experts_input.routing.mc2_mask,
                out=out,
                expert_token_nums=self.expert_token_nums,
            )
            expert_tokens = self.expert_token_nums
        elif get_ascend_config().enable_fused_mc2 == _CANN_MEGAMOE_MODE:
            if get_ascend_device_type() == AscendDeviceType.A2:
                out, expert_tokens = self._apply_cann_megamoe(fused_experts_input, topk_ids)
            else:
                assert fused_experts_input.routing.expert_map is not None, "expert_map cannot be None."
                out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    expert_ids=topk_ids,
                    gmm1_permuted_weight=fused_experts_input.weights.w1,
                    gmm1_permuted_weight_scale=fused_experts_input.weights.w1_scale,
                    gmm2_weight=fused_experts_input.weights.w2,
                    gmm2_weight_scale=fused_experts_input.weights.w2_scale,
                    expert_smooth_scales=None,
                    expert_scales=fused_experts_input.topk_weights.to(torch.float32),
                    group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                    ep_rank_size=self.token_dispatcher.ep_world_size,
                    ep_rank_id=self.token_dispatcher.ep_rank_id,
                    moe_expert_num=self.moe_config.num_experts,
                    global_bs=self.token_dispatcher.global_bs,
                )
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out, expert_tokens=expert_tokens, swiglu_limit=fused_experts_input.swiglu_limit
        )
