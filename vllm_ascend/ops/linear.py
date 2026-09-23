# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""
To customize linear communication groups or forward of classes in this file,
extend new linear operations in linear_op.py.
The classes in this file should not be modified, including AscendQKVParallelLinear,
AscendMergedColumnParallelLinear, AscendMergedColumnParallelLinear,
AscendRowParallelLinear and AscendColumnParallelLinear.
"""

import os

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config
from vllm.distributed import divide, split_tensor_along_last_dim
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.model_executor.layers.linear import (  # noqa
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.linear_op import get_parallel_op, get_replicated_op
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_sp,
    get_ascend_device_type,
    is_310p,
    maybe_trans_nz,
)


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output_shape = (x.shape[0], weight.shape[0])
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="unquantized_gemm",
    op_func=unquantized_gemm,
    fake_impl=unquantized_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _should_keep_nd_for_310p_weight(weight: torch.Tensor) -> bool:
    return is_310p() and weight.ndim >= 2 and (weight.shape[-1] == 1 or weight.shape[-2] == 1)


def _keep_nd_scalar_weight(weight: torch.Tensor) -> bool:
    """True when the weight-side matrix has a unit inner or outer dim.

    FRACTAL_NZ matmul is rejected outright for that shape -- the ACLNN error is
    ``AclNN_Parameter_Error(EZ1001): Not supported mat2 n = 1 or k = 1 when
    format is FRACTAL_NZ`` -- which is exactly the shape of scalar gates such as
    Qwen MoE's ``shared_expert_gate`` (``Linear(hidden, 1)``).  The 310P branch
    above already keeps those in ND; with ``weight_nz_mode=2`` every other
    platform needs the same exemption or start-up dies inside the first
    ``linear`` call.  Staying in ND is a layout decision only: the numerics are
    identical, since ND and NZ are two views of the same matmul.
    """
    return weight.ndim >= 2 and (weight.shape[-1] == 1 or weight.shape[-2] == 1)


class AscendUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Linear method without quantization"""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        keep_nd_weight = _should_keep_nd_for_310p_weight(layer.weight.data) or _keep_nd_scalar_weight(
            layer.weight.data
        )
        # must use fp32 to avoid accuracy degradation in dsv4.
        if getattr(layer, "precast_fp32_weight", False):
            weight_fp32 = layer.weight.data.to(torch.float32)
            layer.weight_fp32 = weight_fp32 if keep_nd_weight else maybe_trans_nz(weight_fp32)
        if "conv1d" not in layer.prefix:
            # torch_npu rejects FRACTAL_NZ matmul when the weight-side matrix
            # has n=1 or k=1. Keep scalar gates such as Qwen MoE's
            # shared_expert_gate in ND format on every platform.
            if not keep_nd_weight:
                layer.weight.data = maybe_trans_nz(layer.weight.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.unquantized_gemm(x, layer.weight, bias)


# TODO(realliujiaxu): Remove this class after linear of vllm supports custom comm group
class AscendLinearBase(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = AscendUnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp

    def update_param_tp_status(self):
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


class AscendQKVParallelLinear(QKVParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.custom_op, _, tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp,
            prefix,
            self,
            "column",
            output_size=sum(output_sizes),
        )
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.output_sizes = output_sizes
        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


def _mm_all_reduce_enabled() -> bool:
    return os.environ.get("QWEN38_MM_AR", "0") == "1"


_mm_ar_warned = False

_hcom_name_logged: set[int] = set()


def _resolve_tp_hcom_name(tp_rank: int) -> str | None:
    """Resolve the TP HCCL communicator name **eagerly**, once per rank.

    This must never run inside a compiled region.  ``ProcessGroup._get_backend``
    is a C++ method that Dynamo cannot trace (``Unsupported method call``) and
    the capture used here takes no graph breaks, so either calling it in the
    graph or wrapping it in ``torch.compiler.disable`` aborts the whole
    npugraph_ex capture -- both have been observed on this build.

    The name is a per-rank constant, so it is resolved where the layer is
    *constructed* (always eager) and read back inside the graph as a plain
    attribute.  Returns ``None`` -- and the caller falls back to the stock
    matmul + all-reduce -- when the group is not up yet.
    """
    try:
        name = get_tp_group().device_group._get_backend(torch.device("npu")).get_hccl_comm_name(tp_rank)
    except Exception:  # noqa: BLE001 - unresolved name just disables the fusion
        logger.warning("QWEN38_MM_AR: could not resolve HCCL comm name for tp_rank=%s", tp_rank, exc_info=True)
        return None
    if tp_rank not in _hcom_name_logged:
        _hcom_name_logged.add(tp_rank)
        logger.info("QWEN38_MM_AR: TP HCCL comm name for tp_rank=%s is %s", tp_rank, name)
    return name


def _maybe_fused_mm_all_reduce(layer, input_: torch.Tensor) -> torch.Tensor | None:
    """Fuse the row-parallel matmul with its TP all-reduce (MC2).

    ``Y = all_reduce(X_i @ W_i^T)`` collapses into a single
    ``torch_npu.npu_mm_all_reduce_base`` call that pipelines the matmul against the
    HCCL all-reduce instead of running them back to back on one stream.  This is
    the *only* place the non sequence-parallel row-parallel layers reduce, so it
    covers every ``o_proj`` / ``down_proj`` of the model.

    Returns ``None`` when the fused path is disabled or does not apply, in which
    case the caller keeps the stock matmul + ``all_reduce``.
    """
    if not _mm_all_reduce_enabled():
        return None
    if layer.tp_size <= 1 or not layer.reduce_results:
        return None
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        return None
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return None

    if layer.input_is_parallel:
        x = input_
    else:
        x = split_tensor_along_last_dim(input_, num_partitions=layer.tp_size)[layer.tp_rank].contiguous()
    if x.dim() != 2:
        return None
    try:
        min_tokens = int(os.environ.get("QWEN38_MM_AR_MIN_TOKENS", "512"))
    except ValueError:
        min_tokens = 512
    if x.shape[0] < min_tokens:
        return None

    from vllm_ascend.device.device_op import DeviceOperator

    # Read the communicator name that was resolved at construction time.  A
    # call of any kind here would be traced into the graph, and this capture
    # tolerates no graph break -- see _resolve_tp_hcom_name.
    hcom_name = getattr(layer, "_mm_ar_hcom_name", None)
    if hcom_name is None:
        return None

    try:
        out = DeviceOperator.npu_mm_all_reduce_base(
            x.contiguous(),
            weight.t(),
            hcom_name,
            reduce_op="sum",
            comm_turn=0,
        )
    except Exception:  # noqa: BLE001 - fall back to the stock unfused path
        global _mm_ar_warned
        if not _mm_ar_warned:
            _mm_ar_warned = True
            logger.warning("QWEN38_MM_AR: fused mm+allreduce failed for %s, falling back", layer.prefix, exc_info=True)
        return None

    # Rank 0 normally fuses the bias into its local GEMM before the reduction;
    # after the fused all-reduce the bias has to be added once on every rank.
    if layer.bias is not None and not layer.skip_bias_add:
        out = out + layer.bias.to(out.dtype)
    return out


class AscendRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    # NOTE: Globally unique prefix identifier used in SP scenarios
    unique_prefix_idx = 0

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # TODO(kunpengW-code): Specifying the prefix in linear layers of some models in the vLLM.
        if enable_sp():
            compilation_config = get_current_vllm_config().compilation_config
            unique_prefix = prefix
            if prefix in compilation_config.static_forward_context:
                unique_prefix = f"{prefix}.unique_prefix{AscendRowParallelLinear.unique_prefix_idx}"
                AscendRowParallelLinear.unique_prefix_idx += 1
            self.unique_prefix = unique_prefix
            compilation_config.static_forward_context[unique_prefix] = self

        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "row")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.out_dtype = out_dtype

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the results can lead to incorrect results")

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        self.update_param_tp_status()

        if self.custom_op is not None:
            self.custom_op.update_attrs()

        # QWEN38_MM_AR needs the TP HCCL communicator name of this rank, and it
        # has to be a plain attribute by the time the forward runs inside a
        # captured graph: resolving it lazily would put
        # ``ProcessGroup._get_backend`` into the trace and abort the capture
        # (see _resolve_tp_hcom_name).  Layer construction is always eager, so
        # this is where the name is looked up.  ``None`` (group not up yet, or
        # the switch is off) simply keeps the stock matmul + all-reduce.
        self._mm_ar_hcom_name = _resolve_tp_hcom_name(self.tp_rank) if _mm_all_reduce_enabled() else None

    def forward(
        self,
        input_,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        fused = _maybe_fused_mm_all_reduce(self, input_)
        if fused is not None:
            if not self.return_bias:
                return fused
            return fused, (self.bias if self.skip_bias_add else None)

        return super().forward(input_)


class AscendColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        #
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp,
            prefix,
            self,
            "column",
            output_size=output_size,
        )
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [divide(output_size, self.tp_size) for output_size in self.output_sizes]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        self.update_param_tp_status()

        if self.custom_op is not None:
            self.custom_op.update_attrs()
        self.prefix = prefix
        if "wo_a" in prefix:
            hf_config = get_current_vllm_config().model_config.hf_text_config
            self.n_local_groups = getattr(hf_config, "o_groups", 0) // self.tp_size
            self.o_lora_rank = getattr(hf_config, "o_lora_rank", 0)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if "wo_a" in self.prefix and get_ascend_device_type() != AscendDeviceType.A5:
            if self.weight.ndim == 2:
                super().weight_loader(param, loaded_weight)
                self.weight.data = (
                    self.weight.data.view(self.n_local_groups, self.o_lora_rank, -1).transpose(2, 1).contiguous()
                )
            else:
                # In RL update flows, wo_a can be loaded again after being
                # transformed into [n_local_groups, hidden_size, o_lora_rank].
                shard_size = self.n_local_groups * self.o_lora_rank
                start_idx = self.tp_rank * shard_size
                if loaded_weight.shape[0] != shard_size:
                    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
                loaded_weight = (
                    loaded_weight.view(
                        self.n_local_groups,
                        self.o_lora_rank,
                        -1,
                    )
                    .transpose(2, 1)
                    .contiguous()
                )

                if loaded_weight.shape != self.weight.shape:
                    raise ValueError(
                        f"Unexpected wo_a weight shape {tuple(loaded_weight.shape)}, "
                        f"expected {tuple(self.weight.shape)}"
                    )
                self.weight.data.copy_(loaded_weight)
        else:
            super().weight_loader(param, loaded_weight)


class AscendReplicatedLinear(ReplicatedLinear):
    """Ascend Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_replicated_op(disable_tp, prefix, self)
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        self.update_param_tp_status()

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)
