import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace

from vllm_ascend.ops.linear import (
    AscendQKVParallelLinear,
)


class AscendQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


class AscendMergedQKVParallelLinearWithLoRA(MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


class AscendMergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


class AscendQKVParallelLinearWithShardedLoRA(QKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


def _strip_gdn_inproj_from_lora_mapping(model=None):
    """Qwen3.5-MoE is a hybrid GDN (linear-attention) + MoE model. The
    GatedDeltaNet ``in_proj_ba`` (beta/alpha gates) has a tiny output dim;
    the punica sgmv_expand kernel rejects it ("hidden in should be smaller
    than hidden out") when the LoRA manager builds dummy adapters for every
    packed module during profile_run -- even though no adapter targets it.

    The *class-level* ``packed_modules_mapping`` is what the LoRA manager
    consumes; base-weight loading uses a separate tuple mapping on the inner
    model, so removing the in_proj keys here only disables LoRA on those
    layers (which nothing adapts) without affecting weight loading.
    """
    # Cheap no-op for unrelated models (avoids importing qwen3_5).
    if model is not None and "qwen3_5" not in type(model).__module__:
        return
    try:
        from vllm.model_executor.models import qwen3_5 as q
    except Exception:
        return
    for cls_name in (
        "Qwen3_5ForCausalLMBase",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        cls = getattr(q, cls_name, None)
        if cls is None or "packed_modules_mapping" not in cls.__dict__:
            continue
        pmm = dict(cls.packed_modules_mapping or {})
        if any(k in pmm for k in ("in_proj_qkvz", "in_proj_ba")):
            for k in ("in_proj_qkvz", "in_proj_ba"):
                pmm.pop(k, None)
            cls.packed_modules_mapping = pmm


def refresh_all_lora_classes():
    ascend_classes = (
        AscendQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithShardedLoRA,
        AscendQKVParallelLinearWithShardedLoRA,
    )

    # MoE LoRA: drop upstream Triton-based wrappers (they assert on TritonExperts
    # in __init__ which does not exist on Ascend) and register Ascend variants.
    # Imported lazily to avoid pulling in torch_npu at module-import time.
    from vllm_ascend.lora.fused_moe import (
        AscendFusedMoE3DWithLoRA,
        AscendFusedMoEWithLoRA,
    )

    moe_ascend_classes = (
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )

    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Filter out upstream Triton-based MoE wrappers and append the Ascend classes.
    vllm.lora.utils._all_lora_classes = (
        tuple(cls for cls in vllm.lora.utils._all_lora_classes if cls not in (FusedMoEWithLoRA, FusedMoE3DWithLoRA))
        + ascend_classes
        + moe_ascend_classes
    )
