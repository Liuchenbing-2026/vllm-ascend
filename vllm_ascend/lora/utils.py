import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    VocabParallelEmbeddingWithLoRA,
)

from vllm_ascend.ops.linear import (
    AscendColumnParallelLinear,
    AscendMergedColumnParallelLinear,
    AscendQKVParallelLinear,
    AscendRowParallelLinear,
)
from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding


class AscendColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendColumnParallelLinear


class AscendMergedColumnParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendMergedColumnParallelLinear


class AscendRowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendRowParallelLinear


class AscendVocabParallelEmbeddingWithLoRA(VocabParallelEmbeddingWithLoRA):
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendVocabParallelEmbedding


class AscendQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
    @classmethod
    # NOTE: no @_not_fully_sharded_can_replace here, on purpose. The dense
    # Ascend wrappers (Column/MergedColumn/Row) above already omit it, so they
    # match in both plain and fully-sharded modes. Keeping the decorator only
    # on QKV made it the odd one out: under --fully-sharded-loras the decorator
    # forces can_replace_layer to return False, and since there is no
    # AscendQKVParallelLinearWithShardedLoRA to pick the layer up, qkv_proj
    # silently loses its LoRA delta while the MLP still gets it. Dropping the
    # decorator lets qkv_proj be wrapped in fully-sharded mode too (using the
    # non-sharded impl, numerically correct).
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
    # See AscendQKVParallelLinearWithLoRA above: decorator intentionally
    # dropped so merged qkv_proj is also wrapped under --fully-sharded-loras.
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


def refresh_all_lora_classes():
    vllm.lora.utils._all_lora_classes.add(AscendColumnParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendMergedColumnParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendRowParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendVocabParallelEmbeddingWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendQKVParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendMergedQKVParallelLinearWithLoRA)
