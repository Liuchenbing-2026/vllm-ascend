from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v4 import DeepseekV4FP8Config

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, (MoERunner, RoutedExperts))


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(Fp8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}
        self.is_per_tensor_fp8 = self.weight_block_size is None
        self.mistral4_dynamic_channelwise = False

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        module_prefixes = [f"{name}." for name in self.ignored_layers]
        mapped_prefixes = hf_to_vllm_mapper.apply_list(module_prefixes)
        self.ignored_layers = [name.removesuffix(".") for name in mapped_prefixes]

    @staticmethod
    def _is_mistral4_model() -> bool:
        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return False
        text_config = getattr(hf_config, "text_config", hf_config)
        return getattr(text_config, "model_type", None) == "mistral4"

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        self.mistral4_dynamic_channelwise = self.is_per_tensor_fp8 and self._is_mistral4_model()
        if not self.mistral4_dynamic_channelwise:
            raise NotImplementedError

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            layer.ascend_quant_method = FP8_METHOD
            scheme_class = get_scheme_class("W8A8FP8_DYNAMIC", "linear")
            assert scheme_class is not None
            logger.warning_once(
                "A2 per-tensor FP8 keeps serialized weights but uses dynamic "
                "activation quantization; checkpoint static activation scales "
                "are not consumed."
            )
            return AscendLinearMethod(scheme_class())

        if _is_fused_moe_layer(layer):
            layer.ascend_quant_method = FP8_METHOD
            scheme_class = get_scheme_class("W8A8FP8_DYNAMIC", "moe")
            assert scheme_class is not None
            return AscendFusedMoEMethod(
                scheme_class(),
                layer.moe_config,
                tid2eid=tid2eid,
            )
        return None


@register_quantization_config("deepseek_v4_fp8")
class AscendDeepseekV4FP8Config(DeepseekV4FP8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            scheme_class = get_scheme_class(FP8_METHOD, "ds_linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_linear"
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if _is_fused_moe_layer(layer):
            if self.expert_dtype == "fp4":
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w4a8_moe")
                assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_w4a8_moe"
            else:
                raise NotImplementedError
            quant_method = AscendFusedMoEMethod(scheme_class(), layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None
