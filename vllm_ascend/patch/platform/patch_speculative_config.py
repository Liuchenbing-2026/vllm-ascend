import typing
from typing import TYPE_CHECKING, Any, Optional

from vllm.config.speculative import SpeculativeConfig
from vllm.logger import logger
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import vllm.model_executor.layers.quantization as me_quant
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

    me_quant = LazyLoader("model_executor", globals(), "vllm.model_executor.layers.quantization")


def _register_dspark_speculative_method() -> None:
    """Teach SpeculativeConfig that method="dspark" is valid on vLLM 0.23."""
    from vllm.config import speculative as spec_mod

    existing = typing.get_args(spec_mod.SpeculativeMethod)
    if "dspark" in existing:
        return

    new_literal = typing.Literal[existing + ("dspark",)]  # type: ignore[valid-type]
    new_annotation = Optional[new_literal]
    spec_mod.SpeculativeMethod = new_literal
    SpeculativeConfig.__annotations__["method"] = new_annotation

    dc_fields = getattr(SpeculativeConfig, "__dataclass_fields__", None)
    if dc_fields is not None and "method" in dc_fields:
        dc_fields["method"].type = new_annotation

    try:
        from pydantic.dataclasses import rebuild_dataclass
    except Exception as e:
        logger.warning("Cannot import rebuild_dataclass (%s); dspark method may not validate.", e)
        return

    try:
        rebuild_dataclass(SpeculativeConfig, force=True)  # type: ignore[arg-type]
    except Exception as e:
        logger.warning("rebuild_dataclass(SpeculativeConfig) failed (%s); dspark method may not validate.", e)
        return

    try:
        from vllm.config.vllm import VllmConfig

        rebuild_dataclass(VllmConfig, force=True)  # type: ignore[arg-type]
    except Exception as e:
        logger.debug("rebuild_dataclass(VllmConfig) failed (%s); nested spec validation may be stale.", e)


def _patch_dspark_post_init() -> None:
    """Let dspark reuse the draft-model validation path, then restore dspark."""
    original_post_init = SpeculativeConfig.__post_init__
    if getattr(original_post_init, "_vllm_ascend_dspark_patched", False):
        return

    def _patched_post_init(self, *args, **kwargs):
        is_dspark = getattr(self, "method", None) == "dspark"
        if is_dspark:
            target_cfg = getattr(self, "target_model_config", None)
            if getattr(self, "model", None) is None and target_cfg is not None:
                self.model = target_cfg.model
                if not getattr(self, "quantization", None):
                    self.quantization = getattr(target_cfg, "quantization", None)
            self.method = "draft_model"
        try:
            result = original_post_init(self, *args, **kwargs)
        finally:
            if is_dspark:
                self.method = "dspark"
                self.parallel_drafting = True
                draft_hf_config = getattr(
                    getattr(self, "draft_model_config", None), "hf_config", None
                )
                target_layer_ids = list(
                    getattr(draft_hf_config, "dspark_target_layer_ids", None) or []
                )
                if draft_hf_config is not None and target_layer_ids:
                    draft_hf_config.update(
                        {
                            "model_type": "deepseek_v4_dspark",
                            "n_predict": len(target_layer_ids),
                            "architectures": ["DeepSeekV4DSpark"],
                        }
                    )
        return result

    _patched_post_init._vllm_ascend_dspark_patched = True  # type: ignore[attr-defined]
    SpeculativeConfig.__post_init__ = _patched_post_init  # type: ignore[assignment]


def _patch_dspark_use_eagle() -> None:
    original_use_eagle = getattr(SpeculativeConfig, "use_eagle", None)
    if original_use_eagle is None or getattr(original_use_eagle, "_vllm_ascend_dspark_patched", False):
        return

    def _patched_use_eagle(self) -> bool:
        if getattr(self, "method", None) == "dspark":
            return True
        return original_use_eagle(self)

    _patched_use_eagle._vllm_ascend_dspark_patched = True  # type: ignore[attr-defined]
    SpeculativeConfig.use_eagle = _patched_use_eagle  # type: ignore[assignment]


def _patch_dspark_vllm_post_init() -> None:
    """Keep hybrid KV enabled for DSpark across repeated VllmConfig post-init.

    vLLM runs VllmConfig.__post_init__ in the API process and again after the
    EngineCore handshake. DSpark intentionally reuses EAGLE-like scheduling via
    SpeculativeConfig.use_eagle(), which trips vLLM's generic
    chunked-local-attention + EAGLE guard. On the second post-init the auto
    enabled hybrid-KV state looks like an explicit user enable and raises.
    """
    try:
        from vllm.config.vllm import VllmConfig
    except Exception as e:
        logger.warning("Cannot patch VllmConfig.__post_init__ for dspark (%s).", e)
        return

    original_post_init = VllmConfig.__post_init__
    if getattr(original_post_init, "_vllm_ascend_dspark_hybrid_kv_patched", False):
        return

    def _patched_post_init(self, *args, **kwargs):
        speculative_config = getattr(self, "speculative_config", None)
        scheduler_config = getattr(self, "scheduler_config", None)
        is_dspark = getattr(speculative_config, "method", None) == "dspark"
        original_disable_hybrid = (
            getattr(scheduler_config, "disable_hybrid_kv_cache_manager", None)
            if scheduler_config is not None
            else None
        )

        # If a previous post-init auto-enabled hybrid KV, make the next
        # post-init go through the automatic branch again instead of treating
        # it as a user-specified "--no-disable-hybrid-kv-cache-manager".
        if is_dspark and scheduler_config is not None and original_disable_hybrid is False:
            scheduler_config.disable_hybrid_kv_cache_manager = None

        try:
            result = original_post_init(self, *args, **kwargs)
        finally:
            if is_dspark and scheduler_config is not None and original_disable_hybrid is not True:
                if getattr(scheduler_config, "disable_hybrid_kv_cache_manager", None) is not False:
                    logger.warning_once("Keeping hybrid KV cache manager enabled for DSpark.")
                scheduler_config.disable_hybrid_kv_cache_manager = False
        return result

    _patched_post_init._vllm_ascend_dspark_hybrid_kv_patched = True  # type: ignore[attr-defined]
    VllmConfig.__post_init__ = _patched_post_init  # type: ignore[assignment]


def _is_dspark_v4_checkpoint(hf_config: PretrainedConfig) -> bool:
    if getattr(hf_config, "model_type", None) != "deepseek_v4":
        return False
    return bool(getattr(hf_config, "dspark_block_size", 0)) and bool(
        getattr(hf_config, "dspark_target_layer_ids", None))


def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    initial_architecture = hf_config.architectures[0]
    if hf_config.model_type == "deepseek_v4" and _is_dspark_v4_checkpoint(hf_config):
        validation_n_predict = getattr(hf_config, "num_nextn_predict_layers", None) or 1
        updates = {
            "model_type": "deepseek_v4_dspark",
            "n_predict": validation_n_predict,
            "architectures": ["DeepSeekV4DSpark"],
        }
        dflash_config = getattr(hf_config, "dflash_config", None)
        dflash_mask_token = (
            dflash_config.get("mask_token_id")
            if isinstance(dflash_config, dict)
            else getattr(dflash_config, "mask_token_id", None)
        )
        if (
            getattr(hf_config, "pard_token", None) is None
            and getattr(hf_config, "ptd_token_id", None) is None
            and dflash_mask_token is None
        ):
            eos = getattr(hf_config, "eos_token_id", None)
            updates["ptd_token_id"] = int(eos[0] if isinstance(eos, (list, tuple)) else eos or 1)
        hf_config.update(updates)
        return hf_config

    if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "deepseek_v4", "glm_moe_dsa"):
        target_model_type = hf_config.model_type
        hf_config.model_type = "deepseek_mtp"
    if hf_config.model_type == "deepseek_mtp":
        if target_model_type == "deepseek_v4":
            hf_config.update({"architectures": ["DeepSeekV4MTPModel"]})
        else:
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]})
    if hf_config.model_type in ("pangu_ultra_moe"):
        hf_config.model_type = "pangu_ultra_moe_mtp"
    if hf_config.model_type == "pangu_ultra_moe_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]})

    if hf_config.architectures[0] == "MiMoForCausalLM":
        hf_config.model_type = "mimo_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"],
            }
        )

    if hf_config.architectures[0] == "Glm4MoeForCausalLM":
        hf_config.model_type = "glm4_moe_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Glm4MoeMTPModel"],
            }
        )

    if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
        hf_config.model_type = "glm4_moe_lite_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["Glm4MoeLiteMTPModel"],
            }
        )

    if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
        hf_config.model_type = "glm_ocr_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["GlmOcrMTPModel"],
            }
        )

    if hf_config.model_type == "ernie4_5_moe":
        hf_config.model_type = "ernie_mtp"
    if hf_config.model_type == "ernie_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["ErnieMTPModel"]})

    if (
        hf_config.model_type == "nemotron_h"
        and hasattr(hf_config, "num_nextn_predict_layers")
        and hf_config.num_nextn_predict_layers > 0
    ):
        # Check if this is an MTP variant
        hf_config.model_type = "nemotron_h_mtp"
    if hf_config.model_type == "nemotron_h_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]})

    if hf_config.model_type == "qwen3_next":
        hf_config.model_type = "qwen3_next_mtp"
    if hf_config.model_type == "qwen3_next_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]})

    if hf_config.model_type == "exaone_moe":
        hf_config.model_type = "exaone_moe_mtp"
    if hf_config.model_type == "exaone_moe_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]})

    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
        is_moe = hf_config.model_type == "qwen3_5_moe"
        hf_config.model_type = "qwen3_5_mtp"
        n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
            }
        )
    if hf_config.model_type == "longcat_flash":
        hf_config.model_type = "longcat_flash_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]})

    if hf_config.model_type == "step3p5":
        hf_config.model_type = "step3p5_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

    if initial_architecture == "MistralLarge3ForCausalLM":
        hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

    return hf_config


SpeculativeConfig.hf_config_override = hf_config_override
_register_dspark_speculative_method()
_patch_dspark_post_init()
_patch_dspark_use_eagle()
_patch_dspark_vllm_post_init()
