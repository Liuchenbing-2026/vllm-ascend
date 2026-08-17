# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Any

from vllm.config import VllmConfig


_MISTRAL3_TEXT_ARCHITECTURES = {
    "mistral": "MistralForCausalLM",
    "ministral3": "Ministral3ForCausalLM",
}


def _get_mistral3_text_architectures(text_config: object) -> list[str]:
    model_type = getattr(text_config, "model_type", None)
    try:
        return [_MISTRAL3_TEXT_ARCHITECTURES[model_type]]
    except KeyError as exc:
        supported = ", ".join(sorted(_MISTRAL3_TEXT_ARCHITECTURES))
        raise ValueError(
            "Unsupported Shieldstral text_config.model_type "
            f"{model_type!r}; expected one of: {supported}"
        ) from exc


def _prepare_llama4_scaling(config: object) -> None:
    if getattr(config, "llama_4_scaling", None) is not None:
        return
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return
    scaling_beta = rope_parameters.get("llama_4_scaling_beta")
    if scaling_beta is None:
        return
    original_max_position = rope_parameters.get(
        "original_max_position_embeddings"
    )
    if original_max_position is None:
        raise ValueError(
            "llama_4_scaling_beta requires original_max_position_embeddings"
        )
    config.llama_4_scaling = {  # type: ignore[attr-defined]
        "beta": scaling_beta,
        "original_max_position_embeddings": original_max_position,
    }


def patch_mistral3_text_model() -> None:
    """Resolve Shieldstral's Ministral3 text model on vLLM 0.26."""
    import vllm.model_executor.models.mistral3 as mistral3

    original = mistral3.init_vllm_registered_model
    if getattr(original, "_vllm_ascend_shieldstral_compat", False):
        return

    @wraps(original)
    def init_vllm_registered_model(
        vllm_config: VllmConfig,
        *,
        prefix: str = "",
        hf_config: Any | None = None,
        architectures: list[str] | None = None,
    ):
        if hf_config is not None and architectures is None:
            architectures = _get_mistral3_text_architectures(hf_config)
            _prepare_llama4_scaling(hf_config)
        return original(
            vllm_config,
            prefix=prefix,
            hf_config=hf_config,
            architectures=architectures,
        )

    init_vllm_registered_model._vllm_ascend_shieldstral_compat = (  # type: ignore[attr-defined]
        True
    )
    mistral3.init_vllm_registered_model = init_vllm_registered_model
