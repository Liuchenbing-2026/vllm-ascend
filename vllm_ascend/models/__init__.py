from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")

    # MiniMax-M3 (Phase-1: text-only, dense full attention).
    # The VL checkpoint's top-level architectures is
    # "MiniMaxM3SparseForConditionalGeneration"; the text_config carries
    # "MiniMaxM3SparseForCausalLM". Register BOTH names to the same text-only
    # AscendMiniMaxM3ForCausalLM so vllm resolves either architectures entry to
    # the Phase-1 model. (Phase-2 / true VL support would split these.)
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:AscendMiniMaxM3ForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:AscendMiniMaxM3ForCausalLM",
    )
