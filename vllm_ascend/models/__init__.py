from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")

    # DSpark draft model. vLLM core rewrites the draft architecture to the bare
    # name ``DeepSeekV4DSpark``; register both that alias and the explicit
    # ``...Model`` form so checkpoints from either path resolve (bug#2).
    for _dspark_arch in ("DeepSeekV4DSparkModel", "DeepSeekV4DSpark"):
        ModelRegistry.register_model(
            _dspark_arch,
            "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
        )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
