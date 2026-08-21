import vllm.envs as envs
from vllm.config.vllm import VllmConfig


def _patched_use_v2_model_runner(self) -> bool:
    """Use the explicit runner selection, with required NPU exceptions.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    primarily by the VLLM_USE_V2_MODEL_RUNNER environment variable;
    model-compatibility decisions are deferred to the NPU runner itself.
    DFlash2 is the narrow exception because its candidate selector exists
    only in the v2 speculator.
    """
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    if self._is_dflash2_draft():
        return True
    return False


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
