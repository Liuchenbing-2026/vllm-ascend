from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_use_v2_model_runner


@pytest.mark.parametrize(
    ("is_dflash2", "expected"),
    [
        (False, False),
        (True, True),
    ],
)
def test_dflash2_requires_v2_when_runner_env_is_unset(
    monkeypatch, is_dflash2: bool, expected: bool
) -> None:
    monkeypatch.setattr(
        patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", None
    )
    config = SimpleNamespace(_is_dflash2_draft=lambda: is_dflash2)

    assert patch_use_v2_model_runner._patched_use_v2_model_runner(config) is expected


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_runner_env_takes_priority(monkeypatch, explicit: bool) -> None:
    monkeypatch.setattr(
        patch_use_v2_model_runner.envs,
        "VLLM_USE_V2_MODEL_RUNNER",
        explicit,
    )
    config = SimpleNamespace(_is_dflash2_draft=lambda: not explicit)

    assert (
        patch_use_v2_model_runner._patched_use_v2_model_runner(config) is explicit
    )
