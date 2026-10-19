# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU launcher contract checks; these do not establish Clef NPU accuracy."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("tp,devices", [("1", "0"), ("2", "0,1")])
def test_clef_launcher_keeps_checkpoint_head_and_device_assignment(tmp_path, tp, devices):
    launcher = Path(__file__).resolve().parents[2] / "examples/models/clef/serve.sh"
    capture = tmp_path / "capture"
    capture.write_text('#!/bin/bash\nprintf "%s\\n" "$@" "$ASCEND_RT_VISIBLE_DEVICES"\n')
    capture.chmod(0o755)
    checkpoint = tmp_path / "Clef checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "joint_head_config.json", "joint_head.safetensors"):
        (checkpoint / name).write_text("fixture only; not model weights")
    env = dict(
        os.environ,
        MODEL_PATH=str(checkpoint),
        ASCEND_RT_VISIBLE_DEVICES=devices,
        TENSOR_PARALLEL_SIZE=tp,
        PYTHON_BIN=str(capture),
    )
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True, check=True)
    args = result.stdout.splitlines()
    assert args[:4] == ["-m", "vllm.entrypoints.cli.main", "serve", str(checkpoint)]
    assert args[args.index("--tensor-parallel-size") + 1] == tp
    assert args[args.index("--runner") + 1] == "pooling"
    assert "--enforce-eager" in args
    assert "--no-enable-prefix-caching" in args
    assert "--no-enable-chunked-prefill" in args
    assert "--hf-overrides" not in args
    assert "--pooler-config" not in args
    assert args[-1] == devices


def test_clef_launcher_requires_device_assignment():
    launcher = Path(__file__).resolve().parents[2] / "examples/models/clef/serve.sh"
    env = dict(os.environ, MODEL_PATH="/models/clef")
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Select the assigned NPU devices" in result.stderr


@pytest.mark.parametrize("name", ["config.json", "joint_head_config.json", "joint_head.safetensors"])
@pytest.mark.parametrize("empty", [False, True])
def test_clef_launcher_rejects_incomplete_checkpoint_before_start(tmp_path, name, empty):
    launcher = Path(__file__).resolve().parents[2] / "examples/models/clef/serve.sh"
    checkpoint = tmp_path / "clef"
    checkpoint.mkdir()
    for required in ("config.json", "joint_head_config.json", "joint_head.safetensors"):
        (checkpoint / required).write_text("fixture")
    if empty:
        (checkpoint / name).write_text("")
    else:
        (checkpoint / name).unlink()
    marker = tmp_path / "started"
    capture = tmp_path / "capture"
    capture.write_text('#!/bin/bash\ntouch "$START_MARKER"\n')
    capture.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH=str(checkpoint),
        ASCEND_RT_VISIBLE_DEVICES="0,1",
        PYTHON_BIN=str(capture),
        START_MARKER=str(marker),
    )
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert str(checkpoint / name) in result.stderr
    assert not marker.exists()
