# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import regex as re
import vllm.engine.arg_utils
import vllm.platforms
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms.cpu import CpuPlatform
from vllm.usage.usage_lib import UsageContext

from tests.ut.base import PytestBase


class TestTutorialDpCommands(PytestBase):
    @pytest.mark.parametrize("doc_name", ["GLM5.2.md", "GLM5.3.md", "GLM5.3-Flash.md", "DeepSeek-V4-Flash-Vision.md"])
    def test_internal_dp_master_uses_headless_compatible_lb(self, doc_name, tmp_path, monkeypatch):
        platform = CpuPlatform()
        monkeypatch.setattr(vllm.platforms, "_current_platform", platform)
        monkeypatch.setattr(vllm.engine.arg_utils, "current_platform", platform)
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "vocab_size": 128,
                    "max_position_embeddings": 128,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                }
            )
        )
        path = Path(__file__).resolve().parents[2] / "docs/source/tutorials/models" / doc_name
        commands = re.sub(r"\\\s*\n\s*", " ", path.read_text()).splitlines()
        checked = 0
        for command in commands:
            if "vllm serve" not in command or "--headless" in command:
                continue
            if "--data-parallel-external-lb" in command or "--data-parallel-hybrid-lb" in command:
                continue

            def argument(name, command=command):
                match = re.search(r"--" + name + r"(?:\s+|=)(\d+)", command)
                return int(match[1]) if match else None

            size, local, start = (
                argument(name)
                for name in ("data-parallel-size", "data-parallel-size-local", "data-parallel-start-rank")
            )
            if size is None or local is None or size <= local or start not in (None, 0):
                continue
            args = EngineArgs(
                model=str(model),
                skip_tokenizer_init=True,
                dtype="float32",
                enforce_eager=True,
                max_model_len=128,
                data_parallel_size=size,
                data_parallel_size_local=local,
                data_parallel_start_rank=start,
            )
            config = args.create_engine_config(UsageContext.OPENAI_API_SERVER)
            assert not config.parallel_config.data_parallel_hybrid_lb, command
            assert not config.parallel_config.data_parallel_external_lb, command
            assert config.parallel_config.data_parallel_rank == 0
            checked += 1
        assert checked
