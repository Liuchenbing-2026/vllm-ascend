# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Target embedding provisioning for Kimi K3 DSpark under pipeline parallelism."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.distributed import parallel_state
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models import quarot_utils
from vllm_ascend.models.kimi_k3 import kimi_k3
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.utils import vllm_version_is

VOCAB_SIZE = 16
HIDDEN_SIZE = 4


class _Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        *,
        prefix: str = "",
    ):
        super().__init__(num_embeddings, embedding_dim, dtype=params_dtype)
        self.prefix = prefix
        self.quant_method = SimpleNamespace(process_weights_after_loading=MagicMock())


def _draft_pp_context(pp_size: int):
    if vllm_version_is("0.33.0"):
        return patch.object(eagle_utils, "get_pp_group", return_value=SimpleNamespace(world_size=pp_size))
    return patch.object(eagle_utils, "in_single_pp_stage", return_value=pp_size > 1)


def _kimi_vllm_config(
    *,
    pp_size: int,
    method: str | None,
    architectures: list[str] | None,
    draft_vocab: int = VOCAB_SIZE,
    model_path: str = "/target",
):
    text_config = SimpleNamespace(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        attn_res_block_size=None,
        rms_norm_eps=1e-5,
        dsa_config=None,
        is_moe=False,
    )
    spec_config = (
        None
        if method is None
        else SimpleNamespace(
            method=method,
            uses_target_hidden_states=True,
            draft_model_config=SimpleNamespace(
                may_use_target_embed=hasattr(AscendQwen3DSparkForCausalLM, "has_own_embed_tokens"),
                hf_config=SimpleNamespace(
                    model_type="qwen3",
                    architectures=architectures or [],
                ),
                get_vocab_size=lambda: draft_vocab,
            ),
        )
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model_path,
            hf_text_config=text_config,
            get_vocab_size=lambda: VOCAB_SIZE,
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=pp_size,
            enable_expert_parallel=False,
            tensor_parallel_size=1,
        ),
        speculative_config=spec_config,
        use_v2_model_runner=True,
    )


def _make_kimi_target(monkeypatch, *, pp_size: int, pp_rank: int, vllm_config):
    group = SimpleNamespace(
        world_size=pp_size,
        is_first_rank=pp_rank == 0,
        is_last_rank=pp_rank == pp_size - 1,
    )
    monkeypatch.setattr(kimi_k3, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kimi_k3, "VocabParallelEmbedding", _Embedding)
    monkeypatch.setattr(kimi_k3, "RMSNorm", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(kimi_k3, "make_layers", lambda *_args, **_kwargs: (0, 0, nn.ModuleList()))
    return kimi_k3.AscendKimiLinearModel(vllm_config=vllm_config, prefix="model")


@pytest.mark.parametrize(
    ("pp_size", "pp_rank", "method", "architectures", "draft_vocab", "expected"),
    [
        (2, 1, "dspark", ["Qwen3DSparkModel"], VOCAB_SIZE, True),
        (2, 1, "dspark", ["K3DSparkModel"], VOCAB_SIZE, False),
        (2, 1, "dspark", ["Qwen3DSparkModel"], VOCAB_SIZE * 2, False),
        (3, 1, "dspark", ["Qwen3DSparkModel"], VOCAB_SIZE, False),
        (2, 1, None, None, VOCAB_SIZE, False),
    ],
)
def test_draft_needs_target_embed_scope(
    monkeypatch,
    pp_size,
    pp_rank,
    method,
    architectures,
    draft_vocab,
    expected,
):
    group = SimpleNamespace(
        world_size=pp_size,
        is_first_rank=pp_rank == 0,
        is_last_rank=pp_rank == pp_size - 1,
    )
    monkeypatch.setattr(kimi_k3, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    vllm_config = _kimi_vllm_config(
        pp_size=pp_size,
        method=method,
        architectures=architectures,
        draft_vocab=draft_vocab,
    )
    assert kimi_k3._draft_needs_target_embed(vllm_config) is expected


def test_kimi_last_stage_provision_and_fail_fast_missing_embed(monkeypatch):
    cfg = _kimi_vllm_config(pp_size=2, method="dspark", architectures=["Qwen3DSparkModel"])
    target = _make_kimi_target(monkeypatch, pp_size=2, pp_rank=1, vllm_config=cfg)
    assert isinstance(target.embed_tokens, _Embedding)
    assert target._draft_embed_provisioned is True

    def fake_super_load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor, *_rest in weights:
            if name in params:
                params[name].data.copy_(tensor)
            loaded.add(name)
        return loaded

    monkeypatch.setattr(kimi_k3.UpstreamKimiLinearModel, "load_weights", fake_super_load_weights)
    with pytest.raises(RuntimeError, match=r"no \*embed_tokens\.weight"):
        target.load_weights([("norm.weight", torch.ones(HIDDEN_SIZE))])

    embed_weight = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(VOCAB_SIZE, HIDDEN_SIZE)
    loaded = target.load_weights([("embed_tokens.weight", embed_weight)])
    assert "embed_tokens.weight" in loaded
    torch.testing.assert_close(target.embed_tokens.weight, embed_weight)

    mla_cfg = _kimi_vllm_config(pp_size=2, method="dspark", architectures=["K3DSparkModel"])
    mla_target = _make_kimi_target(monkeypatch, pp_size=2, pp_rank=1, vllm_config=mla_cfg)
    assert isinstance(mla_target.embed_tokens, PPMissingLayer)
    assert mla_target._draft_embed_provisioned is False


@pytest.mark.parametrize("rotated", [False, True])
def test_kimi_unrotated_and_quarot_dspark_share_flow(monkeypatch, tmp_path, rotated):
    cfg = _kimi_vllm_config(
        pp_size=2,
        method="dspark",
        architectures=["Qwen3DSparkModel"],
        model_path=str(tmp_path),
    )
    target = _make_kimi_target(monkeypatch, pp_size=2, pp_rank=1, vllm_config=cfg)
    assert isinstance(target.embed_tokens, _Embedding)

    embed_weight = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(VOCAB_SIZE, HIDDEN_SIZE)
    head_weight = embed_weight + 10.0

    def fake_super_load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor, *_rest in weights:
            params[name].data.copy_(tensor)
            loaded.add(name)
        return loaded

    monkeypatch.setattr(kimi_k3.UpstreamKimiLinearModel, "load_weights", fake_super_load_weights)
    target.load_weights([("embed_tokens.weight", embed_weight)])
    target_embed_before = target.embed_tokens.weight.detach().clone()

    draft = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    draft.model.fc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE, bias=False)
    target_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
    with torch.no_grad():
        target_head.weight.copy_(head_weight)
    draft.lm_head = target_head
    draft.has_own_lm_head = False
    assert AscendQwen3DSparkForCausalLM.has_own_embed_tokens is False

    with _draft_pp_context(2):
        eagle_utils.maybe_share_target_embed(draft, draft.model, target)
    assert draft.model.embed_tokens is target.embed_tokens

    save_file(
        {"model.embed_tokens.weight": embed_weight, "lm_head.weight": head_weight},
        tmp_path / "model.safetensors",
    )
    rotation = torch.eye(HIDDEN_SIZE, dtype=torch.float32).flip(0)
    original_fc = draft.model.fc.weight.detach().clone()
    monkeypatch.setattr(quarot_utils, "get_rotation_path", lambda _cfg: "/rotation" if rotated else None)
    monkeypatch.setattr(quarot_utils, "get_rotation_matrix", lambda _path: rotation)
    monkeypatch.setattr(quarot_utils, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(quarot_utils, "VocabParallelEmbedding", _Embedding)
    monkeypatch.setattr(quarot_utils, "ParallelLMHead", _Embedding)

    draft.post_process(cfg)

    torch.testing.assert_close(target.embed_tokens.weight, target_embed_before)
    torch.testing.assert_close(target_head.weight, head_weight)
    if rotated:
        assert draft.model.embed_tokens is not target.embed_tokens
        assert draft.lm_head is not target_head
        assert draft.has_own_embed_tokens and draft.has_own_lm_head
        torch.testing.assert_close(draft.model.embed_tokens.weight, embed_weight @ rotation.T)
        torch.testing.assert_close(draft.lm_head.weight, head_weight @ rotation.T)
    else:
        assert draft.model.embed_tokens is target.embed_tokens
        assert draft.lm_head is target_head
        assert draft.has_own_embed_tokens is False
        torch.testing.assert_close(draft.model.fc.weight, original_fc)
