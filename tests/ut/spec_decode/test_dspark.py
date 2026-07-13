#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for DSpark speculators support (GLM-5.2 DSpark)."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm_ascend import envs
from vllm_ascend.patch.platform.patch_speculators_dspark import update_dspark
from vllm_ascend.patch.worker.patch_qwen3_dflash import (
    _dspark_debug_model_forward,
    _resolve_kv_cache_pair,
    _should_capture_dspark_backbone,
    precompute_and_store_context_kv,
)
from vllm_ascend.spec_decode.dspark_proposer import (
    AscendDsparkProposer,
    _build_logit_debug_record,
)

GLM52_SPECULATOR_CONFIG = {
    "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
    "block_size": 8,
    "draft_vocab_size": 154880,
    "mask_token_id": 154856,
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "target_hidden_size": None,
}


class TestUpdateDspark:
    def test_glm52_speculator_mapping(self):
        pre_trained_config: dict = {}
        update_dspark(GLM52_SPECULATOR_CONFIG, pre_trained_config)

        assert pre_trained_config["architectures"] == ["Qwen3DSparkModel"]
        # Target-side aux hook uses the ids as-is; the DFlash model derives
        # the fc fusion width from the shifted dflash_config ids.
        assert pre_trained_config["eagle_aux_hidden_state_layer_ids"] == [8, 23, 39, 55, 70]
        assert pre_trained_config["dflash_config"] == {
            "mask_token_id": 154856,
            "target_layer_ids": [7, 22, 38, 54, 69],
        }
        assert pre_trained_config["draft_vocab_size"] == 154880
        assert pre_trained_config["markov_rank"] == 256
        assert pre_trained_config["block_size"] == 8
        # target_hidden_size is None -> must not be written.
        assert "target_hidden_size" not in pre_trained_config

    def test_target_hidden_size_passthrough(self):
        config = dict(GLM52_SPECULATOR_CONFIG, target_hidden_size=4096)
        pre_trained_config: dict = {}
        update_dspark(config, pre_trained_config)
        assert pre_trained_config["target_hidden_size"] == 4096


class TestReferenceKvCacheResolution:
    def test_direct_key_value_pair(self):
        key_cache = torch.empty(3, 4, 2, 8)
        value_cache = torch.empty_like(key_cache)

        actual_key, actual_value = _resolve_kv_cache_pair(
            [key_cache, value_cache], virtual_engine=0
        )

        assert actual_key is key_cache
        assert actual_value is value_cache

    def test_virtual_engine_stacked_cache(self):
        cache = torch.empty(2, 3, 4, 2, 8)

        actual_key, actual_value = _resolve_kv_cache_pair([cache], virtual_engine=0)

        torch.testing.assert_close(actual_key, cache[0])
        torch.testing.assert_close(actual_value, cache[1])


class _FakeDSparkModel:
    """Minimal draft model exposing the DSpark sampling hooks."""

    def __init__(self, base_logits: torch.Tensor, markov_table: torch.Tensor):
        # [batch, num_spec, vocab] base logits keyed by position.
        self._base_logits = base_logits
        # [vocab, vocab] dense transition bias: bias(prev)[v].
        self._markov_table = markov_table

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, num_spec, vocab = self._base_logits.shape
        return self._base_logits.reshape(batch * num_spec, vocab)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self._markov_table[markov_embed]

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids


class TestMarkovSequentialSampling:
    def _make_proposer(self, base_logits, markov_table, anchors, num_spec):
        batch = base_logits.shape[0]
        num_query_per_req = 1 + num_spec
        input_ids = torch.zeros(batch * num_query_per_req, dtype=torch.int64)
        input_ids[torch.arange(batch) * num_query_per_req] = anchors
        return SimpleNamespace(
            num_speculative_tokens=num_spec,
            model=_FakeDSparkModel(base_logits, markov_table),
            input_ids=input_ids,
            _anchor_indices=torch.arange(batch, dtype=torch.int64) * num_query_per_req,
            _markov_scale=1.0,
        )

    def test_markov_bias_changes_argmax_and_chains(self):
        # vocab=4, batch=1, num_spec=2. Base logits alone argmax to token 1
        # at both positions; the Markov bias conditioned on the anchor
        # (token 3) pushes position 0 to token 2, and the bias conditioned
        # on token 2 pushes position 1 to token 0.
        base = torch.tensor([[[0.0, 1.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]])
        markov = torch.zeros(4, 4)
        markov[3, 2] = 2.0  # after anchor 3, prefer token 2
        markov[2, 0] = 2.0  # after token 2, prefer token 0
        proposer = self._make_proposer(base, markov, torch.tensor([3]), num_spec=2)

        hidden = torch.zeros(2, 8)  # values unused by the fake model
        draft = AscendDsparkProposer._sample_parallel_draft_tokens(proposer, hidden)

        assert draft.tolist() == [[2, 0]]

    def test_zero_bias_matches_parallel_argmax(self):
        # With a zero Markov table DSpark degenerates to DFlash's parallel
        # argmax over the base logits.
        torch.manual_seed(0)
        batch, num_spec, vocab = 3, 4, 16
        base = torch.randn(batch, num_spec, vocab)
        proposer = self._make_proposer(
            base, torch.zeros(vocab, vocab), torch.zeros(batch, dtype=torch.int64), num_spec
        )

        hidden = torch.zeros(batch * num_spec, 8)
        draft = AscendDsparkProposer._sample_parallel_draft_tokens(proposer, hidden)

        assert torch.equal(draft, base.argmax(dim=-1))

    def test_anchor_conditioning_differs_per_request(self):
        # Two requests share base logits but have different anchors; the
        # first sampled tokens must differ accordingly.
        base = torch.zeros(2, 1, 4)
        markov = torch.zeros(4, 4)
        markov[0, 1] = 1.0  # after anchor 0, prefer token 1
        markov[3, 2] = 1.0  # after anchor 3, prefer token 2
        proposer = self._make_proposer(base, markov, torch.tensor([0, 3]), num_spec=1)

        hidden = torch.zeros(2, 8)
        draft = AscendDsparkProposer._sample_parallel_draft_tokens(proposer, hidden)

        assert draft.tolist() == [[1], [2]]

    def test_zero_markov_scale_matches_base_argmax(self):
        base = torch.tensor([[[0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]])
        markov = torch.zeros(3, 3)
        markov[0, 2] = 10.0
        proposer = self._make_proposer(base, markov, torch.tensor([0]), num_spec=2)
        proposer._markov_scale = 0.0

        draft = AscendDsparkProposer._sample_parallel_draft_tokens(
            proposer, torch.zeros(2, 4)
        )

        assert torch.equal(draft, base.argmax(dim=-1))

    def test_debug_chunks_survive_sampling_and_clear_on_record(self, monkeypatch):
        base = torch.tensor([[[0.0, 1.0]]])
        proposer = self._make_proposer(
            base,
            torch.zeros(2, 2),
            torch.tensor([0]),
            num_spec=1,
        )
        backbone_record = {"context_chunks": ["chunk-0", "chunk-1"]}
        backbone = SimpleNamespace(
            _last_dspark_backbone_debug=backbone_record,
            _dspark_context_debug_chunks=["chunk-0", "chunk-1"],
            _dspark_raw_context_debug_chunks=["raw-0", "raw-1"],
            _dspark_backbone_debug_enabled=True,
        )
        proposer.model.model = backbone
        proposer._logit_debug_records = 0
        proposer._last_logit_debug = None
        proposer._last_backbone_debug = None
        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH", "capture")
        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS", "1")

        AscendDsparkProposer._sample_parallel_draft_tokens(
            proposer,
            torch.zeros(1, 4),
        )

        # Sampling also runs for intermediate chunked-prefill steps: it may
        # only consume the per-forward snapshot, never the chunk accumulators.
        assert envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH == "capture"
        assert proposer._last_backbone_debug is backbone_record
        assert backbone._last_dspark_backbone_debug is None
        assert backbone._dspark_context_debug_chunks == ["chunk-0", "chunk-1"]
        assert backbone._dspark_raw_context_debug_chunks == ["raw-0", "raw-1"]

        # The target verification handoff releases the accumulators even when
        # max_records is already exhausted (early-return path).
        proposer._logit_debug_records = 1
        AscendDsparkProposer.record_target_logit_debug(proposer, None, None)

        assert proposer._last_logit_debug is None
        assert proposer._last_backbone_debug is None
        assert backbone._dspark_context_debug_chunks == []
        assert backbone._dspark_raw_context_debug_chunks == []
        assert backbone._dspark_backbone_debug_enabled is False

    def test_disabled_backbone_gate_skips_logit_debug(self, monkeypatch):
        base = torch.tensor([[[0.0, 1.0]]])
        proposer = self._make_proposer(
            base,
            torch.zeros(2, 2),
            torch.tensor([0]),
            num_spec=1,
        )
        proposer.model.model = SimpleNamespace(
            _dspark_backbone_debug_enabled=False,
        )
        proposer._logit_debug_records = 0
        proposer._last_logit_debug = None
        proposer._last_backbone_debug = None
        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH", "capture")
        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS", "1")

        AscendDsparkProposer._sample_parallel_draft_tokens(
            proposer,
            torch.zeros(1, 4),
        )

        assert proposer._last_logit_debug is None
        assert proposer._last_backbone_debug is None


class TestLogitDebugRecord:
    def test_reports_real_target_rank_per_component(self):
        captured = {
            "base_logits": torch.tensor([[[0.0, 3.0, 2.0], [4.0, 1.0, 0.0]]]),
            "markov_bias": torch.tensor([[[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]]]),
            "final_logits": torch.tensor([[[0.0, 3.0, 4.0], [4.0, 4.0, 0.0]]]),
            "prev_token_ids": torch.tensor([[7, 2]]),
            "proposed_token_ids": torch.tensor([[2, 0]]),
            "markov_scale": 1.0,
        }
        target_logits = torch.tensor([[0.0, 1.0, 5.0], [0.0, 6.0, 1.0]])

        record = _build_logit_debug_record(
            captured,
            target_logits,
            num_draft_tokens=[2],
            draft_token_ids=torch.tensor([2, 0]),
            record_index=3,
        )

        assert record["record"] == 3
        assert record["rows"][0]["target_token_id"] == 2
        assert record["rows"][0]["base_target_rank"] == 2
        assert record["rows"][0]["final_target_rank"] == 1
        assert record["rows"][0]["accepted"] is True
        assert record["rows"][1]["target_token_id"] == 1
        assert record["rows"][1]["base_target_rank"] == 2
        assert record["rows"][1]["markov_target_rank"] == 1
        assert record["rows"][1]["accepted"] is False


class _CaptureKvUpdate:
    def __init__(self):
        self.key = None

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self.key = key.clone()


class _OutOfPlaceRotary(nn.Module):
    def forward(self, positions, query, key):
        return query + 10, key + 20


class TestContextKvPrecompute:
    def test_uses_out_of_place_rope_result(self):
        cache_impl = _CaptureKvUpdate()
        self_attn = SimpleNamespace(k_norm=nn.Identity(), rotary_emb=_OutOfPlaceRotary())
        cache_layer = SimpleNamespace(kv_cache=[torch.empty(0), torch.empty(0)], impl=cache_impl)
        model = SimpleNamespace(
            _num_attn_layers=1,
            _kv_size=2,
            _head_dim=2,
            _num_kv_heads=1,
            hidden_norm=nn.Identity(),
            _fused_kv_weight=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            _fused_kv_bias=None,
            layers=[SimpleNamespace(self_attn=self_attn)],
            _attn_layers=[cache_layer],
        )

        precompute_and_store_context_kv(
            model,
            context_states=torch.tensor([[1.0, 2.0]]),
            context_positions=torch.tensor([0]),
            context_slot_mapping=torch.tensor([3], dtype=torch.int32),
        )

        torch.testing.assert_close(cache_impl.key, torch.tensor([[[11.0, 12.0]]]))


class _DebugLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        del positions
        if residual is None:
            residual = hidden_states + 2
        return hidden_states + 1, residual


class _DebugNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, residual


class TestBackboneDebug:
    def test_capture_gate_respects_budget_and_model_disable(self, monkeypatch):
        model = SimpleNamespace(markov_head=object())
        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH", "capture")
        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS", "0")

        assert _should_capture_dspark_backbone(model) is False

        monkeypatch.setenv("VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS", "1")
        assert _should_capture_dspark_backbone(model) is False

        model._dspark_backbone_debug_enabled = True
        assert _should_capture_dspark_backbone(model) is True

        model._dspark_backbone_debug_enabled = False
        assert _should_capture_dspark_backbone(model) is False

    def test_records_equivalent_layer_outputs(self):
        model = SimpleNamespace(
            embed_input_ids=lambda input_ids: input_ids.float().unsqueeze(-1),
            layers=[_DebugLayer()],
            norm=_DebugNorm(),
            config=SimpleNamespace(
                hidden_size=1,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=1,
                rms_norm_eps=1e-6,
                rope_parameters={"rope_theta": 10000},
            ),
            _dspark_context_debug_chunks=[{"context_positions": torch.tensor([0])}],
        )

        output = _dspark_debug_model_forward(
            model,
            input_ids=torch.tensor([3]),
            positions=torch.tensor([4]),
        )

        torch.testing.assert_close(output, torch.tensor([[9.0]]))
        record = model._last_dspark_backbone_debug
        torch.testing.assert_close(record["layers"][0]["input"], torch.tensor([[3.0]]))
        torch.testing.assert_close(record["layers"][0]["output"], torch.tensor([[9.0]]))
        torch.testing.assert_close(record["final_hidden"], torch.tensor([[9.0]]))
        assert record["context_chunks"] == model._dspark_context_debug_chunks
