# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UT for DSpark V4.1 draft weight loading (target embed/head/HC shadowing).

The V4.1 DSpark draft reuses the target checkpoint: the draft's own
``mtp.*`` heads and the target's top-level heads map onto the same
parameters, so the loader must let the draft win regardless of the order in
which the shards are consumed.

For V4.1 this is a latent bug that the shipped checkpoints only hide by shard
order. ``DeepSeek-V4.1-Flash-W8A8-no-wq-wkv`` (272 shards) files the target
head very early and the trained ``mtp.<last>.head`` late, so the draft wins by
luck:

    #2    embed.weight / head.weight (target, shared shard, very early)
    #270  mtp.<last>.head.weight (trained draft)

A checkpoint that ordered these the other way round would silently overwrite
the trained draft head with the target head and collapse the acceptance rate.
These tests pin the order-independent behaviour.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.deepseek_v41.dspark as dspark_mod
from vllm_ascend.models.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM

_VOCAB = 4
_HIDDEN = 8
_NUM_DSPARK_LAYERS = 3

# (destination parameter, draft checkpoint name, target checkpoint name)
_HEAD_SLOTS = {
    "model.embed_tokens.weight": ("mtp.0.embed.weight", "embed.weight"),
    "lm_head.weight": ("mtp.2.head.weight", "head.weight"),
    "model.hc_head_fn": ("mtp.2.hc_head_fn", "hc_head_fn"),
    "model.hc_head_base": ("mtp.2.hc_head_base", "hc_head_base"),
    "model.hc_head_scale": ("mtp.2.hc_head_scale", "hc_head_scale"),
}


class _FakeInner(torch.nn.Module):
    def __init__(self, num_dspark_layers: int) -> None:
        super().__init__()
        self.num_dspark_layers = num_dspark_layers

    def get_expert_mapping(self):
        return []


class _Harness:
    """Minimal stand-in that drives DSparkDeepseekV41ForCausalLM.load_weights."""

    def __init__(self, rotation_path=None) -> None:
        model = object.__new__(DSparkDeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.rotation_path = rotation_path
        model.config = SimpleNamespace(num_hidden_layers=61, num_attention_heads=128)
        model.model = _FakeInner(_NUM_DSPARK_LAYERS)
        model.has_own_embed_tokens = False
        model.has_own_lm_head = False
        self.params = {
            dest: torch.nn.Parameter(torch.full((_VOCAB, _HIDDEN), -1.0))
            for dest in _HEAD_SLOTS
        }
        model.named_parameters = lambda: iter(self.params.items())
        self.model = model

    def load(self, weights):
        return self.model.load_weights(weights)


def _v(value: float) -> torch.Tensor:
    return torch.full((_VOCAB, _HIDDEN), value)


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr(dspark_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(dspark_mod, "enable_dsa_cp", lambda: False)


@pytest.mark.parametrize("draft_first", [True, False])
def test_draft_heads_win_for_every_slot_regardless_of_order(draft_first):
    h = _Harness()
    weights = []
    for draft_name, target_name in _HEAD_SLOTS.values():
        weights.append((draft_name, _v(7.0)))
        weights.append((target_name, _v(99.0)))
    if not draft_first:
        weights.reverse()
    loaded = h.load(weights)
    for dest, param in h.params.items():
        assert torch.equal(param.data, _v(7.0)), dest
    assert set(h.params) <= loaded


def test_target_head_last_does_not_shadow_draft_head():
    # Worst case: the draft's own heads appear early and the target's
    # head.weight is the very last tensor of the stream.
    h = _Harness()
    h.load(
        [
            ("mtp.0.embed.weight", _v(5.0)),
            ("mtp.2.head.weight", _v(7.0)),
            ("embed.weight", _v(88.0)),
            ("head.weight", _v(99.0)),
        ]
    )
    assert torch.equal(h.params["lm_head.weight"].data, _v(7.0))
    assert torch.equal(h.params["model.embed_tokens.weight"].data, _v(5.0))


def test_real_checkpoint_order_v41_w8a8():
    # Stream order approximated for DeepSeek-V4.1-Flash-W8A8-no-wq-wkv:
    # target embed/head land in an early shared shard, the trained draft head
    # arrives near the end. The result must not depend on this ordering.
    h = _Harness()
    h.load(
        [
            ("embed.weight", _v(88.0)),
            ("head.weight", _v(99.0)),
            ("mtp.0.embed.weight", _v(5.0)),
            ("mtp.2.head.weight", _v(7.0)),
        ]
    )
    assert torch.equal(h.params["lm_head.weight"].data, _v(7.0))
    assert torch.equal(h.params["model.embed_tokens.weight"].data, _v(5.0))


def test_target_weights_inherited_when_draft_has_none():
    h = _Harness()
    loaded = h.load([(target, _v(88.0)) for _, target in _HEAD_SLOTS.values()])
    for dest, param in h.params.items():
        assert torch.equal(param.data, _v(88.0)), dest
    assert set(h.params) <= loaded


def test_rotated_checkpoint_does_not_alias_target_heads():
    # ``rotation_path`` is None or a Path, so a truthy value keeps the
    # pre-existing behaviour: top-level embed/head are not aliased, while the
    # HC head keys keep being mapped (they were never gated on the flag).
    h = _Harness(rotation_path="/tmp/rotation.bin")
    h.load(
        [
            ("embed.weight", _v(88.0)),
            ("head.weight", _v(99.0)),
            ("hc_head_fn", _v(66.0)),
        ]
    )
    assert torch.equal(h.params["model.embed_tokens.weight"].data, _v(-1.0))
    assert torch.equal(h.params["lm_head.weight"].data, _v(-1.0))
    assert torch.equal(h.params["model.hc_head_fn"].data, _v(66.0))
