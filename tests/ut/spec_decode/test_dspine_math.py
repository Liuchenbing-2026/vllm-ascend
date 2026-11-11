# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CPU unit tests for the DSpine math (``vllm_ascend/models/dspine_math.py``).
#
# The module is loaded by path so the tests run without vllm/torch_npu installed:
# only torch is needed. The refinement math is checked against a literal
# recomputation of the last code-bridge write, which is what the fused kernels in
# the reference implementation compute.
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MODULE_PATH = next(
    p / "vllm_ascend" / "models" / "dspine_math.py"
    for p in Path(__file__).resolve().parents
    if (p / "vllm_ascend" / "models" / "dspine_math.py").is_file()
)
_spec = importlib.util.spec_from_file_location("dspine_math", _MODULE_PATH)
dspine_math = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["dspine_math"] = dspine_math
_spec.loader.exec_module(dspine_math)


def _bridge(rank, code_size, hidden, vocab, num_layers, block, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)  # noqa: E731
    return SimpleNamespace(
        codebook=rand(vocab, code_size),
        read_weight=rand(code_size, hidden),
        early_read_weights=rand(num_layers - 1, code_size, hidden),
        feature_write_weights=rand(num_layers, hidden, rank),
        packed_weight=rand(3 * rank, code_size),
        gate=SimpleNamespace(bias=rand(rank)),
        rank=rank,
        code_size=code_size,
        eps=1e-6,
        num_layers=num_layers,
        block_size=block,
    )


def _dspine_config(**overrides):
    dflash = {
        "selected_prefix_terminal": False,
        "selected_prefix_decode_mode": "serial",
        "selected_prefix_feature_writes": True,
        "selected_prefix_feature_write_mode": "independent",
        "draft_block_causal": True,
        "selected_prefix_code_bridge": {"code_size": 8, "rank": 4, "eps": 1e-6},
        "selected_prefix_feature_layers": [1, 2],
        "selected_prefix_last_write_refine": {"kind": "exact_sender", "version": 1, "head_topk": 4, "weight": 1.0},
    }
    dflash.update(overrides)
    return SimpleNamespace(hidden_size=8, num_hidden_layers=2, vocab_size=16, block_size=4, dflash_config=dflash)


def test_resolve_spec_rejects_unsupported_checkpoints():
    spec = dspine_math.resolve_dspine_spec(_dspine_config())
    assert spec.active_layers == (0, 1)
    assert spec.head_topk == 4
    assert spec.sqrt_code_size == pytest.approx(math.sqrt(8))

    with pytest.raises(ValueError, match="terminal-disabled"):
        dspine_math.resolve_dspine_spec(_dspine_config(selected_prefix_terminal=True))
    with pytest.raises(ValueError, match="causal"):
        dspine_math.resolve_dspine_spec(_dspine_config(draft_block_causal=False))
    with pytest.raises(ValueError, match="last layer must be active"):
        dspine_math.resolve_dspine_spec(_dspine_config(selected_prefix_feature_layers=[1]))


def test_head_topk_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        dspine_math.resolve_dspine_spec(
            _dspine_config(
                selected_prefix_last_write_refine={"kind": "exact_sender", "version": 1, "head_topk": 6, "weight": 1.0}
            )
        )


def test_bridge_anchor_rows_and_block_isolation():
    torch.manual_seed(0)
    hidden, vocab, code_size, rank, layers, block, batch = 8, 16, 8, 4, 2, 4, 2
    bridge = _bridge(rank, code_size, hidden, vocab, layers, block, torch.bfloat16, seed=1)
    ids = torch.tensor([[3, 9, 9, 9], [5, 9, 9, 9]], dtype=torch.long)
    activations = torch.randn(batch * block, hidden).to(torch.bfloat16)
    norm_w = torch.ones(hidden, dtype=torch.bfloat16)

    normed, out, stash = dspine_math.bridge_layer_write(
        activations, None, ids.reshape(-1), layer=layers - 1, bridge=bridge, next_norm_weight=norm_w, next_norm_eps=1e-6
    )
    assert normed.shape == out.shape == (batch * block, hidden)
    assert stash["raw"].shape == (batch * block, hidden)
    assert stash["rms"].shape == (batch * block,)
    assert stash["packed"].shape == (batch * block, 3 * rank)

    # The anchor row of every block carries no write: out == raw exactly.
    out_blocks = out.view(batch, block, hidden)
    raw_blocks = stash["raw"].view(batch, block, hidden)
    assert torch.equal(out_blocks[:, 0], raw_blocks[:, 0])
    assert not torch.equal(out_blocks[:, 1], raw_blocks[:, 1])

    # Changing another request's anchors must not change block 0.
    other_ids = ids.clone()
    other_ids[1, 0] = 7
    _, out_other, _ = dspine_math.bridge_layer_write(
        activations,
        None,
        other_ids.reshape(-1),
        layer=layers - 1,
        bridge=bridge,
        next_norm_weight=norm_w,
        next_norm_eps=1e-6,
    )
    assert torch.equal(out_other.view(batch, block, hidden)[0], out_blocks[0])
    assert not torch.equal(out_other.view(batch, block, hidden)[1], out_blocks[1])


def _exact_lane_best(cand, raw, rms, q, anchors, bridge, head, pv_w, gate_bias, norm_w):
    """Literal recomputation of the last write for every (row, predecessor) lane."""
    batch, steps, topk = cand.shape
    rank = pv_w.shape[0] // 2
    code_size = pv_w.shape[1]
    sqrt_c = math.sqrt(code_size)
    lane_best = torch.zeros(batch, steps, topk, dtype=torch.long)
    for b in range(batch):
        for i in range(steps):
            for j in range(topk):
                sender = anchors[b] if i == 0 else cand[b, i - 1, j]
                e = sqrt_c * bridge.codebook[sender].to(q.dtype)
                message = torch.sigmoid(q[b, i] + e @ pv_w[:rank].t() + gate_bias) * (e @ pv_w[rank:].t())
                refined = raw[b, i] + rms[b, i] * (message @ bridge.feature_write_weights[-1].t())
                normed = refined / torch.sqrt(refined.pow(2).mean() + 1e-6) * norm_w
                scores = normed @ head[cand[b, i]].t()
                lane_best[b, i, j] = torch.argmax(scores)
    return lane_best


def test_serial_refine_matches_literal_recomputation():
    torch.manual_seed(0)
    dtype = torch.float64
    hidden, vocab, code_size, rank, layers, block, batch, topk = 8, 24, 8, 4, 2, 4, 3, 4
    steps = block - 1

    bridge = _bridge(rank, code_size, hidden, vocab, layers, block, dtype, seed=2)
    head = torch.randn(vocab, hidden, dtype=dtype)
    norm_w = torch.randn(hidden, dtype=dtype).abs() + 0.5
    gate_bias = torch.randn(rank, dtype=dtype) * 0.1
    pv_w = bridge.packed_weight[rank:]

    tg, tsv = dspine_math.build_serial_tables(
        head, norm_w, bridge.feature_write_weights[-1], pv_w, bridge.codebook, cast_bf16=False
    )
    assert tg.dtype == dtype and tsv.dtype == dtype

    raw = torch.randn(batch * block, hidden, dtype=dtype)
    rms = torch.rand(batch * block, dtype=dtype) + 0.5
    packed = torch.randn(batch * block, 3 * rank, dtype=dtype)
    rows = torch.randn(batch, steps, hidden, dtype=dtype)
    anchors = torch.randint(0, vocab, (batch,), dtype=torch.long)

    tokens, cand = dspine_math.serial_refine_greedy(
        rows,
        anchors,
        raw,
        rms,
        packed,
        head,
        tg,
        tsv,
        gate_bias,
        norm_w,
        topk,
        block,
        return_candidates=True,
    )

    assert tokens.shape == (batch, steps)
    assert cand.shape == (batch, steps, topk)
    # Candidates are the head top-k, sorted by id for a stable lane order.
    assert torch.all(cand[:, :, 1:] > cand[:, :, :-1])

    lane_best = _exact_lane_best(
        cand,
        raw.view(batch, block, hidden)[:, 1:],
        rms.view(batch, block)[:, 1:],
        packed.view(batch, block, -1)[:, 1:, :rank],
        anchors,
        bridge,
        head,
        pv_w,
        gate_bias,
        norm_w,
    )

    lane = lane_best[:, 0, 0]
    expected = []
    for i in range(steps):
        expected.append(cand[:, i, :].gather(1, lane.unsqueeze(1)).squeeze(1))
        if i + 1 < steps:
            lane = lane_best[:, i + 1, :].gather(1, lane.unsqueeze(1)).squeeze(1)
    expected = torch.stack(expected, dim=1)
    assert torch.equal(tokens, expected)


def test_serial_refine_is_deterministic_and_block_local():
    torch.manual_seed(3)
    dtype = torch.float64
    hidden, vocab, code_size, rank, layers, block, batch, topk = 8, 24, 8, 4, 2, 4, 2, 4
    steps = block - 1
    bridge = _bridge(rank, code_size, hidden, vocab, layers, block, dtype, seed=4)
    head = torch.randn(vocab, hidden, dtype=dtype)
    norm_w = torch.ones(hidden, dtype=dtype)
    gate_bias = torch.zeros(rank, dtype=dtype)
    pv_w = bridge.packed_weight[rank:]
    tg, tsv = dspine_math.build_serial_tables(
        head, norm_w, bridge.feature_write_weights[-1], pv_w, bridge.codebook, cast_bf16=False
    )
    raw = torch.randn(batch * block, hidden, dtype=dtype)
    rms = torch.ones(batch * block, dtype=dtype)
    packed = torch.randn(batch * block, 3 * rank, dtype=dtype)
    rows = torch.randn(batch, steps, hidden, dtype=dtype)
    anchors = torch.tensor([1, 2], dtype=torch.long)

    first = dspine_math.serial_refine_greedy(
        rows, anchors, raw, rms, packed, head, tg, tsv, gate_bias, norm_w, topk, block
    )
    second = dspine_math.serial_refine_greedy(
        rows, anchors, raw, rms, packed, head, tg, tsv, gate_bias, norm_w, topk, block
    )
    assert torch.equal(first, second)

    # Replacing request 1's block leaves request 0 unchanged.
    raw_other = raw.clone()
    raw_other[block : block + 1] += 5.0
    packed_other = packed.clone()
    packed_other[block:] = torch.randn(block, 3 * rank, dtype=dtype)
    third = dspine_math.serial_refine_greedy(
        rows, anchors, raw_other, rms, packed_other, head, tg, tsv, gate_bias, norm_w, topk, block
    )
    assert torch.equal(third[0], first[0])


def test_serial_refine_bf16_production_path():
    """The served path: bf16 tables/logits (topk and gathers must support bf16)."""
    torch.manual_seed(5)
    dtype = torch.bfloat16
    hidden, vocab, code_size, rank, layers, block, batch, topk = 8, 32, 8, 4, 2, 4, 2, 4
    steps = block - 1
    bridge = _bridge(rank, code_size, hidden, vocab, layers, block, dtype, seed=6)
    head = torch.randn(vocab, hidden, dtype=dtype)
    norm_w = torch.ones(hidden, dtype=dtype)
    gate_bias = torch.zeros(rank, dtype=dtype)
    pv_w = bridge.packed_weight[rank:]
    tg, tsv = dspine_math.build_serial_tables(head, norm_w, bridge.feature_write_weights[-1], pv_w, bridge.codebook)
    assert tg.dtype == dtype and tsv.dtype == dtype

    raw = torch.randn(batch * block, hidden, dtype=dtype)
    rms = torch.ones(batch * block, dtype=dtype)
    packed = torch.randn(batch * block, 3 * rank, dtype=dtype)
    rows = torch.randn(batch, steps, hidden, dtype=dtype)
    anchors = torch.tensor([0, 1], dtype=torch.long)

    tokens, cand = dspine_math.serial_refine_greedy(
        rows, anchors, raw, rms, packed, head, tg, tsv, gate_bias, norm_w, topk, block, return_candidates=True
    )
    assert tokens.shape == (batch, steps)
    assert tokens.dtype == torch.long
    assert int(tokens.min()) >= 0 and int(tokens.max()) < vocab
    # Every selected token is one of that row's candidates.
    assert all(bool((cand[b, i] == tokens[b, i]).any()) for b in range(batch) for i in range(steps))
