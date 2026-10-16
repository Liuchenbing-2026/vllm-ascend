# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import vllm.third_party.flash_linear_attention.ops as fla_ops
import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fla_fused_recurrent

FALLBACK_MODULE = "vllm_ascend.patch.worker.patch_triton_fallback"
WORKER_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend" / "patch" / "worker"
TARGET_NAMES = (
    "fused_post_conv_prep",
    "fused_recurrent_gated_delta_rule_packed_decode",
)


@pytest.fixture
def fallback():
    return importlib.import_module(FALLBACK_MODULE)


@pytest.fixture
def restore_rebound_bindings(monkeypatch):
    """Undo the process-wide rebinding that ``apply()`` performs."""
    for module in list(sys.modules.values()):
        for name in TARGET_NAMES:
            if name in getattr(module, "__dict__", {}):
                monkeypatch.setattr(module, name, vars(module)[name], raising=False)
    yield


def test_apply_installs_the_pytorch_implementations(fallback, restore_rebound_bindings):
    fallback.apply()

    assert fla_ops.fused_post_conv_prep is fallback.fused_post_conv_prep_pytorch
    assert (
        fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode
        is fallback.fused_recurrent_gated_delta_rule_packed_decode_pytorch
    )


def test_apply_rebinds_references_captured_by_from_import(fallback, restore_rebound_bindings, monkeypatch):
    # qwen_gdn_linear_attn does ``from ...ops import fused_post_conv_prep``,
    # which binds the original callable in the importing module. Rebinding the
    # ops package alone leaves that stale reference in place, so apply() has to
    # sweep the modules that already imported the originals.
    original_post_conv_prep = object()
    original_packed_decode = object()
    monkeypatch.setattr(fla_ops, "fused_post_conv_prep", original_post_conv_prep, raising=False)
    monkeypatch.setattr(
        fla_fused_recurrent,
        "fused_recurrent_gated_delta_rule_packed_decode",
        original_packed_decode,
        raising=False,
    )
    consumer = types.ModuleType("_test_18743_gdn_consumer")
    consumer.fused_post_conv_prep = original_post_conv_prep
    consumer.fused_recurrent_gated_delta_rule_packed_decode = original_packed_decode
    monkeypatch.setitem(sys.modules, "_test_18743_gdn_consumer", consumer)

    fallback.apply()

    assert consumer.fused_post_conv_prep is fallback.fused_post_conv_prep_pytorch
    assert (
        consumer.fused_recurrent_gated_delta_rule_packed_decode
        is fallback.fused_recurrent_gated_delta_rule_packed_decode_pytorch
    )


def test_apply_is_idempotent(fallback, restore_rebound_bindings):
    fallback.apply()
    first_post_conv_prep = fla_ops.fused_post_conv_prep
    first_packed_decode = fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode

    fallback.apply()

    assert fla_ops.fused_post_conv_prep is first_post_conv_prep
    assert fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode is first_packed_decode


def test_fused_post_conv_prep_matches_reference_math(fallback):
    torch.manual_seed(0)
    length, num_k_heads, head_k_dim, head_v_dim, num_v_heads = 5, 2, 3, 4, 4
    conv_output = torch.randn(length, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim, dtype=torch.float32)
    a = torch.randn(length, num_v_heads)
    b = torch.randn(length, num_v_heads)
    A_log = torch.randn(num_v_heads)
    dt_bias = torch.randn(num_v_heads)

    q, k, v, g, beta = fallback.fused_post_conv_prep_pytorch(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
    )

    assert q.shape == (length, num_k_heads, head_k_dim)
    assert k.shape == (length, num_k_heads, head_k_dim)
    assert v.shape == (length, num_v_heads, head_v_dim)
    # v is the untouched tail of conv_output.
    torch.testing.assert_close(v, conv_output[:, 2 * num_k_heads * head_k_dim :].reshape(v.shape))
    # q/k are L2-normalised along the head dim in fp32.
    torch.testing.assert_close(q.float().norm(dim=-1), torch.ones(length, num_k_heads))
    torch.testing.assert_close(k.float().norm(dim=-1), torch.ones(length, num_k_heads))
    # g = -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b).
    expected_g = -torch.exp(A_log.unsqueeze(0)) * torch.nn.functional.softplus(a + dt_bias.unsqueeze(0))
    torch.testing.assert_close(g, expected_g)
    torch.testing.assert_close(beta, torch.sigmoid(b))


def test_fused_post_conv_prep_honours_its_flags(fallback):
    torch.manual_seed(0)
    length, num_k_heads, head_k_dim, head_v_dim, num_v_heads = 3, 1, 2, 2, 1
    conv_output = torch.randn(length, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim)
    a = torch.randn(length, num_v_heads)
    b = torch.randn(length, num_v_heads)
    A_log = torch.randn(num_v_heads)
    dt_bias = torch.randn(num_v_heads)
    args = (conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim)

    raw_q, raw_k, _, g_plain, _ = fallback.fused_post_conv_prep_pytorch(*args, apply_l2norm=False)
    torch.testing.assert_close(raw_q, conv_output[:, : num_k_heads * head_k_dim].reshape(raw_q.shape))
    raw_k_slice = conv_output[:, num_k_heads * head_k_dim : 2 * num_k_heads * head_k_dim]
    torch.testing.assert_close(raw_k, raw_k_slice.reshape(raw_k.shape))

    _, _, _, g_exp, _ = fallback.fused_post_conv_prep_pytorch(*args, output_g_exp=True)
    torch.testing.assert_close(g_exp, torch.exp(g_plain))


def test_packed_decode_writes_decode_output_and_updates_state(fallback):
    torch.manual_seed(0)
    batch, num_k_heads, head_k_dim, head_v_dim, num_v_heads = 2, 1, 3, 2, 2
    num_states = 3
    mixed_qkv = torch.randn(batch, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim)
    initial_state = torch.randn(num_states, num_v_heads, head_v_dim, head_k_dim)
    out = torch.zeros(batch, 1, num_v_heads, head_v_dim)
    # index 0 is the "no state" sentinel, index 2 is a real state slot.
    ssm_state_indices = torch.tensor([0, 2])
    original_state = initial_state.clone()

    result_out, result_state = fallback.fused_recurrent_gated_delta_rule_packed_decode_pytorch(
        mixed_qkv,
        torch.randn(batch, num_v_heads),
        torch.randn(batch, num_v_heads),
        torch.randn(num_v_heads),
        torch.randn(num_v_heads),
        0.5,
        initial_state,
        out,
        ssm_state_indices,
    )

    assert result_out is out
    assert result_state is initial_state
    torch.testing.assert_close(result_out[0], torch.zeros_like(result_out[0]))
    torch.testing.assert_close(initial_state[0], original_state[0])
    assert not torch.equal(initial_state[2], original_state[2])
    torch.testing.assert_close(initial_state[1], original_state[1])


def test_worker_init_imports_the_fallback_when_triton_is_missing():
    tree = ast.parse((WORKER_DIR / "__init__.py").read_text(encoding="utf-8"))

    import_blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "HAS_TRITON"
    ]
    assert import_blocks, "patch/worker/__init__.py must guard the Triton patches on HAS_TRITON"

    imported = {alias.name for node in import_blocks[0].orelse if isinstance(node, ast.Import) for alias in node.names}
    assert FALLBACK_MODULE in imported


def test_patch_triton_no_longer_carries_the_unreachable_fallback():
    tree = ast.parse((WORKER_DIR / "patch_triton.py").read_text(encoding="utf-8"))

    dead_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "HAS_TRITON"
    ]
    assert not dead_guards, "patch_triton is only imported with Triton, so a `not HAS_TRITON` branch is dead code"

    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "fused_post_conv_prep_pytorch" not in defined
    assert "fused_recurrent_gated_delta_rule_packed_decode_pytorch" not in defined
