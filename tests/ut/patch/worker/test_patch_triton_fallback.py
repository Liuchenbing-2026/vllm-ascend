# SPDX-License-Identifier: Apache-2.0
"""Reachability and behaviour tests for the no-Triton GDN fallback (#18743).

The fallback used to live in ``patch_triton.py`` under ``if not HAS_TRITON``,
but that module is imported from ``vllm_ascend.patch.worker`` under
``if HAS_TRITON`` -- opposite guards, so the fallback could never run. These
tests pin both the reachability invariant and the installed bindings.
"""

import ast
import importlib.util
import pathlib
import sys
import types

import torch

import vllm_ascend

WORKER_DIR = pathlib.Path(vllm_ascend.__file__).parent / "patch" / "worker"
INIT_PATH = WORKER_DIR / "__init__.py"
FALLBACK_MODULE = "vllm_ascend.patch.worker.patch_triton_fallback"
FALLBACK_PATH = WORKER_DIR / "patch_triton_fallback.py"


def _module_imports(body):
    names = set()
    for node in body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.If):
            names |= _module_imports(node.body) | _module_imports(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _module_imports(node.body) | _module_imports(node.orelse)
            for handler in node.handlers:
                names |= _module_imports(handler.body)
    return names


def _imports_of_worker_entry(has_triton: bool):
    """Module names imported by ``patch/worker/__init__.py`` for a HAS_TRITON state.

    ``if HAS_TRITON:`` bodies are followed only when ``has_triton`` is True.
    Any other condition is assumed reachable in some configuration and both of
    its branches are counted.
    """
    tree = ast.parse(INIT_PATH.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test).strip()
            if test == "HAS_TRITON":
                names |= _module_imports(node.body if has_triton else node.orelse)
            elif test == "not HAS_TRITON":
                names |= _module_imports(node.orelse if has_triton else node.body)
            else:
                names |= _module_imports(node.body) | _module_imports(node.orelse)
    return names


def _modules_declaring_a_no_triton_fallback():
    """Worker patch modules whose top level installs a ``not HAS_TRITON`` fallback."""
    declaring = {}
    repo_root = WORKER_DIR.parent.parent.parent
    for path in sorted(WORKER_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.If) and ast.unparse(node.test).strip() == "not HAS_TRITON":
                module = path.relative_to(repo_root).with_suffix("").as_posix().replace("/", ".")
                declaring[module] = path
    return declaring


def test_no_triton_fallback_is_importable_without_triton():
    """Every ``if not HAS_TRITON`` fallback must be imported when HAS_TRITON is off.

    This is the regression guard for #18743: a fallback defined in a module that
    is only imported under ``if HAS_TRITON`` is dead code.
    """
    declaring = _modules_declaring_a_no_triton_fallback()
    assert declaring, "expected at least one no-Triton fallback module"

    imported_without_triton = _imports_of_worker_entry(has_triton=False)
    unreachable = sorted(mod for mod in declaring if mod not in imported_without_triton)
    assert not unreachable, (
        "these modules install a no-Triton fallback but are not imported when"
        f" HAS_TRITON is False: {unreachable}"
    )


def test_fallback_module_is_imported_in_both_triton_states():
    assert FALLBACK_MODULE in _imports_of_worker_entry(has_triton=True)
    assert FALLBACK_MODULE in _imports_of_worker_entry(has_triton=False)


def _load_fallback_module(name: str):
    spec = importlib.util.spec_from_file_location(name, FALLBACK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_fla(monkeypatch):
    """Register a fake FLA ops hierarchy so ``apply_patch`` can import it."""
    import vllm

    third_party = types.ModuleType("vllm.third_party")
    third_party.__path__ = []
    fla = types.ModuleType("vllm.third_party.flash_linear_attention")
    fla.__path__ = []
    ops = types.ModuleType("vllm.third_party.flash_linear_attention.ops")
    ops.__path__ = []
    fused_recurrent = types.ModuleType(
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
    )

    sentinel = object()
    ops.fused_post_conv_prep = sentinel
    ops.fused_recurrent_gated_delta_rule_packed_decode = sentinel
    fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = sentinel

    ops.fused_recurrent = fused_recurrent
    fla.ops = ops
    third_party.flash_linear_attention = fla

    monkeypatch.setitem(sys.modules, "vllm.third_party", third_party)
    monkeypatch.setitem(sys.modules, "vllm.third_party.flash_linear_attention", fla)
    monkeypatch.setitem(sys.modules, "vllm.third_party.flash_linear_attention.ops", ops)
    monkeypatch.setitem(
        sys.modules,
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
        fused_recurrent,
    )
    monkeypatch.setattr(vllm, "third_party", third_party, raising=False)
    return ops, fused_recurrent


def test_apply_patch_binds_package_and_submodule(monkeypatch):
    ops, fused_recurrent = _install_fake_fla(monkeypatch)
    module = _load_fallback_module("_ut_patch_triton_fallback_bind")

    module.apply_patch()

    assert ops.fused_post_conv_prep is module._fused_post_conv_prep_pytorch
    # The package re-export and the submodule must both be updated: the
    # consumer imports from the package while the kernel lives on the
    # submodule, so binding only one leaves a Triton reference behind.
    assert (
        ops.fused_recurrent_gated_delta_rule_packed_decode
        is module._fused_recurrent_packed_decode_pytorch
    )
    assert (
        fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode
        is module._fused_recurrent_packed_decode_pytorch
    )
    assert (
        ops.fused_recurrent_gated_delta_rule_packed_decode
        is fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode
    )


def test_module_self_guards_and_installs_when_triton_is_absent(monkeypatch):
    import vllm.triton_utils as triton_utils

    ops, _ = _install_fake_fla(monkeypatch)
    monkeypatch.setattr(triton_utils, "HAS_TRITON", False)

    module = _load_fallback_module("_ut_patch_triton_fallback_notriton")

    # Importing the module (not just calling apply_patch) is enough when Triton
    # is unavailable, which is the whole point of the reachability fix.
    assert ops.fused_post_conv_prep is module._fused_post_conv_prep_pytorch


def test_module_is_a_noop_when_triton_is_available(monkeypatch):
    import vllm.triton_utils as triton_utils

    ops, _ = _install_fake_fla(monkeypatch)
    monkeypatch.setattr(triton_utils, "HAS_TRITON", True)
    sentinel = ops.fused_post_conv_prep

    module = _load_fallback_module("_ut_patch_triton_fallback_triton")

    assert module is not None
    assert ops.fused_post_conv_prep is sentinel


def test_fused_post_conv_prep_pytorch_matches_reference():
    module = _load_fallback_module("_ut_patch_triton_fallback_math")

    torch.manual_seed(0)
    L, H, K, V, HV = 5, 2, 3, 4, 2
    conv_output = torch.randn(L, 2 * H * K + HV * V, dtype=torch.float32)
    a = torch.randn(L, HV)
    b = torch.randn(L, HV)
    A_log = torch.randn(HV)
    dt_bias = torch.randn(HV)

    q, k, v, g, beta = module._fused_post_conv_prep_pytorch(
        conv_output, a, b, A_log, dt_bias, H, K, V
    )

    assert q.shape == (L, H, K)
    assert k.shape == (L, H, K)
    assert v.shape == (L, HV, V)
    assert g.shape == (L, HV)
    assert beta.shape == (L, HV)

    q_ref = conv_output[:, : H * K].reshape(L, H, K).float()
    k_ref = conv_output[:, H * K : 2 * H * K].reshape(L, H, K).float()
    q_ref = q_ref / torch.sqrt((q_ref * q_ref).sum(-1, keepdim=True) + 1e-6)
    k_ref = k_ref / torch.sqrt((k_ref * k_ref).sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(q.float(), q_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k.float(), k_ref, rtol=1e-5, atol=1e-6)

    g_ref = -torch.exp(A_log.float().unsqueeze(0)) * torch.nn.functional.softplus(
        (a + dt_bias.unsqueeze(0)).float()
    )
    torch.testing.assert_close(g, g_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(beta, torch.sigmoid(b.float()), rtol=1e-6, atol=1e-6)


def _packed_decode_inputs(batch: int = 2):
    torch.manual_seed(2)
    H, K, V, HV = 2, 3, 4, 2
    B = batch
    mixed_qkv = torch.randn(B, 2 * H * K + HV * V)
    a = torch.randn(B, HV)
    b = torch.randn(B, HV)
    A_log = torch.randn(HV)
    dt_bias = torch.randn(HV)
    initial_state = torch.randn(3, HV, V, K)
    out = torch.zeros(B, 1, HV, V)
    return mixed_qkv, a, b, A_log, dt_bias, initial_state, out, H, K, V, HV


def test_fused_recurrent_packed_decode_pytorch_skips_padding_slots():
    module = _load_fallback_module("_ut_patch_triton_fallback_pad")

    mixed_qkv, a, b, A_log, dt_bias, initial_state, out, H, K, V, HV = (
        _packed_decode_inputs()
    )
    state_before = initial_state.clone()
    ssm_state_indices = torch.tensor([2, 0], dtype=torch.int32)

    out_ret, state_ret = module._fused_recurrent_packed_decode_pytorch(
        mixed_qkv, a, b, A_log, dt_bias, 0.7, initial_state, out, ssm_state_indices
    )
    assert out_ret is out
    assert state_ret is initial_state

    # Row 1 is a padding slot (state_idx <= 0): output zeroed, state untouched.
    assert torch.count_nonzero(out[1]) == 0
    torch.testing.assert_close(initial_state[0], state_before[0])
    assert torch.count_nonzero(out[0]) > 0


def test_fused_recurrent_packed_decode_pytorch_matches_reference():
    module = _load_fallback_module("_ut_patch_triton_fallback_recurrent_ref")

    mixed_qkv, a, b, A_log, dt_bias, initial_state, out, H, K, V, HV = (
        _packed_decode_inputs(batch=1)
    )
    ratio = HV // H
    scale = 0.7
    idx = 1
    start_state = initial_state[idx].float().clone()

    module._fused_recurrent_packed_decode_pytorch(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        torch.tensor([idx], dtype=torch.int32),
    )

    # Independent element-wise reference of one delta-rule step (per v-head).
    q = mixed_qkv[0, : H * K].reshape(H, K).float()
    k = mixed_qkv[0, H * K : 2 * H * K].reshape(H, K).float()
    v = mixed_qkv[0, 2 * H * K :].reshape(HV, V).float()
    x = (a[0] + dt_bias).float()
    softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log.float()) * softplus_x
    beta = torch.sigmoid(b[0].float())

    k_rep = k.repeat_interleave(ratio, dim=0)
    q_rep = q.repeat_interleave(ratio, dim=0) * scale
    h = start_state * torch.exp(g).view(HV, 1, 1)
    delta = torch.stack([h[i] @ k_rep[i] for i in range(HV)])
    v_new = (v - delta) * beta.view(HV, 1)
    h_new = torch.stack([h[i] + torch.outer(v_new[i], k_rep[i]) for i in range(HV)])
    expected = torch.stack([h_new[i] @ q_rep[i] for i in range(HV)])

    torch.testing.assert_close(out[0, 0], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        initial_state[idx].float(), h_new, rtol=1e-5, atol=1e-5
    )
