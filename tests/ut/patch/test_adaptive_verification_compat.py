"""Worker imports must not require a removed, unrelated speculative allocator."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.mark.parametrize("legacy", [False, True])
def test_allocator_patch_matches_available_upstream_api(monkeypatch, legacy):
    module = ModuleType("vllm.v1.worker.gpu.spec_decode.adaptive_verification")
    public_allocator = object()
    module.assign_draft_token_budget = public_allocator

    def original(tensor, indices):
        return tensor.index_fill_(0, indices, 2)

    if legacy:
        module._assign_draft_token_budget = original
        module._assign_draft_token_budget_compiled = object()
    names = ["vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.spec_decode"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys.modules[names[-1]], "adaptive_verification", module, raising=False)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_v2/patch_adaptive_verification.py"
    namespace = runpy.run_path(str(source))
    assert module.assign_draft_token_budget is public_allocator
    if not legacy:
        assert not hasattr(module, "_assign_draft_token_budget_compiled")
        return
    wrapper = module._assign_draft_token_budget_compiled
    calls = []

    def index_fill(tensor, dim, indices, value):
        calls.append((dim, value))
        return tensor.scatter_(dim, indices, value)

    monkeypatch.setitem(wrapper.__globals__, "_index_fill", index_fill)
    assert namespace["_original_assign_draft_token_budget"] is original
    result = wrapper(torch.zeros(4), torch.tensor([1, 3]))
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0, 0.0, 2.0]))
    assert calls == [(0, 2)]
