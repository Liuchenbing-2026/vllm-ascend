"""CPU counterexample for the service failure and untracked LUT mutation contract."""

import importlib.util
import json
from pathlib import Path

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path("/ws/paired_read_inference_20261014")
    old = load("old", "/ws/paired_read_20261014/code/integration/kvtq_native_pair.py")
    fixed = load("fixed", str(root / "code/integration/kvtq_native_pair.py"))
    before = old.NativePagedPairReader.__new__(old.NativePagedPairReader)
    after = fixed.NativePagedPairReader.__new__(fixed.NativePagedPairReader)
    with torch.inference_mode():
        lookup = torch.arange(512, dtype=torch.float32).to(torch.bfloat16).view(256, 2)
        assert torch.is_inference(lookup)
        try:
            before._prepare_lookup(lookup)
        except RuntimeError as error:
            assert "Inference tensors do not track version counter" in str(error)
            failure = str(error)
        else:
            raise AssertionError("original did not reproduce the reported service failure")
        for _ in range(3):
            prepared = after._prepare_lookup(lookup)
            assert torch.equal(prepared[:512].view(torch.int16), lookup.flatten().view(torch.int16))
            lookup.add_(1)
    normal = torch.ones(256, 2, dtype=torch.bfloat16)
    first = after._prepare_lookup(normal)
    assert after._prepare_lookup(normal) is first
    normal.mul_(2)
    updated = after._prepare_lookup(normal)
    assert updated is not first
    assert torch.equal(updated[:512].view(torch.int16), normal.flatten().view(torch.int16))
    (root / "cpu-regression.json").write_text(json.dumps({"original_failure": failure, "untracked_mutations_passed": 3,
                                                      "tracked_reuse_and_mutation_passed": True, "precision_unchanged": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
