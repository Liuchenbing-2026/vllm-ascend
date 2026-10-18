"""CPU function-boundary regression; does not import vLLM or torch_npu.

Load the exact function bodies from the supplied source files. This checks
non-reduce filtering semantics, not device dispatch or end-to-end sampling.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_function(path, name, namespace):
    source = path.read_text()
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def cases():
    ascending = torch.linspace(-3, 3, 128).reshape(1, -1)
    for batch in (1, 7, 8, 9):
        yield f"combined_batch_{batch}", ascending.repeat(batch, 1), [20] * batch, [0.95] * batch
    yield "top_k_only", ascending, [20], None
    yield "top_p_only", ascending, None, [0.95]
    yield "no_filters", ascending, None, None
    yield "full_vocabulary", ascending, [128], [1.0]
    yield "top_one", ascending, [1], [0.95]
    yield "tied_top_p", torch.zeros(1, 128), None, [0.5]
    yield "tied_combined", torch.zeros(1, 128), [20], [0.5]
    yield "mixed_rows", ascending.repeat(4, 1), [1, 20, 64, 128], [0.1, 0.95, 0.5, 1.0]
    masked = ascending.clone()
    masked[:, :32] = -float("inf")
    yield "already_masked", masked, [20], [0.95]
    yield "noncontiguous", ascending.repeat(2, 1).t(), [1] * 128, [0.95] * 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Keep prior run evidence; choose a fresh output path"
    torch.set_num_threads(1)
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    # Fixed vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a.
    assert reference_hash == "bc8c1d91ad624368d2bd87a53c0361e550d28b060bcd460462d2f86565c26426"
    reference = load_function(args.reference, "apply_top_k_top_p_pytorch", {"torch": torch})
    candidate = load_function(
        args.source,
        "_apply_top_k_top_p_pytorch",
        {
            "torch": torch,
            "get_ascend_config": lambda: SimpleNamespace(enable_reduce_sample=False),
            "apply_top_k_top_p_pytorch": reference,
        },
    )
    rows = []
    for name, logits, k_values, p_values in cases():
        k = None if k_values is None else torch.tensor(k_values, dtype=torch.int32)
        p = None if p_values is None else torch.tensor(p_values, dtype=torch.float32)
        expected = reference(logits.clone(), k, p)
        actual = candidate(logits.clone(), k, p)
        expected_mask, actual_mask = torch.isfinite(expected), torch.isfinite(actual)
        error = None
        try:
            assert torch.equal(expected_mask, actual_mask), "retained token mask differs"
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), rtol=0, atol=0)
        except AssertionError as exc:
            error = str(exc)
        rows.append(
            {
                "case": name,
                "passed": error is None,
                "error": error,
                "reference_retained_per_row": expected_mask.sum(-1).tolist(),
                "actual_retained_per_row": actual_mask.sum(-1).tolist(),
                "max_probability_difference": (actual.softmax(-1) - expected.softmax(-1)).abs().max().item(),
            }
        )
    result = {
        "device": "cpu",
        "torch": torch.__version__,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "reference": str(args.reference.resolve()),
        "reference_sha256": reference_hash,
        "scope": "exact extracted functions; non-reduce only; no model or device dispatch",
        "rows": rows,
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"{row['case']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    assert result["passed"] == result["total"], f"{result['total'] - result['passed']} semantic regressions"


if __name__ == "__main__":
    main()
