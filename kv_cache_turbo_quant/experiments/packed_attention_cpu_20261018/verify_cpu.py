"""New CPU-only packed-attention cases; never launches a device or old tests."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import sys
import traceback

import torch


def fixture(lengths, query_heads=16, kv_heads=2, page_size=128, seed=20261018,
            zero_norm=False, strided=False, norm_scale=1.0):
    import kvtq_store as store

    generator = torch.Generator().manual_seed(seed)
    max_pages = max(((n + page_size - 1) // page_size for n in lengths), default=0)
    pages = max(1, len(lengths) * max_pages + 2)
    table = torch.full((len(lengths), max_pages + 1), -1, dtype=torch.int32)
    permutation = torch.randperm(pages, generator=generator)
    cursor = 0
    for batch, length in enumerate(lengths):
        count = (length + page_size - 1) // page_size
        table[batch, :count] = permutation[cursor:cursor + count].int()
        cursor += count
    caches = []
    for _ in range(2):
        indices = torch.randint(0, 256, (pages, page_size, kv_heads, 64),
                                generator=generator, dtype=torch.uint8)
        norm = (torch.rand((pages, page_size, kv_heads), generator=generator)
                * norm_scale).to(torch.bfloat16)
        if zero_norm:
            norm.zero_()
        caches.append(store._pack_rows(indices, norm))
    query = torch.randn((len(lengths), query_heads, 128), generator=generator).to(torch.bfloat16)
    rotation, _, _ = store._get_consts(torch.device("cpu"))
    if strided:
        query = torch.stack((query, query), dim=-1)[..., 0]
        table = torch.stack((table, table), dim=-1)[..., 0]
    return query, caches[0], caches[1], table, list(lengths), rotation


def compare_case(name, inputs, tiles):
    from kvtq_attention_cpu import dense_decode_oracle_cpu, packed_decode_cpu

    snapshots = [v.clone() if isinstance(v, torch.Tensor) else list(v) for v in inputs]
    expected = dense_decode_oracle_cpu(*inputs)
    for tile in tiles:
        storage = torch.empty((*expected.shape, 2), dtype=torch.bfloat16)
        output = storage[..., 0]
        actual = packed_decode_cpu(*inputs, tile_tokens=tile, output=output)
        assert actual is output
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if name == "zero_norm":
            assert not torch.count_nonzero(actual)
    for value, snapshot in zip(inputs, snapshots):
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, snapshot), "input mutation"
        else:
            assert value == snapshot, "length mutation"
    return {"case": name, "lengths": inputs[4], "query_shape": list(inputs[0].shape),
            "cache_shape": list(inputs[1].shape), "tiles": list(tiles),
            "final_bf16_bitwise_equal": True, "inputs_unchanged": True}


def check_rejections():
    from kvtq_attention_cpu import packed_decode_cpu

    inputs = fixture([65])
    invalid = []
    for length in (0, -1, 10000, 1.5):
        args = list(inputs)
        args[4] = [length]
        invalid.append(("length_" + str(length), args, {}))
    for page in (-1, inputs[1].shape[0]):
        args = list(inputs)
        args[3] = inputs[3].clone()
        args[3][0, 0] = page
        invalid.append(("active_page_" + str(page), args, {}))
    args = list(inputs)
    args[0] = args[0].float()
    invalid.append(("query_dtype", args, {}))
    args = list(inputs)
    args[0] = args[0][:, :3]
    invalid.append(("gqa_divisibility", args, {}))
    invalid.extend(("tile_" + str(tile), list(inputs), {"tile_tokens": tile})
                   for tile in (0, -1, 2.5, True))
    invalid.append(("nonfinite_scale", list(inputs), {"scale": float("inf")}))
    passed = []
    for name, args, kwargs in invalid:
        try:
            packed_decode_cpu(*args, **kwargs)
        except ValueError:
            passed.append(name)
        else:
            raise AssertionError("invalid contract accepted: " + name)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--case", action="append", help="select only new or failed cases")
    args = parser.parse_args()
    integration = pathlib.Path(__file__).resolve().parents[2] / "code" / "integration"
    sys.path.insert(0, str(integration))
    torch.set_num_threads(2)
    specs = [
        ("short_b16", [20] * 16, {}, [64]),
        ("long_b60", [2000] * 60, {}, [64]),
        ("ragged_tile_page_tails", [1, 63, 64, 65, 127, 128, 129, 1999, 2000, 2001], {}, [31, 64, 128]),
        ("long_single_8192", [8192], {}, [64, 127]),
        ("zero_norm", [1, 65, 129], {"zero_norm": True}, [64]),
        ("noncontiguous_query_table", [63, 129], {"strided": True}, [64]),
        ("finite_large_norm", [65, 129], {"norm_scale": 1024.0}, [64]),
        ("equal_q_kv_heads", [65, 129], {"query_heads": 4, "kv_heads": 4}, [64]),
        ("alternate_gqa", [129], {"query_heads": 8, "kv_heads": 2, "page_size": 64}, [31, 64]),
        ("empty_batch", [], {}, [64]),
    ]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "new CPU-only FP64 attention math; existing store4 codec; final BF16 zero tolerance",
              "python": sys.version.split()[0], "torch": torch.__version__, "cpu_threads": 2,
              "cases": [], "failures": [], "npu_tested": False,
              "npu_kernel_implemented": False, "performance_claim": None,
              "historical_tests_repeated": False, "full_model_goal_achieved": False}
    if args.case:
        unknown = set(args.case) - {row[0] for row in specs} - {"contract_rejections"}
        if unknown:
            raise ValueError("unknown case names: " + repr(unknown))
        specs = [row for row in specs if row[0] in args.case]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, lengths, kwargs, tiles in specs:
        inputs = fixture(lengths, **kwargs)
        try:
            report["cases"].append(compare_case(name, inputs, tiles))
        except Exception:
            failure = args.report.with_name(args.report.stem + "_" + name + "_failure.pt")
            if failure.exists():
                raise FileExistsError("refusing to overwrite a failure fixture")
            torch.save(inputs, failure)
            report["failures"].append({"case": name, "traceback": traceback.format_exc(),
                                       "fixture": str(failure)})
    if not args.case or "contract_rejections" in args.case:
        try:
            report["contract_rejections"] = check_rejections()
        except Exception:
            report["failures"].append({"case": "contract_rejections", "traceback": traceback.format_exc()})
    report["cpu_math_passed"] = not report["failures"]
    report["source_sha256"] = {
        str(path.relative_to(integration.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (integration / "kvtq_store.py", integration / "kvtq_attention_cpu.py",
                     pathlib.Path(__file__).resolve())}
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(args.report), "cpu_math_passed": report["cpu_math_passed"],
                      "passed_cases": len(report["cases"]), "failures": len(report["failures"])}))
    return 0 if report["cpu_math_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
