"""Only new ABI/planner/OPP-preflight checks; no torch or old CPU cases."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    here = pathlib.Path(__file__).resolve().parent
    root = here.parents[1]
    binary = args.out_dir / "device-plan-host-only"
    compile_command = ["/usr/bin/clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                       str(here / "device_plan.cpp"), "-o", str(binary)]
    report = {"at": datetime.now(timezone.utc).isoformat(), "checks": [], "failures": [],
              "scope": "new host ABI and tiling math only; synthetic capacities, not queried hardware",
              "old_e542_shape_or_contract_cases_repeated": False, "npu_or_cann_build_executed": False,
              "resource_application_created": False, "compile_command": compile_command}
    try:
        compiled = subprocess.run(compile_command, capture_output=True, text=True)
        (args.out_dir / "host-compile.stdout").write_text(compiled.stdout)
        (args.out_dir / "host-compile.stderr").write_text(compiled.stderr)
        compiled.check_returncode()
        normal = [16, 16, 2, 256, 128, 16, 196608, 40, 32768]
        cases = [
            ("planned_b16", normal, True),
            ("planned_b60", [60, *normal[1:]], True),
            ("planned_b1", [1, *normal[1:]], True),
            ("empty_batch", [0, *normal[1:]], True),
            ("ub_exact_tile64", [*normal[:6], 158304, *normal[7:]], True),
            ("ub_below_tile64", [*normal[:6], 158303, *normal[7:]], True),
            ("ub_exact_tile16", [*normal[:6], 78240, *normal[7:]], True),
            ("ub_below_tile16", [*normal[:6], 78239, *normal[7:]], False),
            ("insufficient_ub", [*normal[:6], 1000, *normal[7:]], False),
            ("no_platform_cores", [*normal[:7], 0, normal[8]], False),
            ("invalid_gqa", [16, 15, *normal[2:]], False),
            ("job_count_overflow", [4294967295, *normal[1:]], False),
            ("cache_address_overflow", [1, 4294967295, 4294967295, 4294967295,
                                        4294967295, 16, 196608, 40, 32768], False),
            ("negative_cli_dimension", [-1, *normal[1:]], False),
        ]
        for name, values, expected in cases:
            command = [str(binary), *map(str, values)]
            result = subprocess.run(command, capture_output=True, text=True)
            row = {"case": name, "inputs": values, "exit_code": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr}
            assert (result.returncode == 0) == expected, row
            if expected:
                plan = json.loads(result.stdout)
                expected_tile = {"ub_below_tile64": 32, "ub_exact_tile16": 16}.get(name, 64)
                assert plan["abi_bytes"] == 80 and plan["tile_tokens"] == expected_tile
                assert sum(b["bytes"] for b in plan["buffers"]) == plan["ub_bytes"] <= values[6]
                assert plan["user_workspace_bytes"] == 0
                assert plan["work_items"] == values[0] * values[2]
                assert plan["block_dim"] == (min(plan["work_items"], values[7]) or 1)
                assert plan["key_bytes"] == values[3] * values[4] * values[2] * 66
            report["checks"].append(row)
        schema = json.loads((root / "code/op/TqPackedAttention.json").read_text())[0]
        assert [x["name"] for x in schema["input_desc"]] == [
            "query_rot", "key", "value", "table", "lengths", "centroids"]
        assert [x["name"] for x in schema["output_desc"]] == ["attention_rot", "status"]
        assert [x["name"] for x in schema["attr"]] == ["scale", "api_scratch_bytes"]
        report["checks"].append({"case": "operator_schema_order", "passed": True})
        spec = importlib.util.spec_from_file_location("tq_opp_preflight", here / "opp_entry.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        plan = entry.source_plan()
        assert plan["missing_sources"] == ["code/op/op_kernel/tq_packed_attention.cpp"]
        assert not plan["sources_complete"]
        report["checks"].append({"case": "missing_kernel_not_hidden_by_generator", "plan": plan})
        try:
            entry.build(argparse.Namespace(), plan)
        except FileNotFoundError:
            report["checks"].append({"case": "build_stops_before_cann_or_output_dir", "passed": True})
        else:
            raise AssertionError("missing device kernel unexpectedly accepted")
    except Exception:
        report["failures"].append(traceback.format_exc())
    report["passed"] = not report["failures"]
    paths = [here / "device_plan.cpp", here / "opp_entry.py", pathlib.Path(__file__).resolve(),
             root / "code/op/TqPackedAttention.json", root / "code/op/op_kernel/tq_packed_attention_tiling.h",
             root / "code/op/op_host/tq_packed_attention_plan.h", root / "code/op/op_host/tq_packed_attention.cpp"]
    report["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target = args.out_dir / "result.json"
    with target.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(target), "passed": report["passed"], "checks": len(report["checks"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
