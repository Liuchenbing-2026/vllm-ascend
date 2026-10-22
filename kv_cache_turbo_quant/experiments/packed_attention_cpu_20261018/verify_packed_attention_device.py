"""Future allocated-device BF16 FIA gate. This file has not been run on NPU.

No speed, model quality or stability acceptance is inferred from this test.
Explicit artifact hashes/device/cases are required; no server/container/lease
or installation is created. Failures retain tensor fixtures and raw traceback.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import accumulate
import json
import math
import pathlib
import sys
import traceback


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def artifact_chain(args, root):
    """Read actual existing build/install records; never manufacture a chain."""
    bindings = json.loads(args.binding_build_record.read_text())
    if (not bindings.get("npu_pta_compiled") or not bindings.get("registration_compiled")
            or bindings.get("mode") != "npu" or bindings.get("exit_code") != 0):
        raise ValueError("binding record is not a successful NPU PTA build")
    package = json.loads(args.opp_build_record.read_text())
    installed = json.loads(args.opp_install_record.read_text())
    if bindings["output"]["sha256"] != args.binding_sha256:
        raise ValueError("binding build identity mismatch")
    if bindings["opp_library"]["sha256"] != args.op_api_sha256:
        raise ValueError("binding was built for a different OPP library")
    if (not package.get("package_build_completed") or not installed.get("installed")
            or package.get("exit_code") != 0 or installed.get("exit_code") != 0):
        raise ValueError("requires successful original OPP build and install records")
    if package["package_sha256"] != installed["package_sha256"]:
        raise ValueError("OPP build/install package mismatch")
    recorded_libraries = {str(pathlib.Path(p).resolve()): digest
                          for p, digest in installed["installed_sha256"].items()}
    if recorded_libraries.get(str(args.op_api.resolve())) != args.op_api_sha256:
        raise ValueError("OPP install record does not identify the actual library")

    # Reuse the builder's complete source list. An empty/partial map must not
    # turn source verification into a vacuous pass. Relocated checkouts are
    # permitted, but every current source must match the recorded build input.
    from opp_entry import source_plan

    plan = source_plan()
    manifest = package.get("source_sha256", {})
    if not plan["sources_complete"] or set(manifest) != set(plan["source_sha256"]):
        raise ValueError("OPP source manifest is incomplete or unexpected")
    for relative, digest in manifest.items():
        if identity(root / relative)["sha256"] != digest:
            raise ValueError("OPP source differs from the actual tested source: " + relative)

    header = args.op_api.resolve().parent.parent / "include/aclnn_tq_packed_attention.h"
    header_digest = identity(header)["sha256"]
    if (recorded_libraries.get(str(header)) != header_digest
            or bindings["opp_header"]["sha256"] != header_digest):
        raise ValueError("binding/generated/installed OPP header identity mismatch")
    binding_sources = bindings.get("sources", [])
    required = {"code/integration/torch_ext/" + name: identity(root / "code/integration/torch_ext" / name)["sha256"]
                for name in ("tq_packed_attention_contract.cpp", "tq_packed_attention_contract.h",
                             "tq_packed_attention_npu.cpp")}
    required.update({"aclnn_tq_packed_attention.h": header_digest,
                     "libcust_opapi.so": args.op_api_sha256})
    for relative, digest in required.items():
        matches = [entry for entry in binding_sources
                   if pathlib.Path(entry["path"]).as_posix().endswith("/" + relative)]
        if len(matches) != 1 or matches[0]["sha256"] != digest:
            raise ValueError("binding source differs from the actual tested source: " + relative)
    return [identity(p) for p in [args.binding_build_record, args.opp_build_record, args.opp_install_record]]


def specification(name):
    if name == "short_b16":
        return [20] * 16, {}, None
    if name == "long_b60":
        return [2000] * 60, {}, None
    if name == "ragged_page_tails":
        return [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1999, 2000, 2001, 2199, 2200], {}, None
    if name == "long_single_8192":
        return [8192], {}, None
    if name == "zero_norm":
        return [33, 129], {"zero_norm": True}, None
    if name == "nondefault_stream":
        return [33, 129], {}, "stream"
    if name in ("status_bad_length", "status_bad_page", "status_nonfinite_norm"):
        return [33, 129], {}, name
    raise ValueError("unknown device case: " + name)


def compare_bits(actual, expected):
    import torch

    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16 or actual.shape != expected.shape:
        raise AssertionError("BF16 comparison shape/dtype mismatch")
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    mismatches = int(torch.count_nonzero(actual.contiguous().view(torch.int16) !=
                                         expected.contiguous().view(torch.int16)).item())
    return {"finite": finite, "bf16_mismatches": mismatches,
            "passed": finite and mismatches == 0, "rtol": 0, "atol": 0}


def run_case(name, adapter, device, scale, fixture_state):
    import torch
    import torch_npu
    import kvtq_store as store
    from verify_cpu import fixture  # Reuse the existing fixture/codec; do not run its checks.

    if store._MSE_BITS != 4 or store._ROW_BYTES != 66:
        raise ValueError("requires the unchanged store4 codec configuration")
    lengths, options, trait = specification(name)
    cpu = fixture(lengths, **options)
    query, key, value, table = [v.to(device) for v in cpu[:4]]
    rotation = cpu[5].to(device)
    length_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    _, _, centroids = store._get_consts(device)
    args = [query, key, value, table, length_tensor, centroids, rotation]
    # Keep the actual mutated CPU inputs too, so a negative case can be replayed.
    expected_status = torch.zeros((len(lengths), key.shape[2]), dtype=torch.int32)
    if trait == "status_bad_length":
        length_tensor[0] = 0
        expected_status[0].fill_(1)
    elif trait == "status_bad_page":
        table[0, 0] = key.shape[0]
        expected_status[0].fill_(2)
    elif trait == "status_nonfinite_norm":
        physical = int(cpu[3][0, 0])
        key[physical, 0, 0, 64] = 0x80
        key[physical, 0, 0, 65] = 0x7f  # Exact little-endian BF16 +Inf norm.
        expected_status[0, 0] = 3
    fixture_state.update(case=name, lengths=lengths, scale=scale,
                         inputs=[v.detach().cpu().clone() for v in args],
                         expected_status=expected_status)
    snapshots = [v.clone() for v in args]
    stream = torch_npu.npu.Stream(device=device) if trait == "stream" else torch_npu.npu.current_stream(device)
    if trait == "stream":
        stream.wait_stream(torch_npu.npu.current_stream(device))
    with torch_npu.npu.stream(stream):
        query_rot = (query.float() @ rotation.T).to(torch.bfloat16)
        actual, status = adapter.rotated(query_rot, key, value, table, length_tensor, centroids, scale=scale)
        host_status = status.cpu()
        fixture_state.update(query_rot=query_rot.detach().cpu(),
                             actual_rot=actual.detach().cpu(), status=host_status)
        if not torch.equal(host_status, expected_status):
            raise AssertionError("status mismatch: " + repr(host_status.tolist()))
        if trait and trait.startswith("status_"):
            groups = query.shape[1] // key.shape[2]
            bad_heads = (expected_status != 0).repeat_interleave(groups, dim=1).to(device)
            if torch.count_nonzero(actual[bad_heads]).item():
                raise AssertionError("failed task output is not zero")
            result = {"case": name, "status_expected": expected_status.tolist(), "status_passed": True,
                      "precision_scope": "status-only negative case; not a BF16 precision pass"}
        else:
            # Inactive -1 entries are sanitized only in the baseline's private
            # table copy, because the unchanged dense helper gathers all slots.
            safe_table = table.clone()
            for b, length in enumerate(lengths):
                safe_table[b, (length + key.shape[1] - 1) // key.shape[1]:] = 0
            dense_key = store._dequant_dense_reference(None, key.view(torch.bfloat16), safe_table, lengths)
            dense_value = store._dequant_dense_reference(None, value.view(torch.bfloat16), safe_table, lengths)
            golden, _ = torch_npu.npu_fused_infer_attention_score(
                query=query_rot, key=dense_key, value=dense_value,
                actual_seq_lengths=list(range(1, len(lengths) + 1)),
                actual_seq_lengths_kv=list(accumulate(lengths)),
                input_layout="TND", num_heads=query.shape[1], num_key_value_heads=key.shape[2],
                block_size=key.shape[1], scale=scale, sparse_mode=0)
            golden = golden.view_as(actual)
            final = (actual.float() @ rotation).to(torch.bfloat16)
            final_golden = (golden.float() @ rotation).to(torch.bfloat16)
            fixture_state.update(golden_rot=golden.detach().cpu(), final=final.detach().cpu(),
                                 final_golden=final_golden.detach().cpu())
            result = {"case": name, "core": compare_bits(actual, golden),
                      "whole_attention": compare_bits(final, final_golden), "status_passed": True}
            fixture_state["comparison"] = result
            if not result["core"]["passed"] or not result["whole_attention"]["passed"]:
                raise AssertionError("unchanged BF16 FIA bitwise gate failed: " + repr(result))
        for old, current in zip(snapshots, args):
            if not torch.equal(old.view(torch.uint8), current.view(torch.uint8)):
                raise AssertionError("input mutation")
    stream.synchronize()
    result.update(inputs_unchanged=True, passed=True, nondefault_stream=trait == "stream")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=pathlib.Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--op-api", type=pathlib.Path, required=True)
    parser.add_argument("--op-api-sha256", required=True)
    parser.add_argument("--binding-build-record", type=pathlib.Path, required=True)
    parser.add_argument("--opp-build-record", type=pathlib.Path, required=True)
    parser.add_argument("--opp-install-record", type=pathlib.Path, required=True)
    parser.add_argument("--device", required=True, help="explicit allocated npu:<logical id>")
    parser.add_argument("--api-scratch-bytes", type=int, required=True)
    parser.add_argument("--scale", type=float, default=128 ** -0.5)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if not args.device.startswith("npu:") or not math.isfinite(args.scale):
        parser.error("requires an explicit NPU device and finite original baseline scale")
    for name in args.case:
        specification(name)
    here = pathlib.Path(__file__).resolve().parent
    integration = here.parents[1] / "code/integration"
    sys.path.insert(0, str(integration))
    from kvtq_packed_attention import NativePackedAttention, checked_library

    report = {"at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "device": args.device,
              "requested_binding": str(args.binding), "requested_op_api": str(args.op_api),
              "scale_original_double": args.scale, "baseline_scale_adjusted_to_candidate": False,
              "api_scratch_bytes": args.api_scratch_bytes, "sdk_scratch_requirement_verified": False,
              "cases": [], "passed": False, "full_model_quality_or_speed_acceptance": False,
              "source": [identity(p) for p in [pathlib.Path(__file__), integration / "kvtq_packed_attention.py",
                                               integration / "kvtq_store.py", here / "verify_cpu.py"]]}
    fixture_state = {}
    try:
        checked_library(args.binding, args.binding_sha256)
        checked_library(args.op_api, args.op_api_sha256)
        report.update(binding=identity(args.binding), op_api=identity(args.op_api))
        contract = json.loads((here / "device_validation_contract.json").read_text())
        if identity(integration / "kvtq_store.py")["sha256"] != contract["store_codec_sha256"]:
            raise ValueError("codec differs from the fixed precision contract")
        report["artifact_chain"] = artifact_chain(args, here.parents[1])
        import torch
        import torch_npu

        torch_npu.npu.set_device(args.device)
        device = torch.device(args.device)
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
        adapter = NativePackedAttention(args.binding, args.binding_sha256, args.op_api, args.op_api_sha256,
                                        api_scratch_bytes=args.api_scratch_bytes)
        with torch.inference_mode():
            for name in args.case:
                fixture_state.clear()
                report["cases"].append(run_case(name, adapter, device, args.scale, fixture_state))
        report["passed"] = True
    except Exception:
        report["failure"] = traceback.format_exc()
        if "comparison" in fixture_state:
            report["failed_comparison"] = fixture_state["comparison"]
        if fixture_state:
            fixture_path = args.report.with_name(args.report.stem + "-failed-input.pt")
            with fixture_path.open("xb") as f:
                torch.save(fixture_state, f)
            report["failed_fixture"] = identity(fixture_path)
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({"report": str(args.report), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
