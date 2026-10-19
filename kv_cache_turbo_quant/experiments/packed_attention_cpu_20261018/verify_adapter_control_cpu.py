"""Only new adapter/gate control logic; no NPU import, kernel or model tests."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import sys
import traceback
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-build-record", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    here = pathlib.Path(__file__).resolve().parent
    integration = here.parents[1] / "code/integration"
    sys.path.insert(0, str(integration))
    from kvtq_packed_attention import NativePackedAttention, checked_library
    from verify_packed_attention_device import artifact_chain, compare_bits

    result = {"at": datetime.now(timezone.utc).isoformat(), "checks": [], "passed": False,
              "device_or_model_executed": False, "old_suites_repeated": False}
    try:
        record = json.loads(args.meta_build_record.read_text())
        actual_library = pathlib.Path(record["output"]["path"])
        checked_library(actual_library, record["output"]["sha256"])
        try:
            checked_library(actual_library, "0" * 64)
        except ValueError:
            result["checks"].append("real_cpu_library_hash_accept_and_mismatch_rejection")
        else:
            raise AssertionError("library identity mismatch accepted")
        NativePackedAttention.require_success(torch.zeros((2, 2), dtype=torch.int32))
        try:
            NativePackedAttention.require_success(torch.tensor([[0, 3], [0, 0]], dtype=torch.int32))
        except RuntimeError:
            result["checks"].append("nonzero_status_is_failure")
        else:
            raise AssertionError("nonzero status accepted")
        finite = torch.tensor([0, 1, -1, 128], dtype=torch.bfloat16)
        assert compare_bits(finite, finite.clone())["passed"]
        positive_zero = torch.tensor([0.0], dtype=torch.bfloat16)
        negative_zero = torch.tensor([-0.0], dtype=torch.bfloat16)
        assert not compare_bits(positive_zero, negative_zero)["passed"]
        result["checks"].append("bf16_signed_zero_is_bitwise_not_numeric_equality")
        nonfinite = torch.tensor([float("nan"), float("inf")], dtype=torch.bfloat16)
        assert not compare_bits(nonfinite, nonfinite.clone())["passed"]
        result["checks"].append("identical_nonfinite_bits_cannot_pass_precision")
        # Uses the actual CPU-only build record, and intentionally nonexistent
        # unused OPP paths: it must stop BEFORE opening them or importing NPU.
        probe = SimpleNamespace(binding_build_record=args.meta_build_record,
                                opp_build_record=pathlib.Path("/not-accessed/opp-build.json"),
                                opp_install_record=pathlib.Path("/not-accessed/opp-install.json"))
        try:
            artifact_chain(probe, here.parents[1])
        except ValueError as error:
            assert "not a successful NPU PTA build" in str(error)
            result["checks"].append("real_meta_build_cannot_be_used_as_npu_build_evidence")
        else:
            raise AssertionError("CPU Meta build accepted for device verification")
        assert "torch_npu" not in sys.modules
        result["passed"] = True
    except Exception:
        result["failure"] = traceback.format_exc()
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [pathlib.Path(__file__), integration / "kvtq_packed_attention.py",
                                here / "verify_packed_attention_device.py"]}
    result["meta_build_record_sha256"] = hashlib.sha256(args.meta_build_record.read_bytes()).hexdigest()
    with args.report.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"report": str(args.report), "passed": result["passed"], "checks": len(result["checks"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
