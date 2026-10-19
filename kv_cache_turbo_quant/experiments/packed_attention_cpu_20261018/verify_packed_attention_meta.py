"""New actual C++ Meta registration checks. No CPU or NPU arithmetic backend."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import traceback

import torch


def inputs(batch=2, device="meta"):
    return [torch.empty((batch, 16, 128), dtype=torch.bfloat16, device=device),
            torch.empty((6, 128, 2, 66), dtype=torch.uint8, device=device),
            torch.empty((6, 128, 2, 66), dtype=torch.uint8, device=device),
            torch.empty((batch, 3), dtype=torch.int32, device=device),
            torch.empty((batch,), dtype=torch.int32, device=device),
            torch.empty((16,), dtype=torch.bfloat16, device=device),
            128 ** -0.5, 32768, "/not-opened-by-meta/libcust_opapi.so"]


def checks():
    operator = torch.ops.turboquant.tq_packed_attention.default
    cases = []
    for batch in (0, 2, 16, 60):
        args = inputs(batch)
        output, status = operator(*args)
        assert output.shape == args[0].shape and output.dtype == torch.bfloat16
        assert status.shape == (batch, 2) and status.dtype == torch.int32
        assert output.device.type == status.device.type == "meta"
        assert not torch._C._is_alias_of(output, args[0])
        assert not torch._C._is_alias_of(status, args[4])
        cases.append({"case": "meta_outputs_b" + str(batch), "passed": True})
    invalid = []
    args = inputs(); args[0] = args[0].float(); invalid.append(("query_dtype", args))
    args = inputs(); args[1] = torch.empty((6, 128, 2, 64), dtype=torch.uint8, device="meta"); invalid.append(("row_width", args))
    args = inputs(); args[3] = args[3].long(); invalid.append(("table_dtype", args))
    args = inputs(); args[0] = torch.empty_strided((2, 16, 128), (4096, 256, 2), dtype=torch.bfloat16, device="meta"); invalid.append(("query_stride", args))
    args = inputs(); args[7] = -1; invalid.append(("scratch_range", args))
    args = inputs(); args[6] = float("inf"); invalid.append(("nonfinite_scale", args))
    args = inputs(); args[8] = "relative.so"; invalid.append(("library_identity_path", args))
    for name, args in invalid:
        try:
            operator(*args)
        except RuntimeError:
            cases.append({"case": "meta_reject_" + name, "passed": True})
        else:
            raise AssertionError("Meta accepted " + name)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        args = inputs(device="cpu")
        output, status = operator(*args)
        assert output.shape == args[0].shape and output.device.type == "cpu"
        assert status.shape == (2, 2) and status.dtype == torch.int32
    cases.append({"case": "fake_tensor_metadata_only", "passed": True})
    assert torch._C._dispatch_has_kernel_for_dispatch_key("turboquant::tq_packed_attention", "Meta")
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("turboquant::tq_packed_attention", "CPU")
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("turboquant::tq_packed_attention", "PrivateUse1")
    cases.append({"case": "meta_library_has_no_arithmetic_backend", "passed": True})
    try:
        operator(*inputs(device="cpu"))
    except NotImplementedError:
        cases.append({"case": "real_cpu_execution_rejected", "passed": True})
    else:
        raise AssertionError("CPU fallback unexpectedly exists")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = {"at": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "library": str(args.library.resolve()),
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "checks": [], "passed": False, "npu_executed": False,
              "ascendc_kernel_compiled": False, "old_10_13_17_4_repeated": False}
    try:
        torch.ops.load_library(str(args.library.resolve()))
        result["checks"] = checks()
        result["passed"] = True
    except Exception:
        result["failure"] = traceback.format_exc()
    with args.report.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"report": str(args.report), "passed": result["passed"], "checks": len(result["checks"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
