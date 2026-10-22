"""New provenance rejection cases only; fixture records are NOT build evidence.

No library is loaded, no compiler/NPU/model runs, and no old suite is repeated.
Synthetic files/records remain under the requested private report directory.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import shutil
import sys
import traceback
from types import SimpleNamespace

from opp_entry import source_plan
from verify_packed_attention_device import artifact_chain, identity


def fixture(directory):
    root = directory / "source/kv_cache_turbo_quant"
    actual = pathlib.Path(source_plan()["source_root"])
    package_sources = source_plan()["source_sha256"]
    binding_sources = ["code/integration/torch_ext/" + name for name in (
        "tq_packed_attention_contract.cpp", "tq_packed_attention_contract.h",
        "tq_packed_attention_npu.cpp")]
    for relative in [*package_sources, *binding_sources]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual / relative, target)
    vendor = directory / "opp/vendors/customize/op_api"
    header = vendor / "include/aclnn_tq_packed_attention.h"
    op_api = vendor / "lib/libcust_opapi.so"
    binding = directory / "binding/tq_packed_attention.so"
    for p in [header, op_api, binding]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"CPU PROVENANCE FIXTURE ONLY; NOT A COMPILED LIBRARY\n" + p.name.encode())
    binding_record = {
        "fixture_only_not_build_evidence": True, "mode": "npu", "exit_code": 0,
        "registration_compiled": True, "npu_pta_compiled": True,
        "output": identity(binding), "opp_library": identity(op_api), "opp_header": identity(header),
        # A restored artifact may have a different local checkout root.
        "sources": [{"path": "/original/build/kv_cache_turbo_quant/" + relative,
                     "sha256": identity(root / relative)["sha256"]} for relative in binding_sources]
                   + [identity(header), identity(op_api)]}
    package_record = {
        "fixture_only_not_build_evidence": True, "exit_code": 0,
        "package_build_completed": True, "package_sha256": "a" * 64,
        "source_sha256": deepcopy(package_sources)}
    install_record = {
        "fixture_only_not_build_evidence": True, "exit_code": 0, "installed": True,
        "package_sha256": "a" * 64,
        "installed_sha256": {str(p): identity(p)["sha256"] for p in [header, op_api]}}
    args = SimpleNamespace(binding=binding, binding_sha256=identity(binding)["sha256"],
                           op_api=op_api, op_api_sha256=identity(op_api)["sha256"],
                           binding_build_record=directory / "binding-build.json",
                           opp_build_record=directory / "opp-build.json",
                           opp_install_record=directory / "opp-install.json")
    return root, args, [binding_record, package_record, install_record]


def run_case(name, directory):
    root, args, records = fixture(directory)
    binding, package, installed = records
    if name == "stale_binding_source":
        path = root / "code/integration/torch_ext/tq_packed_attention_npu.cpp"
        path.write_text(path.read_text() + "\n// changed after binding compilation\n")
    elif name == "empty_opp_source_manifest":
        package["source_sha256"] = {}
    elif name == "incomplete_opp_source_manifest":
        package["source_sha256"].pop("code/op/op_kernel/tq_packed_attention_core.h")
    elif name in ("different_generated_header", "modified_installed_header"):
        header = pathlib.Path(binding["opp_header"]["path"])
        header.write_bytes(b"CHANGED CPU HEADER FIXTURE; NO SDK DECLARATIONS\n")
        if name == "different_generated_header":
            installed["installed_sha256"][str(header)] = identity(header)["sha256"]
    elif name == "failed_binding_record":
        binding["exit_code"] = 1
    elif name == "failed_package_record":
        package["exit_code"] = 2
    elif name != "matching_relocated_chain":
        raise ValueError(name)
    for path, record in zip([args.binding_build_record, args.opp_build_record,
                             args.opp_install_record], records):
        with path.open("x") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
    rejected = None
    try:
        artifact_chain(args, root)
    except ValueError as error:
        rejected = str(error)
    expected_rejection = name != "matching_relocated_chain"
    return {"case": name, "expected_rejection": expected_rejection, "rejection": rejected,
            "passed": bool(rejected) == expected_rejection}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    fixtures = args.report.with_suffix(".fixtures")
    fixtures.mkdir(parents=True, exist_ok=False)
    result = {"at": datetime.now(timezone.utc).isoformat(), "cases": [], "passed": False,
              "scope": "CPU record-consistency fixtures only; not original NPU build records or device evidence",
              "compiler_or_library_or_device_executed": False, "old_suites_repeated": False}
    try:
        for name in ("matching_relocated_chain", "stale_binding_source", "empty_opp_source_manifest",
                     "incomplete_opp_source_manifest", "different_generated_header",
                     "modified_installed_header", "failed_binding_record", "failed_package_record"):
            result["cases"].append(run_case(name, fixtures / name))
        result["passed"] = all(case["passed"] for case in result["cases"])
        assert "torch_npu" not in sys.modules
    except Exception:
        result["failure"] = traceback.format_exc()
    result["source"] = [identity(p) for p in [pathlib.Path(__file__),
                         pathlib.Path(__file__).with_name("verify_packed_attention_device.py"),
                         pathlib.Path(__file__).with_name("opp_entry.py")]]
    with args.report.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"report": str(args.report), "passed": result["passed"],
                      "cases": len(result["cases"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
