"""Explicit, isolated OPP reproduction entry; default action only reports a plan.

No docker/SSH/NPU launcher is provided. Build never installs. Missing real source
files block before CANN generation; source completeness is not compile evidence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import traceback


def source_plan():
    root = pathlib.Path(__file__).resolve().parents[2]
    sources = [
        root / "code/op/TqPackedAttention.json",
        root / "code/op/op_host/tq_packed_attention.cpp",
        root / "code/op/op_host/tq_packed_attention_plan.h",
        root / "code/op/op_kernel/tq_packed_attention_tiling.h",
        root / "code/op/op_kernel/tq_packed_attention_core.h",
        root / "code/op/op_kernel/tq_packed_attention.cpp",
    ]
    return {"op": "TqPackedAttention", "target": "ascend910b", "planned_cann": "9.1.0",
            "source_root": str(root),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources if p.is_file()},
            "missing_sources": [str(p.relative_to(root)) for p in sources if not p.is_file()],
            "sources_complete": all(p.is_file() for p in sources),
            "kernel_stage": "AIV source candidate; target CANN compilation pending",
            "scratch_policy": "32KiB is only a reservation; no high-level ReduceSum/Softmax used; target SDK unverified",
            "cann_compile_verified": False, "sdk_scratch_requirement_verified": False,
            "npu_kernel_verified": False, "launches_npu": False}


def sdk_command(cann, arguments, cwd, log):
    command = ["bash", "-c", 'set -eo pipefail; source "$1/set_env.sh"; shift; exec "$@"',
               "tq-opp-entry", str(cann), *map(str, arguments)]
    with log.with_suffix(".command.json").open("x") as handle:
        json.dump(command, handle, indent=2)
        handle.write("\n")
    with log.open("x") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return command


def build(args, result):
    if result["missing_sources"]:
        raise FileNotFoundError("missing real sources; refusing generated dummy kernel: "
                                + ", ".join(result["missing_sources"]))
    if sys.platform != "linux":
        raise RuntimeError("CANN package build requires the approved Linux toolchain environment")
    if args.cann_root is None or args.work_dir is None or args.python_bin is None:
        raise ValueError("build requires --cann-root, --work-dir and --python-bin")
    cann = args.cann_root.resolve()
    if not (cann / "set_env.sh").is_file():
        raise FileNotFoundError(cann / "set_env.sh")
    if not args.python_bin.is_file():
        raise FileNotFoundError(args.python_bin)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    result["work_dir"] = str(work)
    result["cann_set_env_sha256"] = hashlib.sha256((cann / "set_env.sh").read_bytes()).hexdigest()
    root = pathlib.Path(result["source_root"])
    generated = work / "generated"
    result["generate_command"] = sdk_command(
        cann, ["msopgen", "gen", "-i", root / "code/op/TqPackedAttention.json",
               "-c", "ai_core-ascend910b", "-lan", "cpp", "-out", generated],
        work, work / "generate.log")
    for relative in result["source_sha256"]:
        if relative.startswith(("code/op/op_host/", "code/op/op_kernel/")):
            source = root / relative
            target = generated / source.relative_to(root / "code/op")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    presets_path = generated / "CMakePresets.json"
    presets = json.loads(presets_path.read_text())
    for preset in presets["configurePresets"]:
        cache = preset.setdefault("cacheVariables", {})
        cache["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": "ascend910b"}
        cache["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": str(cann)}
        cache["ASCEND_PYTHON_EXECUTABLE"] = {"type": "STRING", "value": str(args.python_bin.resolve())}
    presets_path.write_text(json.dumps(presets, indent=2) + "\n")
    # Generated build.sh only: no -u/-s/-e/-a flags and no installation here.
    result["build_command"] = sdk_command(cann, ["bash", "build.sh"], generated, work / "build.log")
    packages = list(generated.rglob("*.run"))
    if len(packages) != 1:
        raise RuntimeError("expected one OPP package, found " + repr(packages))
    result["package"] = str(packages[0])
    result["package_sha256"] = hashlib.sha256(packages[0].read_bytes()).hexdigest()
    result["package_build_completed"] = True
    result["installed"] = False


def install(args, result):
    if sys.platform != "linux":
        raise RuntimeError("OPP staging requires the approved Linux environment")
    if args.package is None or args.package_sha256 is None or args.opp_prefix is None:
        raise ValueError("install requires --package, --package-sha256 and --opp-prefix")
    package = args.package.resolve()
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    if digest != args.package_sha256:
        raise ValueError("package SHA256 mismatch")
    prefix = args.opp_prefix.resolve()
    if prefix.exists():
        raise FileExistsError("OPP prefix must be new; existing vendor trees are never replaced")
    prefix.mkdir(parents=True, exist_ok=False)
    log = prefix.parent / (prefix.name + "-install.log")
    command = ["bash", str(package), "--quiet", "--install-path=" + str(prefix)]
    with log.open("x") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    vendor = prefix / "vendors/customize"
    expected = [vendor / "bin/set_env.bash", vendor / "op_api/include/aclnn_tq_packed_attention.h",
                vendor / "op_api/lib/libcust_opapi.so"]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError("installed OPP artifacts missing: " + repr(missing))
    result.update(package=str(package), package_sha256=digest, opp_prefix=str(prefix),
                  install_command=command, installed=True,
                  installed_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in expected})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("plan", "build", "install"), default="plan")
    parser.add_argument("--record", type=pathlib.Path)
    parser.add_argument("--cann-root", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--python-bin", type=pathlib.Path)
    parser.add_argument("--package", type=pathlib.Path)
    parser.add_argument("--package-sha256")
    parser.add_argument("--opp-prefix", type=pathlib.Path)
    args = parser.parse_args()
    if args.action != "plan" and args.record is None:
        parser.error("build/install require a fresh --record path")
    if args.record is not None and args.record.exists():
        raise FileExistsError("refusing to overwrite an existing record")
    result = source_plan()
    result.update(action=args.action, at=datetime.now(timezone.utc).isoformat(),
                  host=platform.platform(), permission="entry is not a shared-resource allocation")
    exit_code = 0
    try:
        if args.action == "build":
            build(args, result)
        elif args.action == "install":
            install(args, result)
    except Exception:
        result["failure"] = traceback.format_exc()
        exit_code = 2
    result["exit_code"] = exit_code
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with args.record.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
