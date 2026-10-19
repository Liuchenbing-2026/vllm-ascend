"""Build only real Torch registration/PTA sources, never an AscendC kernel.

meta-cpu needs the installed CPU torch and C++ compiler, with no ninja or SDK.
npu requires an already generated/installed OPP header and library; it is a
future shared build entry, not an allocation or a device execution launcher.
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


def file_identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build(args, result):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    here = pathlib.Path(__file__).resolve().parent
    sources = [here / "tq_packed_attention_contract.cpp"]
    tracked = sources + [here / "tq_packed_attention_contract.h"]
    includes = list(include_paths())
    libraries = list(library_paths())
    links = ["-ltorch", "-ltorch_cpu", "-lc10"]
    if args.mode == "npu":
        if sys.platform != "linux":
            raise RuntimeError("NPU PTA build requires the allocated Linux environment")
        if args.cann_root is None or args.opp_vendor is None:
            raise ValueError("npu requires --cann-root and --opp-vendor")
        import torch_npu

        cann = args.cann_root.resolve()
        vendor = args.opp_vendor.resolve()
        header = vendor / "op_api/include/aclnn_tq_packed_attention.h"
        library = vendor / "op_api/lib/libcust_opapi.so"
        for p in [cann / "set_env.sh", header, library]:
            if not p.is_file():
                raise FileNotFoundError(p)
        package = pathlib.Path(torch_npu.__file__).resolve().parent
        includes += [str(package / "include"), str(cann / "include"), str(header.parent)]
        libraries += [str(package / "lib"), str(cann / "lib64")]
        links += ["-ltorch_npu", "-lascendcl", "-lnnopbase", "-ldl"]
        sources.append(here / "tq_packed_attention_npu.cpp")
        tracked += [sources[-1], header, library]
        result["opp_header"] = file_identity(header)
        result["opp_library"] = file_identity(library)
        result["torch_npu_version"] = torch_npu.__version__
    compiler = args.compiler or shutil.which("c++")
    if not compiler:
        raise FileNotFoundError("C++ compiler unavailable")
    output = args.out_dir / "tq_packed_attention.so"
    command = [compiler, "-std=c++17", "-O1", "-fPIC", "-Wall", "-Wextra", "-Werror",
               "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    command += ["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
    for directory in includes:
        command += ["-isystem", directory]
    command += [str(p) for p in sources]
    for directory in libraries:
        command += ["-L" + directory, "-Wl,-rpath," + directory]
    command += links + ["-o", str(output)]
    result.update(command=command, torch_version=torch.__version__, python=sys.executable,
                  sources=[file_identity(p) for p in tracked],
                  compiler_version=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0])
    with (args.out_dir / "compile.stdout").open("x") as stdout, (args.out_dir / "compile.stderr").open("x") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    result.update(output=file_identity(output), registration_compiled=True,
                  npu_pta_compiled=args.mode == "npu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("meta-cpu", "npu"), required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--compiler")
    parser.add_argument("--cann-root", type=pathlib.Path)
    parser.add_argument("--opp-vendor", type=pathlib.Path)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = {"at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "host": platform.platform(), "registration_compiled": False,
              "npu_pta_compiled": False, "ascendc_kernel_compiled": False,
              "npu_executed": False, "allocation_created": False}
    code = 0
    try:
        build(args, result)
    except Exception:
        result["failure"] = traceback.format_exc()
        code = 1
    result["exit_code"] = code
    record = args.out_dir / "build.json"
    with record.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"record": str(record), "compiled": result["registration_compiled"], "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
