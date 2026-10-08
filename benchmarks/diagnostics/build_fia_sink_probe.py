# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a standalone probe binding from a pinned upstream source archive.

This does not install or replace vLLM's extension. Build/install the matching
two-operator CANN package first and export its vendor path before loading.
"""

import argparse
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    args = parser.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    original = (args.source / "csrc/torch_binding.cpp").read_text()
    start = original.index("TORCH_LIBRARY_FRAGMENT(_C_ascend, m)")
    end = original.index("\n#endif", start)
    schema = args.build / "schema.cpp"
    schema.write_text("#include <torch/library.h>\n" + original[start:end] + "\n")
    binding = args.source / "csrc/attention/fused_infer_attention_score_v2_sink/torch_binding"
    npu = Path(torch_npu.__file__).parent
    library = load(
        name="dspark_fia_sink_probe",
        sources=[str(schema)] + [str(p) for p in sorted(binding.glob("*.cpp"))],
        extra_include_paths=[str(binding), str(npu / "include"), str(args.cann / "include")],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu", f"-L{args.cann / 'lib64'}", "-lascendcl"],
        build_directory=str(args.build),
        is_python_module=False,
        verbose=True,
    )
    print("PROBE_LIBRARY", library)


if __name__ == "__main__":
    main()
