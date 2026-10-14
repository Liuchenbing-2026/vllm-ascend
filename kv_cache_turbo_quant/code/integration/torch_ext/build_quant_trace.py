"""Build the additional reader glue without replacing the original quantizer."""

from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def build(verbose=False):
    cann = Path("/usr/local/Ascend/cann-9.1.0")
    package = Path(torch_npu.__file__).parent
    return load(
        name="tq_quant_trace",
        sources=[str(Path(__file__).with_name("tq_quant_trace.cpp"))],
        extra_include_paths=[str(package / "include"), str(cann / "include")],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[
            f"-L{package / 'lib'}",
            "-ltorch_npu",
            f"-L{cann / 'lib64'}",
            "-lascendcl",
            "-lnnopbase",
            "-ldl",
            f"-Wl,-rpath,{package / 'lib'}",
            f"-Wl,-rpath,{cann / 'lib64'}",
        ],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(build(verbose=True).__file__)
