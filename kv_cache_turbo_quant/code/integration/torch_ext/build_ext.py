"""Build the turboquant torch glue extension (cached under TORCH_EXTENSIONS_DIR)."""
import os

import torch  # noqa: F401
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load

_SP = os.path.dirname(torch_npu.__file__)
_ASCEND = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")
if not os.path.isdir(os.path.join(_ASCEND, "include")):
    _ASCEND = "/usr/local/Ascend/cann-9.1.0"
_OPAPI = os.path.join(_ASCEND, "opp/vendors/customize/op_api")


def build(verbose=False):
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turboquant_torch.cpp")
    return load(
        name="turboquant_torch",
        sources=[src],
        extra_include_paths=[
            os.path.join(_SP, "include"),
            os.path.join(_ASCEND, "include"),
            os.path.join(_OPAPI, "include"),
        ],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[
            f"-L{os.path.join(_SP, 'lib')}", "-ltorch_npu",
            f"-L{os.path.join(_ASCEND, 'lib64')}", "-lascendcl", "-lnnopbase", "-ldl",
            f"-Wl,-rpath,{os.path.join(_SP, 'lib')}",
            f"-Wl,-rpath,{os.path.join(_ASCEND, 'lib64')}",
        ],
        verbose=verbose,
    )


if __name__ == "__main__":
    ext = build(verbose=True)
    print("BUILD OK:", ext.__file__)