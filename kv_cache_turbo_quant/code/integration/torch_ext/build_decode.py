"""Build the host-only TQ decode orchestration extension."""
from pathlib import Path
from torch.utils.cpp_extension import load


def build(verbose=False):
    return load(
        name="tq_decode",
        sources=[str(Path(__file__).with_name("tq_decode.cpp"))],
        extra_cflags=["-O2", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(build(verbose=True).__file__)
