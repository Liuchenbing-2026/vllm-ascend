# SPDX-License-Identifier: Apache-2.0
"""Package a clean committed diagnostic; creates no container or reservation."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def prepare(repo, destination, model):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    base = "8fde6d4e335fe300e2b949aab0e31659cce0811d"
    if git("status", "--porcelain"):
        raise ValueError("Package only a clean committed worktree")
    git("merge-base", "--is-ancestor", base, "HEAD")
    sha = git("rev-parse", "HEAD")
    destination.mkdir(parents=True, exist_ok=False)
    for path in (repo / "tools/task4_bounded_pin").iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copy2(path, destination / path.name)
    helper = repo / "vllm_ascend/task4_bounded_pin.py"
    shutil.copy2(helper, destination / helper.name)
    git("bundle", "create", str(destination / "diagnostic.bundle"), "HEAD", "^" + base)
    lease = {
        "status": "PENDING",
        "scope": "task4-rc2-stage-a-full-model-pin",
        "lease_id": "PENDING_SUPERVISOR",
        "start_epoch": 0,
        "end_epoch": 1500,
        "physical_devices": list(range(8)),
        "allocator_ranks": [0],
        "steps_bytes": [32 << 20, 128 << 20, 256 << 20, 1024 << 20],
        "rank_limit_bytes": 1440 << 20,
        "total_limit_bytes": 1440 << 20,
        "vllm_sha": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "ascend_base_sha": base,
        "ascend_sha": sha,
        "image": (
            "quay.m.daocloud.io/ascend/vllm-ascend@sha256:"
            "8e375bf1cb4a98ecf7d487c7f98a28d1c3a7e24f7f8cfbadbb37914ed68b2a30"
        ),
        "image_id": "sha256:0faf08fb90dd21807266cb719b4947b0111b3bd5f58c46424810a8ecabe82b98",
        "model_path": str(model),
    }
    (destination / "lease.json").write_text(json.dumps(lease, indent=2) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.iterdir()
        if path.is_file() and path.name != "lease.json"
    }
    runtime = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ("vllm_ascend/task4_bounded_pin.py", "vllm_ascend/worker/v2/model_runner.py")
    }
    (destination / "manifest.json").write_text(
        json.dumps({"payload": hashes, "runtime": runtime, "ascend_sha": sha}, indent=2) + "\n"
    )
    return lease


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Future Linux host path; no model read/allocation"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2], args.output.resolve(), args.model_path), indent=2))
