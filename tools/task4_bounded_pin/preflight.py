# SPDX-License-Identifier: Apache-2.0
"""Container-only preflight, reached only in an explicitly approved window."""

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

from task4_bounded_pin import validate_lease


def main(mode):
    root = Path("/task4")
    lease = json.loads((root / "lease.json").read_text())
    validate_lease(lease, time.time())
    if time.time() >= lease["start_epoch"] + 180:
        raise RuntimeError("Three-minute deployment/preflight deadline exceeded")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["ascend_sha"] != lease["ascend_sha"]:
        raise ValueError("Lease SHA differs from packaged diagnostic")
    for name, digest in manifest["payload"].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Payload identity mismatch: " + name)
    ascend = Path("/vllm-workspace/vllm-ascend")
    upstream = Path("/vllm-workspace/vllm")

    def git(repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if git(upstream, "rev-parse", "HEAD") != lease["vllm_sha"] or git(upstream, "status", "--porcelain"):
        raise ValueError("Upstream vLLM source identity/cleanliness mismatch")
    if mode == "install":
        if git(ascend, "rev-parse", "HEAD") != lease["ascend_base_sha"] or git(ascend, "status", "--porcelain"):
            raise ValueError("Requires clean rc2 image base; no runtime-file blanket exemptions")
        git(ascend, "fetch", str(root / "diagnostic.bundle"), "HEAD")
        if git(ascend, "rev-parse", "FETCH_HEAD") != lease["ascend_sha"]:
            raise ValueError("Bundle head mismatch")
        git(ascend, "checkout", "--detach", lease["ascend_sha"])
        return
    if mode != "imports" or git(ascend, "rev-parse", "HEAD") != lease["ascend_sha"]:
        raise ValueError("Unexpected preflight mode or diagnostic HEAD")
    if git(ascend, "status", "--porcelain"):
        raise ValueError("Diagnostic tree dirty before imports")
    identities = {}
    for module_name, relative in (
        ("vllm_ascend.task4_bounded_pin", "vllm_ascend/task4_bounded_pin.py"),
        ("vllm_ascend.worker.v2.model_runner", "vllm_ascend/worker/v2/model_runner.py"),
    ):
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path != ascend / relative or digest != manifest["runtime"][relative]:
            raise ValueError("Actual imported source mismatch: " + str(path))
        identities[module_name] = {"path": str(path), "sha256": digest}
    import vllm

    if not Path(vllm.__file__).resolve().is_relative_to(upstream):
        raise ValueError("vLLM actual import is outside verified checkout")
    identities["vllm"] = {"path": vllm.__file__, "sha": lease["vllm_sha"]}
    identities["torch_npu"] = importlib.metadata.version("torch-npu")
    if identities["torch_npu"] != "2.10.0.post7":
        raise ValueError("Different torch_npu wheel than reviewed rc2 image")
    model = Path("/models/dsv4")
    config = json.loads((model / "config.json").read_text())
    shards = sorted(model.glob("*.safetensors"))
    if len(shards) != 74 or config.get("num_hidden_layers") != 43:
        raise ValueError("Unexpected full model shard count/layer count")
    identities["model"] = {
        "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "shards": len(shards),
        "bytes": sum(path.stat().st_size for path in shards),
    }
    (root / "output/import_identity.json").write_text(json.dumps(identities, indent=2))
    validate_lease(lease, time.time())
    if time.time() >= lease["start_epoch"] + 180:
        raise RuntimeError("Preflight exceeded deadline; do not load model")
    (root / "output/preflight_complete.json").write_text(json.dumps({"time": time.time()}))


if __name__ == "__main__":
    main(sys.argv[1])
