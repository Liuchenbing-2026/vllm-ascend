# SPDX-License-Identifier: Apache-2.0
"""In-container, pre-model import identity gate and actual worker cache check."""

import argparse
import hashlib
import importlib
import json
import runpy
import sys
import time
import urllib.request
from pathlib import Path

from kv_diagnostic_readiness import check_guard_status, write_json


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_imports(contract, observed, report, *, importer=importlib.import_module):
    if observed["image_id"] != contract["image_id"] or not observed["readonly_rootfs"]:
        raise ValueError("unreviewed image or writable root filesystem")
    required = {
        "vllm",
        "vllm_ascend",
        "torch",
        "torch_npu",
        "torch._C",
        "torch_npu._C",
        "vllm_ascend.vllm_ascend_C",
        "mooncake.engine",
    }
    if not required <= contract["modules"].keys():
        raise ValueError("runtime contract omits required framework/transport identities")
    interpreter = Path(sys.executable).resolve(strict=True)
    report.update(
        image_id=observed["image_id"],
        modules={},
        python={
            "executable": sys.executable,
            "resolved": str(interpreter),
            "sha256": file_digest(interpreter),
            "version": sys.version,
        },
    )
    mounts = [
        *observed["mounts"],
        *({"Destination": path, "Source": "tmpfs", "RW": True} for path in observed.get("tmpfs", {})),
    ]
    for name, expected in contract["modules"].items():
        module = importer(name)
        # Resolve in this container namespace. A host-side absolute interpreter
        # symlink is deliberately not used to judge in-image availability.
        origin = Path(module.__file__).resolve(strict=True)
        digest = file_digest(origin)
        covers = [
            item
            for item in mounts
            if origin == Path(item["Destination"]) or Path(item["Destination"]) in origin.parents
        ]
        mount = max(covers, key=lambda item: len(item["Destination"])) if covers else None
        kind = "native" if origin.suffix in (".so", ".pyd") else "python"
        report["modules"][name] = {
            "origin": str(origin),
            "sha256": digest,
            "kind": kind,
            "provenance": "host-bind" if mount else "image-rootfs",
            "covering_mount": mount,
            "module_version": str(getattr(module, "__version__", "unknown")),
        }
        if expected.get("origin") and str(origin) != expected["origin"]:
            raise ValueError("actual import path differs: " + name)
        if expected.get("kind") and kind != expected["kind"]:
            raise ValueError("actual module type differs: " + name)
        if expected.get("sha256") and digest != expected["sha256"]:
            raise ValueError("actual import file hash differs: " + name)
        # An immutable, exact image identifies unmounted binaries. Host-bound
        # binaries/overlays additionally require an independently reviewed hash.
        if mount and (mount.get("RW") or not expected.get("sha256")):
            raise ValueError("unreviewed host-bound import or writable source: " + name)
        if kind == "native" and mount and expected.get("binary_source_image_id") != contract["image_id"]:
            raise ValueError("host-bound native binary lacks image provenance: " + name)
    report["imports_verified"] = True


def check_block_argument(arguments):
    values = []
    for index, value in enumerate(arguments):
        if value == "--block-size":
            values.append(arguments[index + 1] if index + 1 < len(arguments) else None)
        elif value.startswith("--block-size="):
            values.append(value.split("=", 1)[1])
    if values != ["128"]:
        raise ValueError("this fixed diagnosis requires one explicit --block-size 128")


def serve(
    contract_path, observed_path, receipt_path, arguments, *, importer=importlib.import_module, runner=runpy.run_module
):
    report = {"phase": "pre_model_gate", "server_handoff_allowed": False, "imports_verified": False}
    try:
        check_block_argument(arguments)
        contract, observed = json.loads(contract_path.read_text()), json.loads(observed_path.read_text())
        verify_imports(contract, observed, report, importer=importer)
        config = importer("vllm.config").CacheConfig(block_size=128)
        if config.block_size != 128:
            raise ValueError("constructed cache configuration differs from explicit block128")
        report.update(
            server_handoff_allowed=True,
            explicit_block_size=128,
            pre_model_cache_config_block_size=config.block_size,
            effective_worker_cache_check="required after engine initialization, before first prompt",
        )
    except BaseException as error:
        report["error"] = repr(error)
        write_json(receipt_path, report, exclusive=True)
        raise
    write_json(receipt_path, report, exclusive=True)
    # Same interpreter, import namespace and readonly mounts as the actual API
    # server, without replacing its model/worker/connector implementation.
    previous_argv = sys.argv
    try:
        sys.argv = ["vllm.entrypoints.openai.api_server", *arguments]
        return runner("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv


def verify_worker_cache(results):
    if len(results) != 1 or results[0]["config_block_size"] != 128 or results[0]["group_block_sizes"] != [128]:
        raise ValueError("actual TP1 worker configuration/spec must both use block128")
    return results[0]


def check_caches(container_id, *, root=Path("/output"), timeout=110):
    deadline = time.monotonic() + timeout
    while True:
        status = json.loads((root / "guard-status.json").read_text())
        if (
            (root / "STOP").exists()
            or (root / "guard-exit.json").exists()
            or time.time() - status["checked_at_epoch"] > 3
        ):
            raise RuntimeError("independent guard lost before worker cache check")
        if status["required_listeners_observed"]:
            check_guard_status(root / "guard-status.json", container_id)
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("model listener readiness deadline")
        time.sleep(0.2)
    result = {}
    for role, port in (("P", 29510), ("D", 29511)):
        check_guard_status(root / "guard-status.json", container_id)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/collective_rpc",
            data=json.dumps({"method": "mooncake_diagnostic_cache_config", "timeout": 10}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            result[role] = verify_worker_cache(json.load(response)["results"])
    write_json(root / "actual-worker-cache-config.json", result, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "check-cache"))
    parser.add_argument("--contract", type=Path, default=Path("/output/runtime-contract.json"))
    parser.add_argument("--observed", type=Path, default=Path("/output/container-observed.json"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--container-id")
    args, server_args = parser.parse_known_args()
    if args.action == "check-cache":
        check_caches(args.container_id)
    else:
        if args.receipt is None:
            parser.error("serve requires a new per-role receipt path")
        serve(args.contract, args.observed, args.receipt, server_args[1:] if server_args[:1] == ["--"] else server_args)


if __name__ == "__main__":
    main()
