# SPDX-License-Identifier: Apache-2.0
"""Capture D source, P imported and P recomputed KV without changing a golden.

Run only on task-owned idle P/D servers with the snapshot worker extension,
both prefix caches disabled and sufficient source/producer retention time.
All inputs, outputs and tensor dumps stay in the private output directory.
"""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from kv_diagnostic_readiness import check_guard_status
from mooncake_kv_snapshot import compare_snapshots, token_digest


class GuardedClient:
    """Refuse each HTTP operation if the independent host guard is stale."""

    def __init__(self, client, args):
        self.client, self.args = client, args

    def _call(self, method, *args, **kwargs):
        check_guard_status(self.args.guard_status, self.args.container_id)
        return getattr(self.client, method)(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self._call("get", *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._call("post", *args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._call("stream", *args, **kwargs)


def verify_transport_endpoints(client, args):
    """Observe dynamic TE ports before any prompt, without guessing their value."""
    status = check_guard_status(args.guard_status, args.container_id)
    endpoints = {}
    for name, url in (("P", args.prefill), ("D", args.decode)):
        result = (
            client.post(url + "/collective_rpc", json={"method": "mooncake_transport_endpoint", "timeout": 10})
            .raise_for_status()
            .json()["results"]
        )
        if len(result) != 1:
            raise ValueError("expected TP1 endpoint observation")
        endpoint = result[0]
        # The guard has already checked the approved IP/range of owned sockets.
        for port_name in ("rpc_port", "handshake_port"):
            if not any(
                item["local_ip"] == endpoint["host"]
                and item["local_port"] == endpoint[port_name]
                and item["state"] == "0A"
                for item in status["endpoints"]
            ):
                raise ValueError("advertised TE/handshake endpoint not observed by independent host guard")
        endpoints[name] = endpoint
    return endpoints


def metric(client, url):
    text = client.get(url + "/metrics").raise_for_status().text
    return sum(
        float(line.split()[-1]) for line in text.splitlines() if line.startswith("vllm:mooncake_bytes_transferred_sum{")
    )


def prompt_body(prompt, descriptor=None):
    params = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    if descriptor is not None:
        params.update(descriptor)
    return {
        "model": "bidir",
        "prompt": prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
        "kv_transfer_params": params,
    }


def prefill(client, args, prompt, descriptor=None):
    request_id = (
        "___prefill_addr_"
        + args.prefill.removeprefix("http://")
        + "___decode_addr_"
        + args.decode.removeprefix("http://")
        + "_"
        + uuid.uuid4().hex
    )
    body = prompt_body(prompt, descriptor)
    result = (
        client.post(args.prefill + "/v1/completions", json=body, headers={"X-Request-Id": request_id})
        .raise_for_status()
        .json()
    )
    return request_id, result


def decode(client, args, request_id, prompt, p_result):
    body = prompt_body(prompt)
    body.update(stream=True, max_tokens=256)
    body.pop("min_tokens")
    body["kv_transfer_params"] = p_result["kv_transfer_params"]
    output, descriptor, done = [], None, False
    with client.stream(
        "POST", args.decode + "/v1/completions", json=body, headers={"X-Request-Id": request_id}
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            text = line[5:].strip()
            if text == "[DONE]":
                done = True
                continue
            item = json.loads(text)
            if "error" in item:
                raise RuntimeError(item["error"])
            if item.get("kv_transfer_params"):
                descriptor = item["kv_transfer_params"]
            for choice in item.get("choices", []):
                output.extend(choice.get("token_ids") or [])
    if not done or len(output) != 256 or descriptor is None:
        raise RuntimeError("incomplete decode stream or missing retained metadata")
    return output, descriptor


def capture(client, url, descriptor, prefix, block_size, root, phase):
    count = len(prefix)
    blocks = descriptor["remote_block_ids"]
    if len(blocks) != 1:
        raise ValueError("only one cache group is supported")
    specification = {
        "engine_id": descriptor["remote_engine_id"],
        "request_id": descriptor["remote_request_id"],
        "block_ids": blocks[0][: count // block_size],
        "num_tokens": count,
        "prefix_token_ids": prefix,
        "directory": str(root / phase),
        "phase": phase,
    }
    result = (
        client.post(
            url + "/collective_rpc",
            json={"method": "capture_mooncake_kv", "timeout": 120, "args": [json.dumps(specification)]},
        )
        .raise_for_status()
        .json()
    )
    if len(result["results"]) != 1:
        raise RuntimeError("expected one TP1 worker snapshot")
    snapshot = result["results"][0]
    (root / (phase + "-receipt.json")).write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-status", type=Path, required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    report = {
        "scope": "precision diagnosis only, no performance claim",
        "golden_changed": False,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "complete": False,
    }
    try:
        if report["fixture_sha256"] != args.fixture_sha256:
            raise AssertionError("fixture fingerprint differs from the reviewed input")
        with httpx.Client(timeout=120, trust_env=False) as raw_client:
            client = GuardedClient(raw_client, args)
            report["transport_endpoints"] = verify_transport_endpoints(client, args)
            rid, p = prefill(client, args, fixture["first"])
            first, descriptor = decode(client, args, rid, fixture["first"], p)
            if first != fixture["first_reference"]:
                raise AssertionError("first-turn source output differs from original golden")
            retained = descriptor["remote_num_tokens"]
            source_blocks = descriptor["remote_block_ids"]
            if len(source_blocks) != 1 or not source_blocks[0] or retained % len(source_blocks[0]):
                raise ValueError("retained metadata must identify complete blocks in one group")
            block_size = retained // len(source_blocks[0])
            prefix = (fixture["first"] + first[:-1])[:retained]
            if len(prefix) != retained or prefix != fixture["second"][:retained]:
                raise AssertionError("D source prefix differs from P second input")
            report.update(retained_tokens=retained, prefix_sha256_u32le=token_digest(prefix))
            source = capture(client, args.decode, descriptor, prefix, block_size, args.output, "D-source")
            before = metric(client, args.prefill)
            rid, p = prefill(client, args, fixture["second"], descriptor)
            after = metric(client, args.prefill)
            imported = capture(
                client, args.prefill, p["kv_transfer_params"], prefix, block_size, args.output, "P-imported"
            )
            reuse_request, reuse_prefill = rid, p
            reference_request, reference_prefill = prefill(client, args, fixture["second"])
            recomputed = capture(
                client,
                args.prefill,
                reference_prefill["kv_transfer_params"],
                prefix,
                block_size,
                args.output,
                "P-recomputed",
            )
            report["three_snapshots_captured"] = True
            transfer = compare_snapshots(source["manifest"], imported["manifest"], numerical=True)
            arithmetic = compare_snapshots(imported["manifest"], recomputed["manifest"], numerical=True)
            for name, data in (("D-to-P-byte-comparison", transfer), ("imported-vs-recomputed", arithmetic)):
                (args.output / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
            report.update(
                D_to_P_bytes_exact=transfer["all_bytes_exact"],
                imported_equals_recomputed_bytes=arithmetic["all_bytes_exact"],
                snapshot_bytes=source["total_bytes"],
            )
            if not transfer["all_bytes_exact"] or not all(
                item["all_finite"] for data in (transfer, arithmetic) for item in data["results"]
            ):
                raise AssertionError("KV transfer byte mismatch or nonfinite snapshot; stop before further decode")
            report["three_snapshots_validated"] = True
            # All three snapshots exist before this optional model verdict.
            # Producer pages remain pinned until their individual decode ACKs;
            # this diagnosis requires sufficient producer abort time, not reuse
            # of a completed request's released pages.
            reuse, _ = decode(client, args, reuse_request, fixture["second"], reuse_prefill)
            report.update(
                reuse_output_ids=reuse,
                reuse_output_exact=reuse == fixture["second_reference"],
                first_different_index=next(
                    (i for i, pair in enumerate(zip(reuse, fixture["second_reference"])) if pair[0] != pair[1]), None
                ),
            )
            if not report["reuse_output_exact"]:
                raise AssertionError("reuse output differs from original golden; stop without reference decode")
            reference, _ = decode(client, args, reference_request, fixture["second"], reference_prefill)
            report.update(no_hint_output_ids=reference, no_hint_output_exact=reference == fixture["second_reference"])
            if not report["no_hint_output_exact"]:
                raise AssertionError("no-hint output differs from original golden")
            # Prometheus receives connector stats asynchronously; wait only for
            # observation, outside every model request and snapshot operation.
            stats_deadline = time.monotonic() + 15
            while after - before < source["total_bytes"] and time.monotonic() < stats_deadline:
                time.sleep(0.2)
                after = metric(client, args.prefill)
        report.update(
            complete=True,
            reverse_read_bytes=after - before,
        )
        report["acceptance_pass"] = (
            report["reuse_output_exact"]
            and report["no_hint_output_exact"]
            and report["D_to_P_bytes_exact"]
            and report["reverse_read_bytes"] == report["snapshot_bytes"]
        )
    except BaseException as error:
        report["acceptance_pass"] = False
        report["error"] = repr(error)
        raise
    finally:
        fixture_changed = hashlib.sha256(args.fixture.read_bytes()).hexdigest() != report["fixture_sha256"]
        if fixture_changed:
            report.update(acceptance_pass=False, fixture_changed=True)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        # This entrypoint owns one fixture only. The external host process stops
        # the entire owned container after success or first failure, including
        # any producer pages deliberately kept alive for the third snapshot.
        (args.guard_status.parent / "STOP").touch(mode=0o600, exist_ok=True)
        if fixture_changed:
            raise AssertionError("original fixture changed during diagnosis")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("_ids")}))
    return 0 if report["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
