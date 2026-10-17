# SPDX-License-Identifier: Apache-2.0
"""Bounded two-NPU Mooncake READ round trip; no model or TTFT measurement."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import traceback
from pathlib import Path


def aligned_buffer(torch, num_bytes):
    # HCCS registration requires page-table-aligned device memory, as in the
    # production connector's register-region collection.
    alignment = 2 * 1024 * 1024
    allocation = torch.empty(num_bytes + alignment, dtype=torch.uint8, device="npu:0")
    offset = -allocation.data_ptr() % alignment
    tensor = allocation[offset : offset + num_bytes].view(torch.int32)
    assert tensor.data_ptr() % alignment == 0
    return allocation, tensor


def worker(connection, physical_device, local_ip, num_bytes):
    # Spawn imports only standard-library modules before selecting the device.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_device)
    try:
        import torch
        import torch_npu  # noqa: F401

        # Select the runtime device before loading Mooncake's native libraries.
        # Importing them first can corrupt the heap during process teardown.
        torch.npu.set_device(0)
        from mooncake.engine import TransferEngine

        expected = torch.arange(num_bytes // 4, dtype=torch.int32) % 251
        source_allocation, source = aligned_buffer(torch, num_bytes)
        destination_allocation, destination = aligned_buffer(torch, num_bytes)
        source.copy_(expected)
        destination.fill_(-1)
        torch.npu.synchronize()
        engine = TransferEngine()
        if engine.initialize(local_ip, "P2PHANDSHAKE", "ascend", "") != 0:
            raise RuntimeError("Mooncake initialization failed")
        for tensor in (source, destination):
            if engine.register_memory(tensor.data_ptr(), num_bytes) != 0:
                raise RuntimeError("Mooncake NPU registration failed")
        connection.send(
            {
                "ready": True,
                "physical_device": physical_device,
                "session": f"{local_ip}:{engine.get_rpc_port()}",
                "source": source.data_ptr(),
                "destination": destination.data_ptr(),
            }
        )
        while True:
            command = connection.recv()
            if command["op"] == "stop":
                break
            if command["op"] == "read":
                remote = command["remote"]
                status = engine.batch_transfer_sync_read(
                    remote["session"], [destination.data_ptr()], [remote["source"]], [num_bytes]
                )
                if status != 0:
                    raise RuntimeError(f"NPU READ returned {status}")
                torch.npu.synchronize()
                actual = destination.cpu()
                target = expected + command["offset"]
                if not torch.equal(actual, target):
                    raise RuntimeError("NPU READ bytes differ from the expected pattern")
                if command.get("publish_updated"):
                    source.copy_(destination + 1000)
                    torch.npu.synchronize()
                connection.send(
                    {
                        "ok": True,
                        "bytes": num_bytes,
                        "sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                        "torch_peak_bytes": torch.npu.max_memory_allocated(),
                    }
                )
        # Release transport registrations before destroying their NPU storage.
        # Both READs completed before the controller sends this stop command.
        for tensor in (source, destination):
            if engine.unregister_memory(tensor.data_ptr()) != 0:
                raise RuntimeError("Mooncake NPU unregistration failed")
        del tensor, source, destination, source_allocation, destination_allocation
        torch.npu.synchronize()
        torch.npu.empty_cache()
        del engine
    except BaseException:
        connection.send({"ready": False, "ok": False, "error": traceback.format_exc()})
        raise
    finally:
        connection.close()


def receive(connection, timeout):
    if not connection.poll(timeout):
        raise TimeoutError("NPU worker timed out")
    message = connection.recv()
    if message.get("ok") is False or message.get("ready") is False:
        raise RuntimeError(message.get("error", message))
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.devices[0] == args.devices[1] or min(args.devices) < 0:
        parser.error("select two different physical NPUs")
    if not 0 < args.bytes <= 16 * 1024 * 1024 or args.bytes % 4:
        parser.error("bytes must be a multiple of four, at most 16 MiB")
    output = Path(args.output)
    if output.exists():
        parser.error("preserve existing evidence; use a new output path")
    result = {
        "scope": "Real NPU transport buffers, not full KV lifecycle, model quality or TTFT",
        "devices": args.devices,
        "bytes_each_direction": args.bytes,
        "success": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["vllm", "vllm-ascend", "torch", "torch-npu", "mooncake-transfer-engine-npu"]
        },
    }
    context = multiprocessing.get_context("spawn")
    workers = []
    connections = []
    try:
        for device in args.devices:
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, device, args.local_ip, args.bytes))
            process.start()
            child.close()
            workers.append(process)
            connections.append(parent)
        metadata = [receive(connection, args.timeout) for connection in connections]
        connections[1].send({"op": "read", "remote": metadata[0], "offset": 0, "publish_updated": True})
        result["p_to_d"] = receive(connections[1], args.timeout)
        connections[0].send({"op": "read", "remote": metadata[1], "offset": 1000})
        result["d_to_p"] = receive(connections[0], args.timeout)
        result["success"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        for connection in connections:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                connection.send({"op": "stop"})
        for process in workers:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in connections:
            connection.close()
        result["worker_exitcodes"] = [process.exitcode for process in workers]
        result["success"] &= all(process.exitcode == 0 for process in workers)
        output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
