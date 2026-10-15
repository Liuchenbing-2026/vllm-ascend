"""Separate prefill/steady-decode diagnostic windows, excluded from throughput."""

import argparse
import asyncio
import codecs
import hashlib
import json
import pathlib
import random
import time
from datetime import datetime, timezone

import aiohttp

URL = "http://127.0.0.1:18377"


async def main(args):
    payloads = []
    for i in range(args.concurrency):
        payload = {
            "model": "qwen3-30b-tq-ab",
            "prompt": random.Random(20261010 + i).choices(
                range(1000, 10000), k=args.input_len
            ),
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "seed": 20261010,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payloads.append(json.dumps(payload, separators=(",", ":")).encode())
    records = [{"id": i, "text_events": 0} for i in range(args.concurrency)]
    first_seen = set()
    all_started = asyncio.Event()
    events = []
    began = time.perf_counter()
    output = pathlib.Path(args.output)
    timeout = aiohttp.ClientTimeout(total=1200)

    def snapshot():
        output.with_suffix(".progress.json").write_text(
            json.dumps(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - began,
                    "requests_with_text": len(first_seen),
                    "events": events,
                    "requests": records,
                },
                indent=2,
            )
            + "\n"
        )

    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=args.concurrency + 4)
    ) as session:

        async def api(action, phase):
            before = time.perf_counter()
            utc = datetime.now(timezone.utc).isoformat()
            async with session.post(URL + "/" + action) as response:
                body = await response.text()
                event = {
                    "phase": phase,
                    "action": action,
                    "sent_utc": utc,
                    "http_status": response.status,
                    "elapsed_s": time.perf_counter() - before,
                    "response": body[:300],
                }
                events.append(event)
                snapshot()
                if response.status != 200:
                    raise RuntimeError(event)

        async def one(i):
            r = records[i]
            start = time.perf_counter()
            decoder = codecs.getincrementaldecoder("utf-8")()
            carry = ""
            try:
                async with session.post(
                    URL + "/v1/completions",
                    data=payloads[i],
                    headers={"Content-Type": "application/json"},
                ) as response:
                    r["http_status"] = response.status
                    if response.status != 200:
                        raise RuntimeError((await response.text())[:1000])
                    async for chunk in response.content.iter_any():
                        carry += decoder.decode(chunk)
                        while "\n\n" in carry:
                            event, carry = carry.split("\n\n", 1)
                            for line in event.splitlines():
                                if (
                                    not line.startswith("data: ")
                                    or line[6:] == "[DONE]"
                                ):
                                    continue
                                value = json.loads(line[6:])
                                if value.get("error"):
                                    raise RuntimeError(str(value["error"]))
                                if value.get("usage"):
                                    r["usage"] = value["usage"]
                                for choice in value.get("choices", []):
                                    if choice.get("text"):
                                        r["text_events"] += 1
                                        if i not in first_seen:
                                            r["ttft_s"] = time.perf_counter() - start
                                            first_seen.add(i)
                                            if len(first_seen) == args.concurrency:
                                                all_started.set()
                                    if choice.get("finish_reason"):
                                        r["finish_reason"] = choice["finish_reason"]
                assert r["usage"] == {
                    "prompt_tokens": args.input_len,
                    "completion_tokens": args.output_len,
                    "total_tokens": args.input_len + args.output_len,
                }
                assert r["finish_reason"] == "length"
                r["status"] = "success"
            except Exception as error:
                r.update(status="failed", error=repr(error))
                raise
            finally:
                r["latency_s"] = time.perf_counter() - start

        profiling = False
        try:
            if args.phase == "prefill":
                await api("start_profile", "prefill")
                profiling = True
            load = asyncio.gather(*(one(i) for i in range(args.concurrency)))
            if args.phase == "decode":
                # The initial cohort is warm ongoing traffic. Only start the
                # bounded diagnostic window once every request has produced
                # text, otherwise the long-input profile misses steady decode.
                gate = asyncio.create_task(all_started.wait())
                done, _ = await asyncio.wait(
                    [load, gate], timeout=180, return_when=asyncio.FIRST_COMPLETED
                )
                if gate not in done:
                    gate.cancel()
                    if load in done:
                        await load
                    raise RuntimeError(
                        "full concurrency did not reach steady decode before timeout"
                    )
                if any(r.get("status") == "success" for r in records):
                    raise RuntimeError(
                        "request already ended before steady-decode window"
                    )
                await api("start_profile", "decode")
                profiling = True
            await asyncio.sleep(3)
            await api("stop_profile", args.phase)
            profiling = False
            await load
        finally:
            if profiling:
                try:
                    await api("stop_profile", "exception_cleanup")
                except Exception as error:
                    events.append({"cleanup_error": repr(error)})
            snapshot()
    result = {
        "mode": args.mode,
        "scope": "profiling only, excluded from formal throughput",
        "phase": args.phase,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "prepared_payload_sha256": [hashlib.sha256(p).hexdigest() for p in payloads],
        "duration_s": time.perf_counter() - began,
        "successful": sum(r.get("status") == "success" for r in records),
        "events": events,
        "requests": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["requests", "prepared_payload_sha256"]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["prefill", "decode"], required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    asyncio.run(main(parser.parse_args()))
