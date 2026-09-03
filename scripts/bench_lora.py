"""LoRA serve benchmark: TTFT / TPOT / throughput via /v1/completions.

For each prompt: one streaming request (TTFT + stream wall time) and one
non-streaming request (usage token counts + output text for accuracy diff).
Configs: K=1 sequential and K=4 parallel.  Results -> JSON (argv[1]).
"""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:7519/v1/completions"
MAX_TOKENS = 128

# Fixed template with single-digit ids -> all prompts have identical token
# counts, so only one prefill shape is ever captured (no per-shape capture
# noise in TTFT).
PROMPTS = [
    "Describe in detail how the process of photosynthesis works in plants. Task number %d." % i
    for i in range(1, 9)
]


def _req(payload: dict, timeout=900):
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_metrics(prompt: str):
    """Returns (ttft_s, stream_wall_s)."""
    body = {"model": "openscad", "prompt": prompt, "max_tokens": MAX_TOKENS,
            "temperature": 0, "seed": 42, "stream": True}
    t0 = time.monotonic()
    ttft = None
    with _req(body) as r:
        for line in r:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = ev.get("choices") or []
            if choices and (choices[0].get("text")):
                if ttft is None:
                    ttft = time.monotonic() - t0
    wall = time.monotonic() - t0
    return ttft, wall


def complete(prompt: str):
    """Returns (output_text, completion_tokens, wall_s)."""
    body = {"model": "openscad", "prompt": prompt, "max_tokens": MAX_TOKENS,
            "temperature": 0, "seed": 42, "stream": False}
    t0 = time.monotonic()
    with _req(body) as r:
        data = json.loads(r.read().decode())
    wall = time.monotonic() - t0
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return choice.get("text") or "", int(usage.get("completion_tokens") or 0), wall


def bench_one(prompt: str):
    ttft, wall_s = stream_metrics(prompt)
    text, n_tok, wall_n = complete(prompt)
    tpot = (wall_s - ttft) / max(n_tok - 1, 1)
    return {
        "prompt": prompt,
        "ttft": round(ttft, 4),
        "stream_wall": round(wall_s, 4),
        "total_wall": round(wall_n, 4),
        "tokens": n_tok,
        "tpot_ms": round(tpot * 1000, 2),
        "tps": round(n_tok / wall_n, 2),
        "text": text,
    }


def bench_parallel(prompts, k):
    results = [None] * len(prompts)
    lock = threading.Lock()
    idx = 0

    def worker():
        nonlocal idx
        while True:
            with lock:
                if idx >= len(prompts):
                    return
                i, p = idx, prompts[idx]
                idx += 1
            results[i] = bench_one(p)

    threads = [threading.Thread(target=worker) for _ in range(k)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    return results, wall


def summary(items, tag, wall):
    n = len(items)
    total_tok = sum(r["tokens"] for r in items)
    return {
        "tag": tag,
        "n_requests": n,
        "wall_s": round(wall, 3),
        "throughput_tps": round(total_tok / wall, 2),
        "ttft_mean": round(sum(r["ttft"] for r in items) / n, 4),
        "ttft_p50": round(sorted(r["ttft"] for r in items)[n // 2], 4),
        "tpot_ms_mean": round(sum(r["tpot_ms"] for r in items) / n, 2),
        "tps_mean": round(sum(r["tps"] for r in items) / n, 2),
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench.json"

    print("warmup...", flush=True)
    for p in PROMPTS:
        complete(p)
    # also capture the batch-4 prefill shape so K=4 measurement is capture-free
    print("warmup K=4 ...", flush=True)
    bench_parallel(PROMPTS, 4)

    report = {"config": {"max_tokens": MAX_TOKENS}, "runs": []}
    for k in (1, 4):
        print(f"K={k} ...", flush=True)
        items, wall = bench_parallel(PROMPTS, k)
        report["runs"].append({"k": k, "summary": summary(items, f"k{k}", wall),
                               "requests": items})
        for r in items:
            print(f"  ttft={r['ttft']:.3f}s tpot={r['tpot_ms']:.1f}ms "
                  f"tok={r['tokens']} tps={r['tps']:.1f}", flush=True)

    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
