import json, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/completions"
METRICS = "http://127.0.0.1:8000/metrics"
MODEL = "/data02/GLM-5.3-Flash-w8a8-b0829"
SESSIONS = 4
TURNS = 3
MAX_TOKENS = 128
REPEAT = 80

TEXT = ("The history of computing spans mechanical calculators, vacuum-tube machines, "
        "transistors, integrated circuits, and parallel systems. Each generation reduced "
        "cost and increased throughput while changing the software stack. ")
BASE = (TEXT * REPEAT) + "\nQuestion: Summarize the text in one sentence.\nAnswer:"

def post(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS,
                       "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    return data, time.time() - t0

def metric(name):
    with urllib.request.urlopen(METRICS, timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith(name):
                return float(line.rsplit(" ", 1)[1])
    return None

def session(i, out):
    prompt = f"Session {i}: " + BASE
    rows = []
    for turn in range(TURNS):
        data, elapsed = post(prompt)
        text = data["choices"][0]["text"]
        usage = data.get("usage") or {}
        rows.append({"session": i, "turn": turn, "elapsed": elapsed,
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "text": text})
        prompt = prompt + text + "\nNext question: What is the main idea?\nAnswer:"
    out[i] = rows

def main():
    before_hits = metric("vllm:prefix_cache_hits_total")
    before_frontier = metric("vllm:prefix_cache_frontier_hits_total")
    out = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=SESSIONS) as ex:
        list(ex.map(lambda i: session(i, out), range(SESSIONS)))
    wall = time.time() - t0
    rows = [r for i in range(SESSIONS) for r in out[i]]
    after_hits = metric("vllm:prefix_cache_hits_total")
    after_frontier = metric("vllm:prefix_cache_frontier_hits_total")
    total_prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    total_completion = sum(r["completion_tokens"] or 0 for r in rows)
    summary = {
        "mode": sys.argv[1] if len(sys.argv) > 1 else "unknown",
        "sessions": SESSIONS, "turns": TURNS, "max_tokens": MAX_TOKENS,
        "wall_s": wall,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "prefill_tokens_per_s": total_prompt / wall if wall else None,
        "decode_tokens_per_s": total_completion / wall if wall else None,
        "prefix_cache_hits_delta": (after_hits - before_hits) if before_hits is not None else None,
        "frontier_hits_delta": (after_frontier - before_frontier) if before_frontier is not None else None,
        "mean_latency_s": sum(r["elapsed"] for r in rows) / len(rows),
        "max_latency_s": max(r["elapsed"] for r in rows),
        "requests": rows,
    }
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
