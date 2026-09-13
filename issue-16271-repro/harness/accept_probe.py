#!/usr/bin/env python3
"""Model-level judge for the #16271 draft-path change.

Sends a fixed set of natural-language prompts greedily (concurrency 1) and
records, per arm:
  * the generated text for every prompt (so two arms can be diffed)
  * the server's SpecDecoding metrics during the window

A broken draft attention shows up immediately as a collapsed acceptance length
(the target rejects everything the drafter proposes), while the final text stays
valid because the target verifies every token.

Usage: accept_probe.py <tag> [n_tokens]
"""

import json
import sys
import time
import urllib.request

PROMPTS = [
    "Explain in detail how a hash table handles collisions, and compare chaining with open addressing.",
    "Write a short technical summary of how gradient descent with momentum differs from plain gradient descent.",
    "描述一下操作系统里虚拟内存的分页机制，以及缺页中断的处理流程。",
    "List the main steps a compiler takes to turn C source code into an executable, and explain each briefly.",
    "What is the CAP theorem in distributed systems? Give a concrete example of a system for each trade-off.",
    "请解释 TCP 三次握手的过程，以及为什么不能用两次握手。",
    "Describe how a B-tree index speeds up range queries in a relational database compared to a hash index.",
    "Explain the difference between processes and threads, including memory layout and context switch cost.",
    "写一段说明：为什么浮点数比较不能直接用等号，应该怎么做。",
    "Summarise how HTTPS establishes a secure channel, from DNS lookup to the first encrypted byte.",
    "Explain what a race condition is, give an example, and describe two ways to prevent it.",
    "解释一下什么是垃圾回收中的分代假设，以及它为什么有效。",
]


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    base = "http://127.0.0.1:8100"
    out_path = "/nt/logs/accept_%s.json" % tag

    results = []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        r = post(
            base + "/v1/completions",
            {
                "model": "qwen3",
                "prompt": p,
                "max_tokens": n_tokens,
                "temperature": 0.0,
                "seed": 1234,
            },
        )
        txt = r["choices"][0]["text"]
        results.append({"i": i, "prompt": p, "text": txt, "usage": r.get("usage")})
        print("[%2d] %d chars  %.1fs" % (i, len(txt), time.time() - t0), flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "n_tokens": n_tokens, "results": results}, f, ensure_ascii=False, indent=1)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
