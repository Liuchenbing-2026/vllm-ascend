"""Greedy byte-exactness probe.  Usage: greedy.py <tag>

Both patches touch values that feed 30 GDN layers' recurrent state selection and (in
mode AB) the KV length FIA attends over. Those failures do not raise: the model keeps
emitting fluent text, only the acceptance rate and eventually the content drift. A
throughput A/B cannot see it. So every patched cell has to clear a greedy
byte-for-byte comparison against the unpatched cell before its numbers mean anything.

Greedy (temperature 0) makes the output a deterministic function of the model state,
so any divergence is a real state bug rather than sampling noise. Prompts are fixed and
long enough to force many decode steps, and the batch is sent concurrently so requests
actually share steps (a serial run would hide batch-ordering bugs entirely).
"""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TAG = sys.argv[1] if len(sys.argv) > 1 else "x"
URL = "http://127.0.0.1:8011/v1/completions"

PROMPTS = [
    "Explain step by step how a binary search tree stays balanced after insertions.",
    "Write a short technical note on why speculative decoding helps latency.",
    "List the tradeoffs between tensor parallelism and pipeline parallelism.",
    "Describe how a garbage collector decides that an object is unreachable.",
    "Summarise the difference between a mutex and a semaphore for a new engineer.",
    "Explain what happens inside a CPU when a page fault is raised.",
    "Give a careful explanation of how gradient checkpointing trades compute for memory.",
    "Walk through the lifecycle of an HTTP request through a reverse proxy.",
    "Explain the CAP theorem and give a concrete example of each tradeoff.",
    "Describe how a log-structured merge tree handles writes and compactions.",
    "Explain why floating point addition is not associative, with an example.",
    "Describe the role of a memory barrier in a lock-free queue.",
    "Explain how a JIT compiler decides which functions to optimise.",
    "Summarise how consistent hashing reduces rebalancing when a node leaves.",
    "Explain the difference between latency and throughput using a concrete system.",
    "Describe how a modern branch predictor uses history to guess a branch.",
]


def one(p):
    body = json.dumps({
        "model": "Qwen36",
        "prompt": p,
        "max_tokens": 192,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"]


with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
    outs = list(ex.map(one, PROMPTS))

blob = "\n<<<---->>>\n".join(outs)
digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
with open("/work/results/greedy_%s.txt" % TAG, "w", encoding="utf-8") as f:
    f.write(blob)
print("GREEDY_SHA %s %s nchars=%d" % (TAG, digest, len(blob)))
