#!/usr/bin/env python3
"""Pooled acceptance over the *accept-probe* phase of a serve log only.

The clean-form leg was killed after arm cl1's probe and before its first bench,
so the published pooled numbers (which cover probe + warm + measure) have no
counterpart for cl1. What every arm does have is the probe: the same 12 prompts,
concurrency 1, issued immediately after READY. Restricting every log to that
phase makes the arms comparable again on the one judge that separated them.

The cut is exact rather than time-based: accept_probe.py issues exactly 12
completions and nothing else does until the warm bench starts, so every
SpecDecoding window logged before the 13th completion response covers probe
traffic and only probe traffic.
"""
import re
import sys

K = 8
N_PROBE = 12
spec = re.compile(r"Accepted:\s*(\d+)\s*tokens?,\s*Drafted:\s*(\d+)\s*tokens?")
done = re.compile(r'POST /v1/completions HTTP/1\.1" 200 OK')

for path in sys.argv[1:]:
    a = d = n = 0
    seen = 0
    for line in open(path, errors="replace"):
        if done.search(line):
            seen += 1
            if seen > N_PROBE:
                break
        m = spec.search(line)
        if m:
            a += int(m.group(1))
            d += int(m.group(2))
            n += 1
    tag = path.split("serve_")[-1].replace(".log", "")
    if d:
        r = a / d
        print("%-8s windows=%-3d Accepted=%-6d Drafted=%-6d ratio=%.6f  "
              "probe accept_len=%.4f" % (tag, n, a, d, r, 1 + K * r))
    else:
        print("%-8s no spec windows in the probe phase" % tag)
