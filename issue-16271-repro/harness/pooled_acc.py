#!/usr/bin/env python3
"""Pooled acceptance for one run, from vLLM's own per-window counters.

This is the independent side of the `delta == rejected` test: the trace gives
mean(delta), the log gives the acceptance ratio, and only if delta really is the
rejection count do they satisfy mean(delta) = K * (1 - Accepted/Drafted).
"""
import re
import sys

K = 8
pat = re.compile(r"Accepted:\s*(\d+)\s*tokens?,\s*Drafted:\s*(\d+)\s*tokens?")
a = d = n = 0
for path in sys.argv[1:]:
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                a += int(m.group(1))
                d += int(m.group(2))
                n += 1
print("windows=%d Accepted=%d Drafted=%d" % (n, a, d))
if d:
    r = a / d
    print("ratio=%.6f  pooled accept_len = 1 + %d*r = %.4f" % (r, K, 1 + K * r))
    print("=> mean(delta) predicted by `delta == rejected` is K*(1-r) = %.4f" % (K * (1 - r)))
