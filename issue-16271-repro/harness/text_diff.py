#!/usr/bin/env python3
"""Byte-compare the generated text of every arm against the first one.

Under temperature=0 the rejection sampler degenerates to an exact comparison, so
the speculative-sampling guarantee ("the output distribution equals the target's
p for any draft distribution q") predicts identical text no matter how wrong the
draft's KV length is. A difference here would mean the approximation escaped
``actual_seq_lengths_kv`` and reached something that changes the output, which
would invalidate treating it as an acceptance-only trade.

Also prints the per-arm repeat comparison, because a run that is not even
self-reproducible cannot say anything about the arms.
"""

import hashlib
import json
import os
import sys


def load(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return [r["text"] for r in d["results"]]


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("need at least two json files")
        return 1
    arms = {}
    for p in paths:
        if not os.path.exists(p):
            print("MISSING %s" % p)
            continue
        arms[os.path.basename(p)] = load(p)
    if not arms:
        return 1
    names = list(arms)
    ref_name = names[0]
    ref = arms[ref_name]
    print("reference: %s (%d completions, %d chars)" % (ref_name, len(ref), sum(len(t) for t in ref)))
    allsame = True
    for name in names:
        txts = arms[name]
        h = hashlib.sha256("\x00".join(txts).encode()).hexdigest()[:16]
        if len(txts) != len(ref):
            print("  %-28s sha=%s  LENGTH MISMATCH %d vs %d" % (name, h, len(txts), len(ref)))
            allsame = False
            continue
        diff = [i for i, (a, b) in enumerate(zip(ref, txts)) if a != b]
        status = "identical" if not diff else "DIFFERS at prompts %s" % diff[:6]
        if diff:
            allsame = False
        print("  %-28s sha=%s  %s" % (name, h, status))
    print("VERDICT: %s" % ("all arms byte-identical" if allsame else "ARMS DIFFER -- see above"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
