"""Compare two eval_gsm8k.py outputs: protocol equality, accuracy, and
byte-identity of every transcript."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for k in ("model", "n", "max_tokens", "protocol"):
    if a[k] != b[k]:
        print("PROTOCOL MISMATCH %s: %r vs %r" % (k, a[k], b[k]))
        sys.exit(2)
same = [ra["text"] == rb["text"] for ra, rb in zip(a["rows"], b["rows"])]
nid = sum(same)
print("%s acc=%.4f tok/s=%.2f wall=%.0fs | %s acc=%.4f tok/s=%.2f wall=%.0fs"
      % (a["arm"], a["acc"], a["tok_per_s"], a["wall_s"],
         b["arm"], b["acc"], b["tok_per_s"], b["wall_s"]))
print("transcripts identical: %d/%d" % (nid, a["n"]))
for i, s in enumerate(same):
    if not s:
        ra, rb = a["rows"][i], b["rows"][i]
        pa, pb = ra["text"], rb["text"]
        j = next((k for k in range(min(len(pa), len(pb))) if pa[k] != pb[k]),
                 min(len(pa), len(pb)))
        print("  q%d diverges at char %d: %r vs %r (correct %s/%s)"
              % (i, j, pa[max(0, j - 30):j + 30], pb[max(0, j - 30):j + 30],
                 ra["correct"], rb["correct"]))
