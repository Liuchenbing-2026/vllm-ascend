"""Sanity-check a msmodelslim GLM-5.3-Flash w8a8 artifact.

    python3 inspect_artifact.py /data02/GLM-5.3-Flash-w8a8-b0829

Checks that matter:
  * FLOAT list must contain all 347 model.visual.* tensors  -- if it is empty you built with
    branch glm5_next_quant_0830, which drops the ViT. That artifact is unusable.
  * mlp.gate.weight / e_score_correction_bias must be FLOAT  -- msmodelslim warns that the
    exclude pattern '*gate' matched nothing; that warning is benign, the gate is not a
    quantizable Linear and never enters the candidate set.
  * shared_experts tells the two variants apart: W8A8_DYNAMIC = b0829, FLOAT = b0829se.
"""
import json, collections, re, sys
p = sys.argv[1]
d = json.load(open(p + "/quant_model_description.json"))
def norm(k):
    k = re.sub(r"layers\.\d+\.", "layers.N.", k)
    k = re.sub(r"experts\.\d+\.", "experts.E.", k)
    k = re.sub(r"\.\d+\.", ".N.", k)
    return k
pat = collections.defaultdict(collections.Counter)
for k, v in d.items():
    if not isinstance(v, str) or v not in ("W8A8_DYNAMIC", "FLOAT"):
        continue
    pat[norm(k)][v] += 1
print("=== FLOAT (kept bf16) patterns ===")
for k, c in sorted(pat.items()):
    if c["FLOAT"]:
        print("  %6d  %s" % (c["FLOAT"], k))
print()
print("=== W8A8_DYNAMIC patterns ===")
for k, c in sorted(pat.items(), key=lambda x: -x[1]["W8A8_DYNAMIC"]):
    if c["W8A8_DYNAMIC"]:
        print("  %6d  %s" % (c["W8A8_DYNAMIC"], k))
print()
print("=== gate / shared / indexer / visual / lm_head spot check ===")
for probe in ["gate", "shared_expert", "index", "visual", "lm_head", "embed_tokens", "nextn", "mtp", "eh_proj", "mhc"]:
    hits = [(k, v) for k, v in d.items() if probe in k and isinstance(v, str)]
    kinds = collections.Counter(v for _, v in hits)
    print("  %-14s n=%-7d %s" % (probe, len(hits), dict(kinds)))
    for k, v in hits[:3]:
        print("         e.g. %s -> %s" % (k, v))
