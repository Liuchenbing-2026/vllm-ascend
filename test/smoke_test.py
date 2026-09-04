import json, time, urllib.request
import os
URL = os.environ.get("GLM53_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("GLM53_MODEL", "glm53")
def post(p, payload, timeout=900):
    req = urllib.request.Request(URL + p, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.time() - t0, r
def counters():
    txt = urllib.request.urlopen(URL + "/metrics", timeout=60).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or "_bucket{" in line or "_created{" in line:
            continue
        for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                  "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                  "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k):
                out[k.replace("vllm:", "")] = float(line.rsplit(" ", 1)[1])
    return out

print("### 1. correctness")
for name, msg in [("math", "What is 17 * 23? Reply with only the number."),
                  ("zh",   "北京是哪个国家的首都？只回答国家名。")]:
    t, r = post("/v1/chat/completions", {"model": MODEL,
        "messages": [{"role": "user", "content": msg}], "max_tokens": 256, "temperature": 0})
    c = r["choices"][0]["message"]
    txt = (c.get("content") or "")
    print("  %-5s %6.2fs %3dtok  %s" % (name, t, r["usage"]["completion_tokens"], repr(txt[-160:])))

print()
print("### 2. prefix cache (same 5k-token prompt twice)")
b = counters()
prompt = "The quick brown fox jumps over the lazy dog. " * 500
t1, r1 = post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 8, "temperature": 0})
t2, r2 = post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 8, "temperature": 0})
a = counters()
dq = a.get("prefix_cache_queries_total",0) - b.get("prefix_cache_queries_total",0)
dh = a.get("prefix_cache_hits_total",0) - b.get("prefix_cache_hits_total",0)
print("  prompt_tokens=%d  cold=%.2fs  warm=%.2fs  speedup=%.1fx" % (r1["usage"]["prompt_tokens"], t1, t2, t1/t2 if t2 else 0))
print("  queries=%d hits=%d  hit_rate=%.1f%%   (hits/640 = %.2f blocks)" % (dq, dh, 100*dh/dq if dq else 0, dh/640.0))

print()
print("### 3. MTP acceptance (64-token generation)")
b = counters()
t, r = post("/v1/chat/completions", {"model": MODEL,
    "messages": [{"role": "user", "content": "Write a short poem about the sea."}],
    "max_tokens": 200, "temperature": 0})
a = counters()
d  = a.get("spec_decode_num_draft_tokens_total",0) - b.get("spec_decode_num_draft_tokens_total",0)
ac = a.get("spec_decode_num_accepted_tokens_total",0) - b.get("spec_decode_num_accepted_tokens_total",0)
nd = a.get("spec_decode_num_drafts_total",0) - b.get("spec_decode_num_drafts_total",0)
print("  gen %.2fs %d tokens" % (t, r["usage"]["completion_tokens"]))
print("  drafts=%d draft_tokens=%d accepted=%d  ACCEPTANCE=%.1f%%" % (nd, d, ac, 100*ac/d if d else 0))
print()
print("VERDICT: APC=%s  MTP=%s" % ("HIT" if dh>0 else "NO", "%.0f%%"%(100*ac/d) if d else "NO DRAFTS"))
