import json, time, urllib.request, re
import os
URL = os.environ.get("GLM53_URL", "http://127.0.0.1:8000") + "/v1"
MODEL = os.environ.get("GLM53_MODEL", "glm53")

def post(path, payload, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.time() - t0, r

def metrics():
    txt = urllib.request.urlopen(URL.rsplit("/v1",1)[0] + "/metrics", timeout=60).read().decode()
    out = {}
    for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
              "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total"):
        for line in txt.splitlines():
            if line.startswith(k) and not line.startswith("#"):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out

# ~4000 token prompt: comfortably more than one 2176-token block at TP2
prompt = ("The quick brown fox jumps over the lazy dog. " * 500)

print("metrics before:", metrics())
t1, r1 = post("/completions", {"model": MODEL, "prompt": prompt,
                               "max_tokens": 4, "temperature": 0})
print("run1: %.3fs  prompt_tokens=%d" % (t1, r1["usage"]["prompt_tokens"]))
m1 = metrics(); print("metrics after run1:", m1)

t2, r2 = post("/completions", {"model": MODEL, "prompt": prompt,
                               "max_tokens": 4, "temperature": 0})
print("run2: %.3fs  prompt_tokens=%d" % (t2, r2["usage"]["prompt_tokens"]))
m2 = metrics(); print("metrics after run2:", m2)

q = m2.get("vllm:prefix_cache_queries_total", m2.get("vllm:gpu_prefix_cache_queries_total", 0))
h = m2.get("vllm:prefix_cache_hits_total", m2.get("vllm:gpu_prefix_cache_hits_total", 0))
print()
print("PREFIX CACHE: queries=%d hits=%d  hit_rate=%.1f%%" % (q, h, 100.0 * h / q if q else 0))
print("LATENCY: run1=%.3fs run2=%.3fs  speedup=%.2fx" % (t1, t2, t1 / t2 if t2 else 0))
print("VERDICT:", "APC HIT" if h > 0 else "NO HIT")
