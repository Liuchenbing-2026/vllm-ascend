"""How much of a decode step is LoRA?

With --enable-lora, a request routed to the BASE model name still goes through
PunicaWrapperNPU, but `compute_meta` returns no_lora=True and
_shrink_prefill/_expand_slice_prefill return immediately
(vllm_ascend/lora/punica_npu.py) -- and vLLM V1 always takes the prefill path
(vllm/v1/worker/lora_model_runner_mixin.py:57).  So "base" vs "openscad" on the
SAME server isolates the whole LoRA op cost, and "base" is identical across all
three arms, which makes it a control for run-to-run drift.
"""
import json
import statistics as st
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:7519/v1/completions"
ARM = sys.argv[1] if len(sys.argv) > 1 else "unknown"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/work/evals/share_%s.json" % ARM
MAXTOK = 96
PROMPTS = ["Describe in detail how the process of photosynthesis works in "
           "plants. Task number %d." % i for i in range(1, 9)]


def stream(prompt, model):
    body = {"model": model, "prompt": prompt, "max_tokens": MAXTOK,
            "temperature": 0, "seed": 42, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    last = t0
    n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            n += 1
    return ((first - t0) * 1e3,
            ((last - first) * 1e3 / (n - 1)) if n > 1 else None, n)


def run(model, k, npr):
    res = []
    idx = [0]
    lock = threading.Lock()
    ps = PROMPTS[:npr]

    def w():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(ps):
                return
            r = stream(ps[i], model)
            with lock:
                res.append(r)
    t0 = time.perf_counter()
    ths = [threading.Thread(target=w) for _ in range(k)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ntok = sum(r[2] for r in res)
    return {"model": model, "k": k, "n": npr, "wall_s": round(wall, 2),
            "tput": round(ntok / wall, 2),
            "ttft_med": round(st.median(r[0] for r in res), 2),
            "tpot_med": round(st.median(r[1] for r in res if r[1]), 3)}


def main():
    print("warmup", flush=True)
    for m in ("base", "openscad"):
        stream(PROMPTS[0], m)
    out = []
    for k, npr in ((1, 3), (2, 4), (4, 8)):
        for model in ("base", "openscad", "base"):
            r = run(model, k, npr)
            out.append(r)
            print("  K=%d %-9s tput %7.2f tok/s | ttft_med %8.1f | tpot_med %8.3f ms"
                  % (k, model, r["tput"], r["ttft_med"], r["tpot_med"]), flush=True)
        b = [x for x in out if x["k"] == k and x["model"] == "base"]
        a = [x for x in out if x["k"] == k and x["model"] == "openscad"][0]
        bmed = st.median(x["tpot_med"] for x in b)
        print("  ==> K=%d  base tpot %.3f ms (2 runs) | lora tpot %.3f ms | LoRA share of step = %.1f%% (+%.3f ms)"
              % (k, bmed, a["tpot_med"], 100.0 * (a["tpot_med"] - bmed) / a["tpot_med"],
                 a["tpot_med"] - bmed), flush=True)
    json.dump({"arm": ARM, "rows": out}, open(OUT, "w"), indent=1)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
