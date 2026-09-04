"""Prefill-side probe: TTFT at a long prompt, base vs adapter.

Prefill is where the branch's launcher bug bites hardest (the sgmv grid equals
the number of prefill tokens, so blockDim goes far past the 40 physical AIV
blocks), and prefill also runs fully eager -- max_cudagraph_capture_size is 8,
so any step with more than 8 tokens is not replayed from a graph and pays the
python wrapper per op call.  A short prompt hides both.

max_tokens=1 so the measurement is prefill only.
"""
import json
import statistics as st
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7519/v1/completions"
ARM = sys.argv[1] if len(sys.argv) > 1 else "unknown"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/work/evals/ttft_%s.json" % ARM
# max_model_len is 1024; leave room for the generated token
WORDS = ("the quick brown fox jumps over the lazy dog while counting numbers "
         "and reciting facts about photosynthesis chlorophyll and sunlight ")


def prompt_of(ntok_target, salt):
    reps = max(1, int(ntok_target / 24.2))  # measured 677 real tok at 28 reps
    return ("item %d. " % salt) + WORDS * reps


def one(prompt, model):
    body = {"model": model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0, "seed": 42}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        o = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except Exception as e:
        print("  REQUEST FAILED (%s): %s" % (model, e), flush=True)
        return None, None
    dt = (time.perf_counter() - t0) * 1e3
    return dt, (o.get("usage") or {}).get("prompt_tokens")


def main():
    out = []
    for target in (128, 448, 896):
        # warm the shape once per model so JIT/compile is not in the numbers
        for model in ("base", "openscad"):
            one(prompt_of(target, 999), model)
        for model in ("base", "openscad", "base", "openscad"):
            ds = []
            ptok = None
            for s in range(6):
                d, ptok = one(prompt_of(target, s), model)
                if d is not None:
                    ds.append(d)
            if not ds:
                continue
            row = {"target": target, "prompt_tokens": ptok, "model": model,
                   "ttft_ms_med": round(st.median(ds), 2),
                   "ttft_ms_min": round(min(ds), 2),
                   "all": [round(x, 2) for x in ds]}
            out.append(row)
            print("  ~%4d tok (%s real) %-9s  TTFT med %8.2f ms  min %8.2f  %s"
                  % (target, ptok, model, row["ttft_ms_med"], row["ttft_ms_min"],
                     row["all"]), flush=True)
        b = st.median([r["ttft_ms_med"] for r in out
                       if r["target"] == target and r["model"] == "base"])
        a = st.median([r["ttft_ms_med"] for r in out
                       if r["target"] == target and r["model"] == "openscad"])
        print("  ==> %4d tok: base %.2f ms | lora %.2f ms | LoRA adds %.2f ms (%.1f%% of prefill)"
              % (target, b, a, a - b, 100.0 * (a - b) / a), flush=True)
    json.dump({"arm": ARM, "rows": out}, open(OUT, "w"), indent=1)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
