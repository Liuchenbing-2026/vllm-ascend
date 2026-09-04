"""gsm8k eval against a local vLLM /v1/chat/completions endpoint.

Chat mode because this model emits EOS immediately on ~75% of raw completion
prompts (measured: completion nonempty 2/8 even for the BASE model, chat 8/8).
Sequential (K=1) so scheduling and batch composition are identical across
arms -- with bit-exact ops the transcripts must then match byte for byte,
which is the accuracy gate.

usage: eval_gsm8k.py <arm-name> <out.json> [n] [max_tokens] [port] [model]
"""
import json
import re
import socket
import sys
import time
import urllib.request

socket.setdefaulttimeout(600)

ARM = sys.argv[1]
OUT = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 512
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 7519
MODEL = sys.argv[6] if len(sys.argv) > 6 else "openscad"
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
SUFFIX = "\nEnd your reply with '#### <number>'."

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract(text):
    m = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    cand = m[-1] if m else None
    if cand is None:
        allm = NUM.findall(text)
        cand = allm[-1] if allm else None
    if cand is None:
        return None
    cand = cand.replace(",", "").rstrip(".")
    try:
        f = float(cand)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def ask(question):
    body = {"model": MODEL, "max_tokens": MAXTOK, "temperature": 0, "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": question + SUFFIX}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    parts = []
    fin = None
    usage = {}
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if first is None:
                        first = time.perf_counter()
                    parts.append(delta)
                if ch.get("finish_reason"):
                    fin = ch["finish_reason"]
    e2e = time.perf_counter() - t0
    return ("".join(parts), fin, usage,
            None if first is None else (first - t0) * 1e3, e2e * 1e3)


def main():
    qs = [json.loads(l) for l in open("/work/evals/gsm8k_test.jsonl")][:N]
    rows = []
    ok = ne = 0
    t_start = time.perf_counter()
    for i, d in enumerate(qs):
        gold = extract(d["answer"])
        txt, fin, usage, ttft, e2e = ask(d["question"])
        got = extract(txt)
        correct = got is not None and got == gold
        ok += int(correct)
        ne += int(bool(txt.strip()))
        rows.append({"i": i, "gold": gold, "got": got, "correct": correct,
                     "finish": fin, "ttft_ms": ttft, "e2e_ms": e2e,
                     "ptok": usage.get("prompt_tokens"),
                     "ctok": usage.get("completion_tokens"), "text": txt})
        if (i + 1) % 8 == 0:
            print("  [%s] %d/%d acc=%d nonempty=%d elapsed=%.0fs"
                  % (ARM, i + 1, N, ok, ne, time.perf_counter() - t_start),
                  flush=True)
    wall = time.perf_counter() - t_start
    ctok = sum(r["ctok"] or 0 for r in rows)
    out = {"arm": ARM, "model": MODEL, "n": N, "max_tokens": MAXTOK,
           "protocol": "chat-seq-t0-seed42" + SUFFIX,
           "acc": ok / N, "nonempty": ne, "wall_s": round(wall, 1),
           "completion_tokens": ctok,
           "tok_per_s": round(ctok / wall, 2),
           "ttft_med_ms": round(sorted(r["ttft_ms"] for r in rows
                                       if r["ttft_ms"])[len(rows) // 2], 1),
           "rows": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    print("[%s] DONE acc=%.4f (%d/%d) nonempty=%d wall=%.0fs tok/s=%.2f -> %s"
          % (ARM, ok / N, ok, N, ne, wall, ctok / wall, OUT), flush=True)


if __name__ == "__main__":
    main()
