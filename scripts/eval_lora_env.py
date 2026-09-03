"""3-dataset LoRA A/B eval: TTFT/e2e + outputs via /v1/completions.

Checkpointed: writes <out>.part.json every 100 done (resume-safe).
Stall guards: global socket timeout 120s (connect+read); watchdog thread
exits the process (code 3) if no progress for 300s — the cron watchdog
then prunes the checkpoint, restarts serve, and resumes.
Usage: python3 eval_lora.py <gsm8k|humaneval|mmlu> <out.json> [max_tokens] [resume]
"""
import csv
import json
import os
import socket
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:7519/v1/completions"
K = int(os.environ.get("EVAL_K", "4"))
socket.setdefaulttimeout(120)  # connect AND read stall guard

lock = threading.Lock()
done_count = 0
out_path = ""
DATASET = ""
ITEMS = []
LAST_PROGRESS = [time.perf_counter()]


def watchdog():
    while True:
        time.sleep(60)
        idle = time.perf_counter() - LAST_PROGRESS[0]
        if idle > 300:
            print(f"STALL: no progress for {idle:.0f}s, exiting", flush=True)
            os._exit(3)


def load_gsm8k():
    out = []
    for line in open("/tmp/evals/gsm8k_test.jsonl"):
        d = json.loads(line)
        out.append({"q": f"Question: {d['question']}\nAnswer:", "ref": d["answer"]})
    return out


def load_humaneval():
    out = []
    for line in open("/tmp/evals/humaneval_test.jsonl"):
        d = json.loads(line)
        out.append({"q": d["prompt"], "ref": None})
    return out


def load_mmlu():
    subjects = ["abstract_algebra", "college_physics", "computer_security",
                "high_school_geography", "professional_psychology"]
    out = []
    for s in subjects:
        rows = list(csv.reader(open(f"/tmp/evals/mmlu/test/{s}_test.csv")))
        for r in rows[:100]:
            out.append({"q": (f"Question: {r[0]}\n"
                              f"A. {r[1]}\nB. {r[2]}\nC. {r[3]}\nD. {r[4]}\n"
                              f"Answer:"),
                        "ref": r[5]})
    return out


def complete(prompt, max_tokens):
    body = {"model": "openscad", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "seed": 42, "stream": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req)
    ttft = None
    last = None
    text = ""
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t0
        try:
            text += json.loads(payload)["choices"][0].get("text", "")
        except Exception:
            pass
        last = now
    return text, ttft, last - t0 if last else None


def save_partial(items, results, dataset, max_tokens):
    report = {"dataset": dataset, "n": len(items), "max_tokens": max_tokens,
              "wall_s": round(time.perf_counter() - T0, 1), "checkpoint": True,
              "requests": [{"id": i, "ref": items[i]["ref"], **results[i]}
                           if results[i] else None
                           for i in range(len(items))]}
    json.dump(report, open(out_path + ".part", "w"), indent=1, ensure_ascii=False)


def run_one(idx, max_tokens):
    try:
        text, ttft, e2e = complete(ITEMS[idx]["q"], max_tokens)
        return {"ok": True, "text": text,
                "ttft_ms": round(ttft * 1e3, 1) if ttft else None,
                "e2e_ms": round(e2e * 1e3, 1) if e2e else None}
    except Exception as e:
        return {"ok": False, "err": str(e)[:200]}


def worker(items, results, start, max_tokens):
    global done_count
    for i in range(start, len(items), K):
        with lock:
            if results[i] is not None:
                continue
        res = run_one(i, max_tokens)
        with lock:
            results[i] = res
            done_count += 1
            LAST_PROGRESS[0] = time.perf_counter()
            if done_count % 100 == 0:
                save_partial(items, results, DATASET, max_tokens)
            if done_count % 20 == 0 or done_count == len(items):
                print(f"  {done_count}/{len(items)}", flush=True)


def retry_worker(failed, results, start, max_tokens):
    for i in range(start, len(failed), K):
        idx = failed[i]
        results[idx] = run_one(idx, max_tokens)
        LAST_PROGRESS[0] = time.perf_counter()


def main():
    global out_path, DATASET, ITEMS, done_count, T0
    dataset, out_path = sys.argv[1], sys.argv[2]
    DATASET = dataset
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    resume = len(sys.argv) > 4 and sys.argv[4] == "resume"
    ITEMS = {"gsm8k": load_gsm8k, "humaneval": load_humaneval,
             "mmlu": load_mmlu}[dataset]()
    items = ITEMS
    results = [None] * len(items)
    T0 = time.perf_counter()
    if resume and os.path.exists(out_path + ".part"):
        part = json.load(open(out_path + ".part"))
        for r in part.get("requests", []):
            if r:
                results[r["id"]] = r
        n_done = sum(1 for r in results if r)
        print(f"resume: {n_done}/{len(items)} already done", flush=True)
    done_count = sum(1 for r in results if r)
    threading.Thread(target=watchdog, daemon=True).start()
    threads = [threading.Thread(target=worker, args=(items, results, k, max_tokens))
               for k in range(K)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [i for i in range(len(items))
              if results[i] is None or not results[i].get("ok")]
    if failed:
        print(f"retrying {len(failed)} failed...", flush=True)
        rt = [threading.Thread(target=retry_worker,
                               args=(failed, results, k, max_tokens))
              for k in range(K)]
        for t in rt:
            t.start()
        for t in rt:
            t.join()
    wall = time.perf_counter() - T0
    report = {"dataset": dataset, "n": len(items), "max_tokens": max_tokens,
              "wall_s": round(wall, 1),
              "requests": [{"id": i, "ref": items[i]["ref"], **results[i]}
                           for i in range(len(items))]}
    json.dump(report, open(out_path, "w"), indent=1, ensure_ascii=False)
    ok = sum(1 for r in results if r and r.get("ok"))
    print(f"DONE {dataset}: {ok}/{len(items)} ok, {wall:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()

