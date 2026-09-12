#!/usr/bin/env bash
set -euo pipefail

arm_id=__ARM__

# PHASE B: the concurrency sweep that reproduces SGLang's published kou-jing on my stack.
#
# sgl-project/sglang#35629 (Ascend 910C, TP2, GSM8K, greedy, RadixCache off) reports:
#     C1   33.2 -> 101.2 tok/s  3.05x   accept 5.8
#     C8  176.2 -> 329.4        1.87x   accept 5.8
#     C16 247.1 -> 431.2        1.75x   accept 5.8
# My C32 on gsm8k-exact4096 gives DFlash 149.87 vs nospec 184.25 = 0.81x -- slower than no
# speculation at all. Four axes separate those: input length (4096 vs ~100 tokens), concurrency
# (32 vs 16), sampling (default vs greedy), and, for the MTP comparison only, MTP depth (3 vs 7).
# This run pins sampling to greedy and input length to short, then sweeps concurrency, so the
# published curve and mine are read in the same units.
#
# The whole sweep runs inside ONE service instance. That is the point: C1 through C32 then share
# an instance, so the shape of the curve carries no restart variance ([[ab-session-is-the-variance-unit]]),
# and only the warm-up artifact has to be handled -- which the discarded warmup round does
# ([[vllm-bench-first-run-is-warmup]]).
#
# --ignore-eos with a fixed 256-token output is KEPT, deliberately, even though SGLang let GSM8K
# stop at EOS. Two reasons: it makes the C32 cell a single-variable pair against the existing
# gsm8k-exact4096 C32 numbers (only token content changed), and forcing generation past the end
# of the answer produces degenerate repeated text, which is the regime where DFlash scores
# HIGHEST (0.9955 on the repeated prompts, 20260911bk). So this choice is conservative with
# respect to the claim being tested: it can only flatter DFlash.

devices=0,2
serve_max_seqs=32
out_len=2048
root=/data2/dflash2-v026-official
# Built by 20260911v: the 16 distinct GSM8K questions recovered from gsm8k-exact8192 by cutting
# at the first '?', i.e. the genuine short prompts the padded 4096-token rows were made from.
# CONTAINER path -- the bench runs through docker exec and the container mounts $root at /dflash-out.
dataset_path=/dflash-out/datasets/gsm8k-short/prompts.jsonl
evidence="$root/upstream-audit/benchserve-sgl-${arm_id}-20260912a"
model=/data1/dflash2-models/Qwen3.8-27B
container="benchserve-${arm_id}-sgl-20260912a"

case "$arm_id" in
  nospec) needle='vllm serve' ;;
  mtp3)   needle='qwen3_5_mtp' ;;
  mtp7)   needle='"num_speculative_tokens":7' ;;
  k3)     needle='"num_speculative_tokens":3' ;;
  k7)     needle='dflash_full_kv_allocation' ;;
  *) echo "unknown arm_id=$arm_id" >&2; exit 2 ;;
esac

cleanup_own_service() {
  if docker ps -aq --filter name="^/${container}$" | grep -q .; then
    docker logs "$container" > "$evidence/service.$1.log" 2>&1 || true
    timeout -k 10s 60s docker kill "$container" >/dev/null 2>&1 || true
    timeout -k 10s 60s docker rm -f "$container" >/dev/null 2>&1 || true
  fi
}
on_error() {
  status=$?
  printf 'BENCHSGL_ERROR arm=%s line=%s status=%s command=%q\n' "$arm_id" "$LINENO" "$status" "$BASH_COMMAND" >&2
  cleanup_own_service failed
  exit "$status"
}
trap on_error ERR

test "$(hostname)" = kylin10-6018
test -d "$evidence"
backend="$(awk -F'backend=| endpoint=' '{print $2}' "$evidence/backend.txt")"
endpoint="$(awk -F'endpoint=' '{print $2}' "$evidence/backend.txt")"
test -n "$backend"
test -n "$endpoint"

fatal='Traceback \(most recent call last\)|Engine core initialization failed|WorkerProc hit an exception|OutOfMemoryError|AIVEC|507011|507033|507034|107025|561002'
ready=0
for attempt in $(seq 1 30); do
  state="$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || true)"
  docker logs "$container" > "$evidence/readiness.log" 2>&1 || true
  health="$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)"
  echo "attempt=$attempt state=$state health=$health time=$(date +%H:%M:%S)"
  if grep -Eq "$fatal" "$evidence/readiness.log"; then tail -n 200 "$evidence/readiness.log" > "$evidence/readiness.tail.txt"; exit 43; fi
  if [[ "$state" != running ]]; then tail -n 200 "$evidence/readiness.log" > "$evidence/readiness.tail.txt"; exit 44; fi
  if [[ "$health" = 200 ]]; then ready=1; break; fi
  sleep 10
done
if (( ready != 1 )); then
  # Do NOT tear down: the capture ladder here is 6-7 sizes at ~50 s each, so a not-ready service
  # is usually still capturing. Re-publishing this task is cheaper than paying startup again.
  trap - ERR
  echo "BENCHSGL_NOT_READY arm=$arm_id (service left running; re-publish the bench task)"
  exit 46
fi

for attempt in $(seq 1 40); do
  docker exec "$container" ps -ww -eo args > "$evidence/service.args.txt" 2>/dev/null || true
  if grep -Fq 'vllm serve' "$evidence/service.args.txt" && ! grep -Fq 'bin/bash -lc' "$evidence/service.args.txt"; then break; fi
  sleep 2
done
grep -Fq "$needle" "$evidence/service.args.txt"
grep -Fq -- "--max-num-seqs $serve_max_seqs" "$evidence/service.args.txt"

docker exec "$container" bash -lc "test -f $dataset_path && wc -l $dataset_path && head -c 120 $dataset_path" \
  | tee "$evidence/dataset-visible.txt"
grep -q '"prompt"' "$evidence/dataset-visible.txt"

# How long are these prompts really, under this model's tokenizer? The host python has no
# transformers, so ask the container. This is the variable the whole experiment turns on.
docker exec "$container" bash -lc "cd /tmp && python3 -c \"
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$model', trust_remote_code=True)
lens = [len(tok(json.loads(l)['prompt'])['input_ids']) for l in open('$dataset_path', encoding='utf-8') if l.strip()]
lens.sort()
print('prompt_tokens n=%d min=%d median=%d max=%d mean=%.1f' % (len(lens), lens[0], lens[len(lens)//2], lens[-1], sum(lens)/len(lens)))
\"" > "$evidence/prompt-lens.txt" 2>&1 || true
cat "$evidence/prompt-lens.txt"

run_one() {
  tag="$1"; prompts="$2"; conc="$3"; eos="${4---ignore-eos}"
  echo "===== bench tag=$tag prompts=$prompts concurrency=$conc eos='$eos' ====="
  curl -fsS --max-time 5 http://127.0.0.1:8000/metrics -o "$evidence/metrics.$tag.before.prom"
  t0="$(date +%s.%N)"
  docker exec "$container" bash -lc "cd /tmp && vllm bench serve \
    --backend $backend \
    --model $model \
    --base-url http://127.0.0.1:8000 \
    --endpoint $endpoint \
    --num-prompts $prompts \
    --trust-remote-code \
    --dataset-name custom \
    --dataset-path $dataset_path \
    --custom-output-len $out_len \
    $eos \
    --temperature 0 \
    --seed 12345 \
    --max-concurrency $conc" > "$evidence/bench.$tag.log" 2>&1
  t1="$(date +%s.%N)"
  curl -fsS --max-time 5 http://127.0.0.1:8000/metrics -o "$evidence/metrics.$tag.after.prom"
  awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}' > "$evidence/bench.$tag.wall.txt"
  grep -E 'Output token throughput|Mean TTFT|Mean TPOT|Acceptance rate|Acceptance length' "$evidence/bench.$tag.log" || true
}

# Order: warmup (discarded), then C32 FIRST, the ascending sweep, then C32 again. The first
# attempt ran c32/c32b back to back at the end and failed its own order control (drift 0.9262,
# TPOT 57.36 -> 62.68 ms). Two fixes here:
#   - 96 prompts at C32 instead of 64. At 64 the run is two 32-wide waves, so the trailing wave's
#     queueing dominates "output token throughput" and the metric is mostly wave structure.
#   - c32 at both ends, so the pair brackets the ENTIRE sweep rather than measuring the last two
#     minutes of it. If the bracket closes, every interior point inherits that guarantee.
# LONG-OUTPUT variant. Every cell respects EOS and allows 2048 tokens instead of 256.
#
# This is the last kou-jing axis still misaligned with the published runs. LMSYS's setup line is
# "greedy decoding, THINKING ENABLED, max new tokens 4096"; mine capped output at 256. The target
# does have a thinking template (20260911ap: chat_template carries enable_thinking / <think> /
# </think>), every sweep here already goes through /v1/chat/completions, and the EOS cells came
# back at mean_out 191-200 against a 256 cap -- i.e. the answers were being TRUNCATED, not
# finishing. So the 256 runs measured a chopped-off prefix of the real workload.
#
# Why this axis can move the verdict rather than just the numbers: long structured
# chain-of-thought is the best case for a drafter (repetitive, highly predictable), and the
# per-request fixed costs -- prefill, the scheduler slot churn that cost DFlash 40% of its
# decoding slots at C32 -- amortise over ~8x more generated tokens. If DFlash has a regime where
# it wins at useful concurrency, this is where it lives.
# The "" is the stop flag: EOS respected. run_one defaults to --ignore-eos, so leaving it off
# here would have silently measured forced 2048-token continuations -- the exact kou-jing this
# variant exists to get away from.
run_one warmup  4  8 ""
run_one c8     16  8 ""
run_one c1      4  1 ""
run_one c16    32 16 ""
run_one c8b    16  8 ""

# No separate *noeos cells here: every cell above already passes "" for the stop flag, so the
# whole sweep respects EOS. Carrying the old pair over would have re-measured the 256-token
# kou-jing under a 2048 budget and quietly mixed two workloads in one table.

python3 - "$evidence" "$arm_id" <<'SUMPY'
import json, re, sys
from pathlib import Path
ev, arm = Path(sys.argv[1]), sys.argv[2]

def parse(path):
    tot = {"accepted": 0.0, "drafted": 0.0, "drafts": 0.0}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        base = name.split("{", 1)[0]
        try:
            v = float(value)
        except ValueError:
            continue
        if base == "vllm:spec_decode_num_accepted_tokens_total":
            tot["accepted"] += v
        elif base == "vllm:spec_decode_num_draft_tokens_total":
            tot["drafted"] += v
        elif base == "vllm:spec_decode_num_drafts_total":
            tot["drafts"] += v
    return tot

tags = ("warmup", "c8", "c1", "c16", "c8b")
out = {"arm": arm}
for tag in tags:
    b, a = parse(ev / ("metrics.%s.before.prom" % tag)), parse(ev / ("metrics.%s.after.prom" % tag))
    delta = {k: a[k] - b[k] for k in b}
    text = (ev / ("bench.%s.log" % tag)).read_text(encoding="utf-8", errors="replace")
    def grab(label):
        m = re.search(re.escape(label) + r"[^\d\-]*(-?[\d.]+)", text)
        return float(m.group(1)) if m else None
    rec = {
        "output_throughput_tok_s": grab("Output token throughput"),
        "mean_ttft_ms": grab("Mean TTFT"),
        "mean_tpot_ms": grab("Mean TPOT"),
        "acceptance_length_reported": grab("Acceptance length"),
        "acceptance_rate_pct_reported": grab("Acceptance rate"),
        "bench_wall_s": round(float((ev / ("bench.%s.wall.txt" % tag)).read_text(encoding="utf-8").strip()), 3),
    }
    # advance = 1 + accepted/drafts is the cross-depth comparable quantity; acceptance_rate is
    # not, because it averages over a different number of positions at each K.
    rec["advance"] = round(1 + delta["accepted"] / delta["drafts"], 4) if delta["drafts"] else 1.0
    if rec["mean_tpot_ms"]:
        rec["t_iter_ms"] = round(rec["mean_tpot_ms"] * rec["advance"], 2)
    out[tag] = rec

# C8 brackets this sweep (first measured cell and last), the way C32 did in the 256-token one.
first, last = out["c8"], out["c8b"]
if first["output_throughput_tok_s"] and last["output_throughput_tok_s"]:
    drift = last["output_throughput_tok_s"] / first["output_throughput_tok_s"]
    out["sweep_drift_c8"] = round(drift, 4)
    out["order_control_ok"] = bool(abs(drift - 1.0) < 0.05)

# How long did the answers actually get? That IS the workload here: if thinking is on, these
# should land well above the 191-200 tokens the 256-capped runs were truncated to.
for tag in tags:
    log = (ev / ("bench.%s.log" % tag)).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Total generated tokens[^\d]*(\d+)", log)
    n = re.search(r"Successful requests[^\d]*(\d+)", log)
    if m and n and int(n.group(1)):
        out[tag]["mean_output_tokens"] = round(int(m.group(1)) / int(n.group(1)), 1)

(ev / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print("SGL_SWEEP_RESULT " + json.dumps(out, sort_keys=True))
print("\n--- %s: greedy, short GSM8K prompts, EOS respected, up to 2048 out ---" % arm)
print("%-8s %12s %10s %10s %10s %12s" % ("conc", "tok/s", "advance", "TPOT ms", "T_iter ms", "mean_out"))
for tag, conc in (("c1", "1"), ("c8", "8"), ("c16", "16"), ("c8b", "8(rep)")):
    r = out[tag]
    print("%-8s %12s %10s %10s %10s %12s" % (
        conc, r["output_throughput_tok_s"], r["advance"], r["mean_tpot_ms"],
        r.get("t_iter_ms"), r.get("mean_output_tokens")))
print("order_control_ok=%s drift=%s" % (out.get("order_control_ok"), out.get("sweep_drift_c8")))
SUMPY

cleanup_own_service completed
# The shared-machine rule: the cards go back the moment the measurement is done.
test -z "$(ss -ltnH 'sport = :8000' || true)"
for d in 0 2; do
  hbm="$(npu-smi info -t usages -i "$d" -c 0 | awk -F: '/HBM Usage Rate/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
  printf 'released device=%s hbm_usage_rate=%s\n' "$d" "$hbm"
done
echo "EVIDENCE=$evidence"
echo "BENCHSGL_PASS arm=$arm_id"
