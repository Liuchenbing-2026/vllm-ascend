#!/usr/bin/env bash
set -euo pipefail

# Card-free prep for the SGLang-kou-jing reproduction.
#
# SGLang's published NPU numbers (sgl-project/sglang#35629, 910C TP2, GSM8K, greedy, no RadixCache):
#   C1  33.2 -> 101.2 tok/s (3.05x, accept 5.8)
#   C8  176.2 -> 329.4     (1.87x)
#   C16 247.1 -> 431.2     (1.75x)
# My C32 run on gsm8k-exact4096 gives DFlash 149.87 against nospec 184.25, i.e. 0.81x -- BELOW
# no-spec. Between those two sits: 4096-token inputs vs ~100-token GSM8K questions, C32 vs C16,
# non-greedy vs greedy. This builds the dataset that lets me change input length alone.
#
# The existing gsm8k-exact4096 rows are ONE question repeated to fill 4096 tokens (20260911bk),
# so cutting at the first '?' recovers the genuine short question -- the same construction that
# novel1..4 used at C1. Here I take ALL 16 distinct base questions.

root=/data2/dflash2-v026-official
exact="$root/aisbench_io_sweep_20260824/datasets/gsm8k-exact8192/test.jsonl"
outdir="$root/datasets/gsm8k-short"
evidence="$root/upstream-audit/gsm8kshort-prep-20260911v"

trap 'status=$?; printf "PREP_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -f "$exact"
test ! -e "$evidence"
mkdir -p "$evidence" "$outdir"

printf '===== who is on the cards right now =====\n'
npu-smi info | head -n 24
printf -- '--- per-device HBM usage rate ---\n'
for d in 0 1 2 3 4 5 6 7; do
  hbm="$(npu-smi info -t usages -i "$d" -c 0 2>/dev/null | awk -F: '/HBM Usage Rate/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' || true)"
  printf 'device=%s hbm_usage_rate=%s\n' "$d" "${hbm:-?}"
done
printf -- '--- containers on this box ---\n'
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | head -n 20
printf -- '--- my leftovers (must be none) ---\n'
docker ps -a --format '{{.Names}}' | grep -E '^(fourarm|onearm|csweep|mbtc8|c8delta|benchserve)-[a-z0-9_-]+-2026091[01]' || echo '(none)'
printf -- '--- port 8000 ---\n'
ss -ltnH 'sport = :8000' || echo '(free)'

printf '\n===== build the short GSM8K prompt set =====\n'
python3 - "$exact" "$outdir" "$evidence" <<'BUILDPY'
import json, sys
from pathlib import Path
exact, out, ev = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(l) for l in exact.read_text(encoding="utf-8").splitlines() if l.strip()]
seen, prompts = set(), []
for r in rows:
    q = r["question"]
    unit = q[: q.find("?") + 1] if "?" in q else q[:400]
    unit = unit.strip()
    if unit and unit not in seen:
        seen.add(unit)
        prompts.append(unit)
dst = out / "prompts.jsonl"
with dst.open("w", encoding="utf-8") as fh:
    for p in prompts:
        fh.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
print("distinct short questions: %d" % len(prompts))
print("char lengths: " + " ".join(str(len(p)) for p in prompts))
print("\nfirst two:")
for p in prompts[:2]:
    print("  " + p[:160])
(ev / "prompts.head.txt").write_text("\n".join(prompts[:4]), encoding="utf-8")
BUILDPY

printf '\n===== token lengths under the real tokenizer =====\n'
python3 - "$outdir" <<'TOKPY'
import json, sys
from pathlib import Path
try:
    from transformers import AutoTokenizer
except Exception as exc:
    print("transformers unavailable on host python (%s); lengths will be checked in-container" % exc)
    raise SystemExit(0)
tok = AutoTokenizer.from_pretrained("/data1/dflash2-models/Qwen3.8-27B", trust_remote_code=True)
lens = []
for line in Path(sys.argv[1], "prompts.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        lens.append(len(tok(json.loads(line)["prompt"])["input_ids"]))
lens.sort()
print("n=%d  min=%d  median=%d  max=%d  mean=%.1f" % (len(lens), lens[0], lens[len(lens)//2], lens[-1], sum(lens)/len(lens)))
TOKPY

ls -la "$outdir"
echo "EVIDENCE=$evidence"
echo GSM8KSHORT_PREP_PASS
