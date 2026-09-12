#!/usr/bin/env bash
set -euo pipefail

# Card-free. Before blaming hardware for acceptance 5.18 vs SGLang's 5.8, check the prompts.
#
# My "GSM8K" is NOT the GSM8K test set. It is 16 questions RECONSTRUCTED by me (20260911v) from
# gsm8k-exact8192, whose rows are one question repeated 96-178 times to fill a context. I cut each
# row at the FIRST '?' to recover a single question.
#
# That reconstruction has a specific failure mode I never checked: a GSM8K item whose body
# contains an earlier question mark gets TRUNCATED mid-problem, leaving a prompt that asks
# something incomplete or nonsensical. A model answering a malformed question produces less
# predictable text, and less predictable text is exactly what lowers draft acceptance. So a bug
# here would look like "our drafter is worse" while actually being "our prompts are broken".
#
# Two things to establish:
#   (1) For each of the 16, how much of the original question did the cut throw away?
#       Zero discarded tail = clean. A discarded tail containing sentences = truncated prompt.
#   (2) Is the real GSM8K test set anywhere on this box? If it is, the fix is to use it: 1319
#       genuine questions instead of 16 reconstructions, which also removes the oversampling
#       (96 requests drawn from 16 distinct prompts) that no published run has.

root=/data2/dflash2-v026-official
exact="$root/aisbench_io_sweep_20260824/datasets/gsm8k-exact8192/test.jsonl"
short="$root/datasets/gsm8k-short/prompts.jsonl"
evidence="$root/upstream-audit/gsm8k-fidelity-20260912g"

trap 'status=$?; printf "FIDELITY_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -f "$exact"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (1) how faithful is each reconstructed prompt =====\n'
python3 - "$exact" "$short" "$evidence" <<'FID'
import json, sys
from pathlib import Path
exact, short, ev = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

rows = [json.loads(l) for l in exact.read_text(encoding="utf-8").splitlines() if l.strip()]
# Recover the single repeated unit the padded row was built from, then compare the first-'?' cut
# against that unit -- not against the whole padded row, which is 96-178 copies.
def unit_of(q):
    # The row is one question repeated; find the shortest prefix that tiles the string.
    for n in range(20, len(q) // 2 + 1):
        p = q[:n]
        if q.startswith(p * 2):
            return p
    return q

bad = 0
for i, r in enumerate(rows):
    q = r["question"]
    u = unit_of(q).strip()
    cut = u[: u.find("?") + 1] if "?" in u else u[:400]
    tail = u[len(cut):].strip()
    status = "clean" if not tail else "TRUNCATED"
    if tail:
        bad += 1
    print("\n[%02d] %s  unit_chars=%d cut_chars=%d discarded=%d" % (i, status, len(u), len(cut), len(tail)))
    print("   cut : %s" % cut[:200])
    if tail:
        print("   LOST: %s" % tail[:200])
print("\nTRUNCATED_PROMPTS=%d of %d" % (bad, len(rows)))
FID

printf '\n===== (2) do the served prompts end like real questions =====\n'
python3 - "$short" <<'ENDS'
import json, sys
from pathlib import Path
lines = [json.loads(l)["prompt"] for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
print("served prompts: %d" % len(lines))
for i, p in enumerate(lines):
    print("[%02d] ends_with_qmark=%-5s chars=%-4d tail=...%s" % (i, p.rstrip().endswith("?"), len(p), p[-70:].replace("\n", " ")))
ENDS

printf '\n===== (3) is the real GSM8K test set on this box =====\n'
for d in /data1 /data2 /root /home /opt; do
  [[ -d "$d" ]] || continue
  find "$d" -maxdepth 6 -iname '*gsm8k*' \( -type d -o -name '*.jsonl' -o -name '*.json' -o -name '*.parquet' \) \
    -printf '%y %10s  %p\n' 2>/dev/null
done | sort -u > "$evidence/gsm8k-candidates.txt" || true
printf 'candidates found: %s\n' "$(wc -l < "$evidence/gsm8k-candidates.txt")"
head -n 40 "$evidence/gsm8k-candidates.txt"

printf '\n===== (4) which of those is the genuine 1319-row test split =====\n'
while read -r kind size path; do
  case "$path" in *.jsonl) ;; *) continue ;; esac
  n="$(wc -l < "$path" 2>/dev/null || echo 0)"
  # The real GSM8K test split is 1319 rows; the padded ones here are 16.
  first="$(head -c 200 "$path" 2>/dev/null | tr '\n' ' ')"
  printf '%-6s rows=%-6s %s\n    %s\n' "$kind" "$n" "$path" "$first"
done < "$evidence/gsm8k-candidates.txt" | head -n 40

echo "EVIDENCE=$evidence"
echo GSM8K_FIDELITY_PASS
