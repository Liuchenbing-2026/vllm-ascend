#!/usr/bin/env bash
set -euo pipefail

# Card-free. At high concurrency the per-iteration cost is dominated by verify width,
# batch x (K+1), so the quantity that decides the winner is accepted tokens per verify slot,
# advance/(K+1). DFlash's per-position acceptance decays steeply, which means its deep positions
# buy almost nothing while costing a full slot each -- invisible at C1, decisive at C32.
#
# Rather than blind-running K=1,2,3 at C32 (four extra service starts, ~35 min), read the
# per-position acceptance that the C32 runs ALREADY recorded and compute the optimum directly.
# Only the winning K then needs a real run.
#
# The per-position numbers live in the bench logs of the runs already on disk; the summary step
# only ever printed five grep'd lines, so they were never surfaced.

root=/data2/dflash2-v026-official
evidence="$root/upstream-audit/optimal-k-20260911o"

trap 'status=$?; printf "OPTK_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== per-position acceptance already on disk =====\n'
for d in "$root"/upstream-audit/benchserve-c32gsm-k7-20260911 \
         "$root"/upstream-audit/benchserve-c32warm-k7-20260910 \
         "$root"/upstream-audit/benchserve-c8t0b-k7-20260910 \
         "$root"/upstream-audit/benchserve-c32gsm-mtp3-20260911 \
         "$root"/upstream-audit/benchserve-c32warm-mtp3-20260910; do
  [[ -d "$d" ]] || continue
  for log in "$d"/bench.default.log "$d"/bench.default2.log; do
    [[ -f "$log" ]] || continue
    printf -- '--- %s / %s ---\n' "$(basename "$d")" "$(basename "$log")"
    sed -n '/Per-position acceptance/,/^=\{10,\}/p' "$log" | head -n 12
  done
done | tee "$evidence/positions.txt"

printf '\n===== optimal K under a verify-width cost model =====\n'
python3 - "$evidence/positions.txt" <<'PY'
import re, sys
from pathlib import Path

blocks, cur, name = {}, [], None
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.match(r"--- (\S+) / (\S+) ---", line)
    if m:
        if name and cur:
            blocks[name] = cur
        name, cur = "%s|%s" % (m.group(1), m.group(2)), []
        continue
    m = re.match(r"\s*Position (\d+):\s+([\d.]+)", line)
    if m and name:
        cur.append(float(m.group(2)) / 100.0)
if name and cur:
    blocks[name] = cur

if not blocks:
    print("no per-position blocks found")
    raise SystemExit(0)

# MTP3 reference from the same concurrency, if present in this dump.
mtp = None
for k, v in blocks.items():
    if "mtp3" in k and len(v) == 3:
        mtp = v
        break

for k, pos in sorted(blocks.items()):
    depth = len(pos)
    print("\n%s   depth=%d" % (k, depth))
    print("  positions: " + " ".join("%.4f" % p for p in pos))
    best = None
    for K in range(1, depth + 1):
        adv = 1.0 + sum(pos[:K])
        per_slot = adv / (K + 1)
        star = ""
        if best is None or per_slot > best[1]:
            best = (K, per_slot, adv)
        print("  K=%d  advance=%.4f  width=%d  advance/width=%.4f%s" % (K, adv, K + 1, per_slot, star))
    print("  -> best K=%d  advance=%.4f  advance/width=%.4f" % (best[0], best[2], best[1]))
    if mtp is not None and "mtp3" not in k:
        adv_m = 1.0 + sum(mtp)
        ps_m = adv_m / 4.0
        print("  -> MTP3 same-dump: advance=%.4f advance/width=%.4f  ratio(best/mtp3)=%.3f"
              % (adv_m, ps_m, best[1] / ps_m))
PY

echo "EVIDENCE=$evidence"
echo OPTIMAL_K_PASS
