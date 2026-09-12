#!/usr/bin/env bash
set -euo pipefail

# Card-free. Where do the DFlash drafter's ~38 ms per iteration go?
#
# The measurement that raises the question (C32 gsm8k, relative to a no-spec iteration = 1.000):
#     nospec  width 1  cost 1.000
#     K=1     width 2  cost 3.383     <- adding the drafter alone costs +2.38 target forwards
#     MTP3    width 4  cost 2.788
#     K=7     width 8  cost 4.328     <- width 2 -> 8 only costs +28%
# So depth is nearly free and the drafter's fixed cost dominates. At C1 the same split gives
# c0 = 33.85 ms and H_k7 = 38.28 ms, i.e. the 5-layer drafter costs about as much as one full
# 62-layer target forward. On layer count alone it should cost ~8% of that: off by 15-30x.
#
# Leading hypothesis: the drafter is NOT inside the captured ACL graph and pays host dispatch per
# operator every iteration. On this box host dispatch is already known to dominate (DSV4: 28 ms of
# compute behind 345 ms of dispatch). That would explain an order of magnitude with no FLOP change.
#
# This audit is read-only: capture evidence from logs already on disk, plus the proposer source.
# No profiler -- msprof op under-reports 10-15% here and distorts A/B ratios.

root=/data2/dflash2-v026-official
runtime="$root/runtime-source/dflash2-mrv1-full-kv-device-queryloc-v2-runtime-e6519803"
evidence="$root/upstream-audit/drafter-cost-audit-20260911t"

trap 'status=$?; printf "AUDIT_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -d "$runtime"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== what the service logs say about graph capture =====\n'
# Write to a file first and head the FILE: heading a pipe kills grep with SIGPIPE, which
# pipefail turns into a fatal error (that is how 20260910bj died at exit 141).
for d in "$root"/upstream-audit/benchserve-c32gsm-k7-20260911 \
         "$root"/upstream-audit/benchserve-c32gsm-k1-20260911 \
         "$root"/upstream-audit/benchserve-c32gsm-mtp3-20260911; do
  [[ -f "$d/readiness.log" ]] || continue
  printf -- '--- %s ---\n' "$(basename "$d")"
  grep -iE 'capturing|captured|replay|cudagraph|graph capture|aclgraph|Skipping' "$d/readiness.log" > "$evidence/cap.tmp" 2>/dev/null || true
  head -n 14 "$evidence/cap.tmp"
  printf '  (capture-related lines: %s)\n' "$(wc -l < "$evidence/cap.tmp")"
done | tee "$evidence/capture-lines.txt"
rm -f "$evidence/cap.tmp"

printf '\n===== does the proposer wrap its forward in a graph at all =====\n'
grep -rn --include=*.py -e 'cudagraph' -e 'npugraph' -e 'graph_pool' -e 'capture' \
  "$runtime/vllm_ascend/spec_decode/" > "$evidence/proposer-graph.txt" 2>/dev/null || true
head -n 40 "$evidence/proposer-graph.txt"
printf '  (total matches: %s)\n' "$(wc -l < "$evidence/proposer-graph.txt")"

printf '\n===== the dflash propose() path =====\n'
ls "$runtime/vllm_ascend/spec_decode/" > "$evidence/spec-files.txt"
cat "$evidence/spec-files.txt"
for f in "$runtime/vllm_ascend/spec_decode/"*.py; do
  n="$(basename "$f")"
  case "$n" in
    *dflash*|*llm_base_proposer*) ;;
    *) continue ;;
  esac
  printf -- '--- %s: def lines ---\n' "$n"
  grep -nE '^\s*(def |class )' "$f" > "$evidence/defs.$n.txt" 2>/dev/null || true
  head -n 60 "$evidence/defs.$n.txt"
done | tee "$evidence/proposer-structure.txt"

printf '\n===== per-iteration host-side work in the proposer =====\n'
# Host syncs, .item(), .cpu(), python loops over layers: each is a dispatch stall on Ascend.
for f in "$runtime/vllm_ascend/spec_decode/"*.py; do
  n="$(basename "$f")"
  case "$n" in *dflash*|*llm_base_proposer*) ;; *) continue ;; esac
  printf -- '--- %s ---\n' "$n"
  grep -nE '\.item\(\)|\.cpu\(\)|\.tolist\(\)|synchronize\(\)|torch\.npu\.current_stream|for .* in range\(self\.num_speculative_tokens|for layer' "$f" > "$evidence/host.$n.txt" 2>/dev/null || true
  head -n 30 "$evidence/host.$n.txt"
  printf '  (matches: %s)\n' "$(wc -l < "$evidence/host.$n.txt")"
done | tee "$evidence/host-work.txt"

printf '\n===== draft model layer count and shape =====\n'
for c in /data1/dflash2-models/Qwen3.8-27B-DFlash2/config.json; do
  [[ -f "$c" ]] || continue
  python3 -c "
import json
d = json.load(open('$c', encoding='utf-8'))
keep = {k: v for k, v in d.items() if not isinstance(v, (dict, list)) or k in ('target_layer_ids',)}
print(json.dumps(keep, indent=1, sort_keys=True)[:2000])
"
done | tee "$evidence/draft-config.txt"

echo "EVIDENCE=$evidence"
echo DRAFTER_COST_AUDIT_PASS
