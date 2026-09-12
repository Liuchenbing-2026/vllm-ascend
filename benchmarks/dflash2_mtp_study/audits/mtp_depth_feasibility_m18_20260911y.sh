#!/usr/bin/env bash
set -euo pipefail

# Card-free. Can qwen3_5_mtp run deeper than 3 steps on this stack?
#
# Why it matters: the LMSYS figure everyone quotes ("DFlash 1.5x MTP") is measured against MTP at
# SEVEN steps. I have only ever run MTP at 3. Those are different baselines in the direction that
# matters -- each extra MTP step is another SEQUENTIAL head call, so MTP-7's per-iteration cost is
# much higher than MTP-3's, and block-parallel drafting is exactly what cashes that in. If MTP-7
# is reachable here, the honest comparison against DFlash K7 is MTP-7, not MTP-3; if it is not
# reachable, then MTP-3 is a structurally stronger baseline than the published one and that is
# itself the finding.
#
# Three things decide it, all readable without a card:
#   1. how many MTP modules the Qwen3.8-27B checkpoint actually ships (1 loopable head vs N heads)
#   2. whether vLLM/vllm-ascend clamps or rejects num_speculative_tokens for method qwen3_5_mtp
#   3. whether the weight names show per-step heads

root=/data2/dflash2-v026-official
vllm_src="$root/vllm"
runtime="$root/runtime-source/dflash2-mrv1-full-kv-device-queryloc-v2-runtime-e6519803"
model=/data1/dflash2-models/Qwen3.8-27B
evidence="$root/upstream-audit/mtp-depth-20260911y"

trap 'status=$?; printf "MTPDEPTH_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -d "$runtime"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (1) what the target config says about MTP =====\n'
python3 - "$model" "$evidence" <<'CFGPY'
import json, sys
from pathlib import Path
m, ev = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((m / "config.json").read_text(encoding="utf-8"))
keys = [k for k in cfg if any(t in k.lower() for t in ("nextn", "mtp", "predict", "draft", "spec", "num_hidden"))]
print("mtp-ish top-level keys: %s" % (keys or "(none)"))
for k in keys:
    print("  %s = %s" % (k, cfg[k]))
print("num_hidden_layers = %s" % cfg.get("num_hidden_layers"))
(ev / "target-config-keys.json").write_text(json.dumps({k: cfg[k] for k in keys}, indent=1), encoding="utf-8")
CFGPY

printf '\n===== (2) how many MTP head tensors are in the checkpoint =====\n'
python3 - "$model" "$evidence" <<'WPY'
import json, re, sys
from collections import Counter
from pathlib import Path
m, ev = Path(sys.argv[1]), Path(sys.argv[2])
idx = m / "model.safetensors.index.json"
if not idx.is_file():
    print("no model.safetensors.index.json")
    raise SystemExit(0)
wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map", {})
hits = sorted(n for n in wm if any(t in n.lower() for t in ("mtp", "nextn", "eh_proj", "enorm", "hnorm")))
print("total tensors: %d   mtp-ish tensors: %d" % (len(wm), len(hits)))
# Per-step heads show up as an index inside the name; one loopable head does not.
steps = Counter()
for n in hits:
    mm = re.search(r"(?:mtp|nextn)[._](?:layers?[._])?(\d+)", n.lower())
    steps[mm.group(1) if mm else "-"] += 1
print("distinct step indices in mtp tensor names: %s" % dict(steps))
for n in hits[:30]:
    print("  " + n)
(ev / "mtp-tensors.txt").write_text("\n".join(hits), encoding="utf-8")
WPY

printf '\n===== (3) does the code clamp num_speculative_tokens for qwen3_5_mtp =====\n'
for d in "$vllm_src/vllm/config" "$vllm_src/vllm/transformers_utils" "$runtime/vllm_ascend"; do
  [[ -d "$d" ]] || continue
  printf -- '--- %s ---\n' "$d"
  grep -rn --include=*.py -e 'num_speculative_tokens' -e 'n_predict' -e 'num_nextn_predict_layers' "$d" \
    > "$evidence/clamp.tmp" 2>/dev/null || true
  grep -iE 'min\(|max\(|assert|raise|clamp|> *[0-9]|<= *[0-9]|num_nextn' "$evidence/clamp.tmp" > "$evidence/clamp.filtered" 2>/dev/null || true
  head -n 25 "$evidence/clamp.filtered"
  printf '  (mentions: %s, constraint-shaped: %s)\n' "$(wc -l < "$evidence/clamp.tmp")" "$(wc -l < "$evidence/clamp.filtered")"
done | tee "$evidence/clamp-sites.txt"
rm -f "$evidence/clamp.tmp" "$evidence/clamp.filtered"

printf '\n===== (4) the mtp proposer: is the head called in a loop =====\n'
for f in "$runtime/vllm_ascend/spec_decode/"*.py; do
  n="$(basename "$f")"
  case "$n" in *mtp*|*llm_base_proposer*) ;; *) continue ;; esac
  printf -- '--- %s ---\n' "$n"
  grep -nE 'range\(self\.num_speculative_tokens|num_speculative_tokens' "$f" > "$evidence/loop.$n.txt" 2>/dev/null || true
  head -n 20 "$evidence/loop.$n.txt"
done | tee "$evidence/mtp-loop.txt"

printf '\n===== (5) what the k7 service actually reported for its own spec config =====\n'
for d in "$root"/upstream-audit/benchserve-c32gsm-mtp3-20260911; do
  [[ -f "$d/readiness.log" ]] || continue
  grep -oE "SpeculativeConfig\(method='[^']*'[^)]{0,200}" "$d/readiness.log" > "$evidence/specconf.txt" 2>/dev/null || true
  head -n 3 "$evidence/specconf.txt"
done

echo "EVIDENCE=$evidence"
echo MTP_DEPTH_FEASIBILITY_PASS
