#!/usr/bin/env bash
set -euo pipefail

# Card-free. WHICH kou-jing axes are still misaligned with the published SGLang/LMSYS runs?
#
# Settled so far: greedy (yes), prefix caching off (yes, = their RadixCache off), TP2 (yes),
# same target model (Qwen3.8-27B), concurrency swept 1..32 (theirs stops at 16), and EOS (being
# measured right now in 20260911al..ao).
#
# Two axes I have NOT checked, and one of them is potentially bigger than EOS:
#
# (1) THINKING / CHAT TEMPLATE. The LMSYS setup line says "greedy decoding, THINKING ENABLED,
#     max new tokens 4096". My sweep caps output at 256 and may not be applying a chat template
#     at all. Long structured chain-of-thought is the regime speculative decoding is best at:
#     the text is repetitive and highly predictable, and the fixed per-request costs amortise over
#     ~10x more generated tokens. A 256-token raw-completion run is close to the worst case for
#     it. If my runs are /v1/completions with no template, that is a first-order mismatch, not a
#     detail -- so record which endpoint the sweeps actually used and whether a template exists.
#
# (2) BLOCK SIZE. LMSYS used block 16 (results also shown for 8); this checkpoint is configured
#     at 8. Block size is a property of the trained drafter, so if the config says 8 it is not a
#     knob I can turn -- but that has to be established, not assumed, because it caps how much of
#     the published result is reachable here at all.

root=/data2/dflash2-v026-official
model=/data1/dflash2-models/Qwen3.8-27B
draft=/data1/dflash2-models/Qwen3.8-27B-DFlash2
evidence="$root/upstream-audit/align-gaps-20260911ap"

trap 'status=$?; printf "ALIGN_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (1a) which endpoint did each sweep actually use =====\n'
for d in "$root"/upstream-audit/benchserve-sgl-*-2026091*; do
  [[ -f "$d/backend.txt" ]] || continue
  printf '%-46s %s\n' "$(basename "$d")" "$(cat "$d/backend.txt")"
done | tee "$evidence/backends.txt"

printf '\n===== (1b) does the target have a chat template, and does it have a thinking mode =====\n'
python3 - "$model" "$evidence" <<'TPL'
import json, sys
from pathlib import Path
m, ev = Path(sys.argv[1]), Path(sys.argv[2])
tc = m / "tokenizer_config.json"
cfg = json.loads(tc.read_text(encoding="utf-8")) if tc.is_file() else {}
tpl = cfg.get("chat_template")
print("chat_template present: %s" % bool(tpl))
if isinstance(tpl, list):
    print("  (template is a LIST of %d named templates: %s)" % (len(tpl), [t.get("name") for t in tpl]))
    tpl = "\n".join(t.get("template", "") for t in tpl)
if tpl:
    print("  length: %d chars" % len(tpl))
    marks = ["enable_thinking", "thinking", "<think>", "</think>", "reasoning"]
    for mk in marks:
        print("  contains %-16s : %s" % (mk, mk in tpl))
    (ev / "chat_template.jinja").write_text(tpl, encoding="utf-8")
gc = m / "generation_config.json"
if gc.is_file():
    print("\ngeneration_config.json:")
    print(json.dumps(json.loads(gc.read_text(encoding="utf-8")), indent=1, sort_keys=True)[:800])
TPL

printf '\n===== (2) is block size a knob or a checkpoint property =====\n'
python3 - "$draft" <<'BLK'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
keys = [k for k in cfg if any(t in k.lower() for t in ("block", "conv", "selector", "target_layer", "rank", "window"))]
for k in sorted(keys):
    print("  %-28s = %s" % (k, cfg[k]))
print("\nfull key list: %s" % sorted(cfg))
BLK

printf '\n===== (3) how long are the answers when EOS is respected =====\n'
# The EOS-respected cells generated far fewer tokens per request. That number IS the workload:
# it says how much of a 256-token budget a real GSM8K answer uses, and therefore how far my
# fixed-256 kou-jing sits from a chain-of-thought workload.
for d in "$root"/upstream-audit/benchserve-sgl-k7-20260911c \
         "$root"/upstream-audit/benchserve-sgl-k3-20260911b; do
  [[ -d "$d" ]] || continue
  for tag in c1 c1noeos c16 c16noeos; do
    log="$d/bench.$tag.log"
    [[ -f "$log" ]] || continue
    tot="$(grep -E 'Total generated tokens' "$log" | head -1 | tr -dc '0-9' || true)"
    req="$(grep -E 'Successful requests' "$log" | head -1 | tr -dc '0-9' || true)"
    acc="$(grep -E 'Acceptance length' "$log" | head -1 | awk '{print $NF}' || true)"
    printf '%-40s %-10s requests=%-4s generated=%-7s mean_out=%-7s accept=%s\n' \
      "$(basename "$d")" "$tag" "${req:-?}" "${tot:-?}" \
      "$(awk -v t="${tot:-0}" -v r="${req:-1}" 'BEGIN{if(r>0) printf "%.1f", t/r; else print "?"}')" "${acc:-?}"
  done
done | tee "$evidence/output-lengths.txt"

echo "EVIDENCE=$evidence"
echo ALIGN_GAPS_PASS
