#!/usr/bin/env bash
set -euo pipefail

# Card-free follow-up to 20260911t.
#
# 20260911t killed the "drafter runs eager" hypothesis: the logs say
#   [spec_decode/base] Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL
# for all three arms (k7 / k1 / mtp3), and the proposer tree has full ACLGraph plumbing.
# So the ~38 ms/iteration is NOT explained by per-operator host dispatch in an uncaptured drafter.
#
# Two candidates survive, and this audit separates them. Both are card-free.
#
# (A) The drafter is not small. config.json says 5 layers, hidden 5120, ffn 17408,
#     vocab 248320, tie_word_embeddings=false -> it carries its OWN embed table AND its own
#     lm_head (1.27B params each). Against a ~23B target that is ~18% of the weights, not the
#     ~8% that "5 of 64 layers" suggests. And block_size=8 means its lm_head runs on 8x the
#     rows per iteration that MTP3's single head does. Measure the real parameter bytes.
#
# (B) The wrapper is installed but the drafter's graph never actually replays -- the criterion
#     is a Capturing/Replaying line for the DRAFT model, and 20260911t only found target-model
#     capture bars (4 / 2 / 3 graphs = the batch x (K+1) sizes <= 256). A wrapper that misses
#     its shape silently falls back to eager with no warning.
#     Plus llm_base_proposer.py has .item()/.tolist() in the metadata build path (lines 2016,
#     2169, 2243, 2265). Each is a device->host sync. The question that decides whether they
#     explain a DFlash-vs-MTP3 *difference* is whether they sit on the shared path or the
#     dflash-only path -- print the enclosing def for each.

root=/data2/dflash2-v026-official
runtime="$root/runtime-source/dflash2-mrv1-full-kv-device-queryloc-v2-runtime-e6519803"
prop="$runtime/vllm_ascend/spec_decode"
evidence="$root/upstream-audit/drafter-cost-audit2-20260911u"

trap 'status=$?; printf "AUDIT2_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -d "$prop"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (A) how big is the drafter really =====\n'
for d in /data1/dflash2-models/Qwen3.8-27B /data1/dflash2-models/Qwen3.8-27B-DFlash2; do
  [[ -d "$d" ]] || continue
  bytes="$(du -sb "$d" 2>/dev/null | awk '{print $1}')"
  wb="$(find "$d" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}')"
  printf '%-52s dir=%s weight_bytes=%s approx_params_bf16=%.3fB\n' "$(basename "$d")" "$bytes" "$wb" "$(awk -v w="$wb" 'BEGIN{print w/2e9}')"
done | tee "$evidence/sizes.txt"

printf '\n--- does the drafter keep its own lm_head / embed, and how big are they ---\n'
python3 - "$evidence" <<'SZPY'
import json, sys
from pathlib import Path
ev = Path(sys.argv[1])
d = Path("/data1/dflash2-models/Qwen3.8-27B-DFlash2")
idx = d / "model.safetensors.index.json"
rows = []
if idx.is_file():
    wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map", {})
    for name in sorted(wm):
        if any(k in name for k in ("lm_head", "embed_tokens", "selector", "conv")):
            rows.append(name)
    print("draft tensors of interest (%d of %d total):" % (len(rows), len(wm)))
    for r in rows[:40]:
        print("  " + r)
else:
    print("no index json; listing safetensors headers instead")
cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
H = cfg.get("hidden_size"); V = cfg.get("vocab_size"); L = cfg.get("num_hidden_layers")
I = cfg.get("intermediate_size"); nh = cfg.get("num_attention_heads"); nkv = cfg.get("num_key_value_heads")
hd = cfg.get("head_dim", H // nh if nh else 0)
attn = H*nh*hd + 2*H*nkv*hd + nh*hd*H
mlp = 3*H*I
print("\nper draft layer: attn=%.1fM mlp=%.1fM total=%.1fM" % (attn/1e6, mlp/1e6, (attn+mlp)/1e6))
print("%d draft layers          = %.3fB" % (L, L*(attn+mlp)/1e9))
print("embed  %d x %d          = %.3fB   (tie_word_embeddings=%s)" % (V, H, V*H/1e9, cfg.get("tie_word_embeddings")))
print("lm_head %d x %d         = %.3fB" % (V, H, V*H/1e9))
print("DRAFT TOTAL (if untied)  = %.3fB" % ((L*(attn+mlp) + 2*V*H)/1e9))
print("\nlm_head GEMM per iteration at C32:")
for tag, rowsn in (("MTP3 (3 seq calls, 32 seqs)", 3*32), ("DFlash block_size=8 (32 seqs)", 8*32)):
    print("  %-32s rows=%4d  flops=%.1f GFLOP" % (tag, rowsn, 2*rowsn*H*V/1e9))
SZPY

printf '\n===== (B1) did the DRAFT model graph ever capture or replay =====\n'
for d in "$root"/upstream-audit/benchserve-c32gsm-k7-20260911 \
         "$root"/upstream-audit/benchserve-c32gsm-mtp3-20260911; do
  [[ -f "$d/readiness.log" ]] || continue
  printf -- '--- %s ---\n' "$(basename "$d")"
  grep -inE 'draft|proposer|eagle|spec_decode' "$d/readiness.log" > "$evidence/draft.tmp" 2>/dev/null || true
  printf '  lines mentioning draft/proposer: %s\n' "$(wc -l < "$evidence/draft.tmp")"
  head -n 25 "$evidence/draft.tmp"
  grep -cE 'Capturing CUDA graphs' "$d/readiness.log" > "$evidence/capbars.tmp" 2>/dev/null || echo 0 > "$evidence/capbars.tmp"
  printf '  "Capturing CUDA graphs" progress bars: %s\n' "$(cat "$evidence/capbars.tmp")"
done | tee "$evidence/draft-capture.txt"
rm -f "$evidence/draft.tmp" "$evidence/capbars.tmp"

printf '\n===== (B2) the ACLGraphWrapper site and what shapes it captures =====\n'
sed -n '690,740p' "$prop/llm_base_proposer.py" > "$evidence/wrap-site.txt"
cat "$evidence/wrap-site.txt"

printf '\n===== (B3) is the graph actually entered on the decode path =====\n'
sed -n '920,960p' "$prop/llm_base_proposer.py" > "$evidence/decode-path.txt"
cat "$evidence/decode-path.txt"
printf -- '--- and the dflash-specific one ---\n'
sed -n '235,265p' "$prop/dflash_proposer.py" > "$evidence/dflash-decode-path.txt"
cat "$evidence/dflash-decode-path.txt"

printf '\n===== (C) which host syncs are dflash-only vs shared with MTP3 =====\n'
python3 - "$prop/llm_base_proposer.py" <<'CTXPY'
import re, sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
targets = [2016, 2169, 2243, 2265, 1430, 1481, 842]
defs = []
for i, l in enumerate(lines, start=1):
    m = re.match(r"(\s*)def (\w+)", l)
    if m:
        defs.append((i, len(m.group(1)), m.group(2)))
for t in targets:
    if t > len(lines):
        continue
    owner = None
    for (ln, indent, name) in defs:
        if ln <= t:
            owner = (ln, name)
        else:
            break
    print("line %-5d in def %-40s | %s" % (t, owner[1] if owner else "?", lines[t-1].strip()[:110]))
CTXPY

printf '\n--- which proposer class each method belongs to ---\n'
grep -nE '^class |^\s{4}def (propose|_propose|load_model|dummy_run|_prepare|build)' "$prop/llm_base_proposer.py" > "$evidence/base-classes.txt" 2>/dev/null || true
head -n 50 "$evidence/base-classes.txt"

printf '\n--- _maybe_share_lm_head: does the drafter reuse the target head ---\n'
sed -n '70,100p' "$prop/dflash2_proposer.py" > "$evidence/share-head.txt"
cat "$evidence/share-head.txt"

echo "EVIDENCE=$evidence"
echo DRAFTER_COST_AUDIT2_PASS
