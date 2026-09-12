#!/usr/bin/env bash
set -euo pipefail

# Card-free. Are we running the SAME draft weights the published results were produced with?
#
# This is the one axis that, if wrong, invalidates the acceptance comparison outright -- and it
# was never checked. I had been explaining my acceptance (5.18) vs SGLang's (5.8) partly by
# "block 16 vs my block 8", but the upstream card settles that differently: block_size 8 IS the
# official configuration for the Qwen3.8-27B draft. LMSYS's block-16 figure is for
# Qwen3.5-397B-A17B, an entirely different target. So that explanation is gone and the weights
# themselves have to be pinned instead of assumed.
#
# Ground truth from huggingface.co/z-lab/Qwen3.8-27B-DFlash2 (revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4,
# last modified 2026-08-19):
#   model.safetensors  3848817896 bytes  sha256 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c
#   config.json        1239 bytes
#   dflash_config: block_size 8, conv_kernel_size 2, conv_group_size 16, selector_rank 256,
#                  selector_top_k 16, mask_token_id 248070, target_layer_ids [5,19,33,47,61]
#   architectures DFlash2DraftModel, 5 layers, hidden 5120, vocab 248320, sliding_window 2048,
#   tie_word_embeddings false
#
# 20260911u already showed the local weight bytes are 3848817896 -- an exact size match. Size is
# not content, so hash it. A size-equal but content-different checkpoint is exactly the failure
# mode that [[modelscope-parallel-download-corruption]] recorded (right size, wrong SHA).

draft=/data1/dflash2-models/Qwen3.8-27B-DFlash2
target=/data1/dflash2-models/Qwen3.8-27B
evidence=/data2/dflash2-v026-official/upstream-audit/weights-identity-20260912f

official_sha=67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c
official_bytes=3848817896

trap 'status=$?; printf "WEIGHTS_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -d "$draft"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (1) what is actually on disk =====\n'
ls -la "$draft" | tee "$evidence/draft-ls.txt"

printf '\n===== (2) does the draft checkpoint hash match the published one =====\n'
f="$draft/model.safetensors"
if [[ -f "$f" ]]; then
  b="$(stat -c %s "$f")"
  printf 'bytes  local=%s  official=%s  %s\n' "$b" "$official_bytes" \
    "$([[ "$b" == "$official_bytes" ]] && echo SIZE_MATCH || echo SIZE_DIFFERS)"
  printf 'hashing %s (about 4 GB, one pass)...\n' "$f"
  h="$(sha256sum "$f" | awk '{print $1}')"
  printf 'sha256 local   =%s\n' "$h"
  printf 'sha256 official=%s\n' "$official_sha"
  if [[ "$h" == "$official_sha" ]]; then
    printf 'DRAFT_WEIGHTS=IDENTICAL_TO_PUBLISHED\n'
  else
    printf 'DRAFT_WEIGHTS=DIFFERENT -- every acceptance number in this study is against a different drafter\n'
  fi
  printf '%s  %s\n' "$h" "$f" > "$evidence/draft.sha256"
else
  printf 'no single model.safetensors; sharded checkpoint. per-shard hashes:\n'
  find "$draft" -maxdepth 1 -name '*.safetensors' -print0 | sort -z | xargs -0 sha256sum \
    | tee "$evidence/draft.sha256"
fi

printf '\n===== (3) the full config, including the dflash_config block =====\n'
python3 -c "
import json
d = json.load(open('$draft/config.json', encoding='utf-8'))
print(json.dumps(d, indent=1, sort_keys=True))
" | tee "$evidence/draft-config.json"

printf '\n--- compare against the published config field by field ---\n'
python3 -c "
import json
d = json.load(open('$draft/config.json', encoding='utf-8'))
fc = d.get('dflash_config', {}) or {}
want_top = {'architectures': ['DFlash2DraftModel'], 'num_hidden_layers': 5, 'hidden_size': 5120,
            'vocab_size': 248320, 'sliding_window': 2048, 'tie_word_embeddings': False}
want_fc  = {'block_size': 8, 'conv_kernel_size': 2, 'conv_group_size': 16, 'selector_rank': 256,
            'selector_top_k': 16, 'mask_token_id': 248070,
            'target_layer_ids': [5, 19, 33, 47, 61]}
bad = 0
for k, v in want_top.items():
    got = d.get(k)
    ok = (got == v)
    bad += 0 if ok else 1
    print('  %-22s local=%-28s official=%-28s %s' % (k, got, v, 'ok' if ok else 'MISMATCH'))
for k, v in want_fc.items():
    got = fc.get(k)
    ok = (got == v)
    bad += 0 if ok else 1
    print('  dflash_config.%-8s local=%-28s official=%-28s %s' % (k, got, v, 'ok' if ok else 'MISMATCH'))
print('CONFIG_MISMATCHES=%d' % bad)
"

printf '\n===== (4) provenance traces: where did these files come from =====\n'
for p in "$draft/README.md" "$draft/.gitattributes"; do
  [[ -f "$p" ]] && printf -- '--- %s (%s bytes) ---\n' "$p" "$(stat -c %s "$p")"
done
find "$draft" -maxdepth 2 \( -name '*.json' -o -name 'README*' -o -name '.gitattributes' \) \
  -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' 2>/dev/null | sort
# A huggingface-cli / snapshot_download leaves the source repo id and revision behind.
for c in "$draft/.cache/huggingface" "$draft/../.cache/huggingface"; do
  [[ -d "$c" ]] && { printf -- '--- %s ---\n' "$c"; find "$c" -maxdepth 3 | head -n 20; }
done || true

printf '\n===== (5) the target model, for the record =====\n'
# Both arms share the target, so it cannot bias DFlash-vs-MTP -- except that the MTP head LIVES
# in the target checkpoint, so a different target revision means a different MTP drafter.
ls -la "$target" | head -n 20
python3 -c "
import json
d = json.load(open('$target/config.json', encoding='utf-8'))
keys = [k for k in d if not isinstance(d[k], (dict, list))]
print(json.dumps({k: d[k] for k in sorted(keys)}, indent=1)[:1200])
" | tee "$evidence/target-config.txt"

echo "EVIDENCE=$evidence"
echo WEIGHTS_IDENTITY_PASS
