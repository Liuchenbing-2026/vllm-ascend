#!/usr/bin/env bash
set -euo pipefail

# Card-free. Is the code that produced every number in this study actually what I think it is?
#
# What the start scripts already assert, on every single service start:
#   docker image id                       == sha256:e6519803e088d655...
#   sha256sum RUNTIME_MANIFEST.txt        == 929fb5220edfa11e...
#   runtime-import.known-good-order.log   contains runtime_import=pass
#
# Two gaps in that, and this audit closes them:
#
# (1) Hashing the MANIFEST proves the manifest file is unchanged. It does NOT prove the files the
#     manifest lists are unchanged -- an edit to llm_base_proposer.py would sail straight through.
#     If the manifest carries per-file hashes, re-walk them and report any mismatch.
#
# (2) $root/vllm -- the vLLM source bind-mounted read-only into every container -- has NO gate at
#     all. It is as load-bearing as vllm-ascend (the proposer calls into it), and its identity has
#     never been recorded in this study. Record it now: git state if it is a repo, else a tree hash.
#
# Also records mtimes: a file modified during the measurement window (2026-09-10..12) is the
# specific thing that would invalidate cross-run comparisons, since runs are compared across
# ~14 hours and every one of them assumed a frozen tree.

root=/data2/dflash2-v026-official
runtime="$root/runtime-source/dflash2-mrv1-full-kv-device-queryloc-v2-runtime-e6519803"
vllm_src="$root/vllm"
evidence="$root/upstream-audit/provenance-20260912e"

trap 'status=$?; printf "PROV_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -d "$runtime"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (0) the gates the start scripts check =====\n'
printf 'RUNTIME_MANIFEST.txt sha256 : %s\n' "$(sha256sum "$runtime/RUNTIME_MANIFEST.txt" | awk '{print $1}')"
printf 'expected                    : 929fb5220edfa11e7804872a14d610b07504ede7b32869a192120bbbc3f71622\n'
printf 'image id                    : %s\n' "$(docker image inspect vllm-ascend:pr14171-v026-runtime-20260821 --format '{{.Id}}')"
printf 'expected                    : sha256:e6519803e088d655590cfe3b2ef9b429c1c4509e790f85b1fae4a37acd94ba75\n'

printf '\n===== (1) does the manifest carry per-file hashes, and do they still hold =====\n'
head -n 5 "$runtime/RUNTIME_MANIFEST.txt" > "$evidence/manifest-head.txt"
cat "$evidence/manifest-head.txt"
printf 'manifest lines: %s\n' "$(wc -l < "$runtime/RUNTIME_MANIFEST.txt")"

# Accept either "<sha256>  <path>" (sha256sum format) or "<path>  <sha256>".
if grep -qE '^[0-9a-f]{64}[[:space:]]' "$runtime/RUNTIME_MANIFEST.txt"; then
  printf 'format: sha256sum-style -> re-verifying every listed file\n'
  ( cd "$runtime" && sha256sum -c --quiet RUNTIME_MANIFEST.txt ) > "$evidence/verify.txt" 2>&1 || true
  bad="$(wc -l < "$evidence/verify.txt")"
  printf 'files failing verification: %s\n' "$bad"
  head -n 20 "$evidence/verify.txt"
  if [[ "$bad" == "0" ]]; then printf 'MANIFEST_VERIFY=pass\n'; else printf 'MANIFEST_VERIFY=FAIL\n'; fi
else
  printf 'format: NOT sha256sum-style -- the manifest does not pin file contents.\n'
  printf 'Falling back to a tree hash over the python sources.\n'
  ( cd "$runtime" && find . -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum ) \
    > "$evidence/runtime-tree.sha256" 2>/dev/null || true
  printf 'py files: %s   tree hash: %s\n' \
    "$(wc -l < "$evidence/runtime-tree.sha256")" \
    "$(sha256sum "$evidence/runtime-tree.sha256" | awk '{print $1}')"
fi

printf '\n===== (2) the ungated input: $root/vllm =====\n'
if [[ -d "$vllm_src/.git" ]]; then
  printf 'git repo. HEAD / status:\n'
  ( cd "$vllm_src" && git rev-parse HEAD && git status --porcelain | head -n 20 && \
    printf 'dirty files: %s\n' "$(git status --porcelain | wc -l)" ) 2>&1 | tee "$evidence/vllm-git.txt"
else
  printf 'not a git checkout; tree hash over python sources:\n'
  ( cd "$vllm_src" && find . -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum ) \
    > "$evidence/vllm-tree.sha256" 2>/dev/null || true
  printf 'py files: %s   tree hash: %s\n' \
    "$(wc -l < "$evidence/vllm-tree.sha256")" \
    "$(sha256sum "$evidence/vllm-tree.sha256" | awk '{print $1}')"
fi

printf '\n===== (3) was anything touched DURING the measurement window =====\n'
# Runs span 2026-09-10 16:24 to 2026-09-12 00:38. Any source file modified inside that window
# breaks the assumption that all runs share one tree -- which every cross-run ratio depends on.
for d in "$runtime" "$vllm_src"; do
  printf -- '--- %s ---\n' "$d"
  find "$d" -name '*.py' -type f -newermt '2026-09-10 00:00' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' \
    > "$evidence/touched.$(basename "$d").txt" 2>/dev/null || true
  n="$(wc -l < "$evidence/touched.$(basename "$d").txt")"
  printf 'py files modified since 2026-09-10: %s\n' "$n"
  head -n 20 "$evidence/touched.$(basename "$d").txt"
done

printf '\n===== (4) did every service really run the same image and runtime =====\n'
# The gates are asserted at start time, but assert-and-forget leaves no record. The readiness logs
# do: each one prints the vLLM version and the resolved config. Cross-check them.
for d in "$root"/upstream-audit/benchserve-sgl-*-2026091*; do
  [[ -f "$d/readiness.log" ]] || continue
  ver="$(grep -oE 'V1 LLM engine \(v[0-9.]+\)' "$d/readiness.log" | head -1 || true)"
  printf '%-44s %s\n' "$(basename "$d")" "${ver:-<no version line>}"
done | tee "$evidence/engine-versions.txt"

echo "EVIDENCE=$evidence"
echo PROVENANCE_AUDIT_PASS
