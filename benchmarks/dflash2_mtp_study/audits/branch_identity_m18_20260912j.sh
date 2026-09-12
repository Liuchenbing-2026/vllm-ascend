#!/usr/bin/env bash
set -euo pipefail

# Card-free, and this one can invalidate the study, so it runs before anything else is claimed.
#
# RUNTIME_MANIFEST.txt (read in 20260912e) says the tree every measurement ran on is:
#   classification=speculative local DFlash-only Python optimization over immutable c1b3 runtime
#   source_commit=cbca145742d01d6011caa33a631c46d1165396c7
#   source_base=c1b3f6b70a37d51c3da90a3e118eccb0c4f13e84
#   base_runtime=.../dflash2-mrv1-full-kv-c1b3f6b7-runtime-e6519803
#
# So this is NOT a stock runtime and NOT any upstream branch. It is a local patch series
# (the directory name spells it out: mrv1 + full-kv + device-queryloc-v2), and the manifest
# labels the delta "DFlash-only".
#
# That word is the problem. If the local patches touch the DFlash path but not the MTP path, then
# every DFlash-vs-MTP number I have compares PATCHED DFlash against STOCK MTP -- the two arms are
# not running the same code, and the asymmetry runs in an unknown direction. "speculative" in the
# classification line suggests these patches were never validated either.
#
# What has to be established, in order:
#   (1) the full manifest, verbatim
#   (2) which runtime trees exist on this box and how they chain
#   (3) the exact file-level delta between the tree I measured and its base
#   (4) whether that delta touches the MTP path, the DFlash path, or both
#   (5) whether any of it is a git checkout with a real branch, and what the image ships stock

root=/data2/dflash2-v026-official
rs="$root/runtime-source"
active="$rs/dflash2-mrv1-full-kv-device-queryloc-v2-runtime-e6519803"
base="$rs/dflash2-mrv1-full-kv-c1b3f6b7-runtime-e6519803"
evidence="$root/upstream-audit/branch-identity-20260912j"

trap 'status=$?; printf "BRANCH_ERROR line=%s status=%s command=%q\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

test "$(hostname)" = kylin10-6018
test -d "$active"
test ! -e "$evidence"
mkdir -p "$evidence"

printf '===== (1) the manifest, in full =====\n'
cat "$active/RUNTIME_MANIFEST.txt"

printf '\n===== (2) every runtime tree on this box =====\n'
ls -la "$rs" 2>/dev/null | tee "$evidence/runtime-trees.txt"
for d in "$rs"/*/; do
  [[ -f "$d/RUNTIME_MANIFEST.txt" ]] || continue
  printf -- '\n--- %s ---\n' "$(basename "$d")"
  cat "$d/RUNTIME_MANIFEST.txt"
done

printf '\n===== (3) file-level delta: measured tree vs its declared base =====\n'
if [[ -d "$base" ]]; then
  # Write to files and head the FILES. `grep|head` under pipefail dies of SIGPIPE at exit 141 --
  # that is how 20260910bj died, and how 20260912g died again an hour ago.
  diff -rq "$base" "$active" > "$evidence/delta.txt" 2>&1 || true
  printf 'differing/new/missing entries: %s\n' "$(wc -l < "$evidence/delta.txt")"
  head -n 60 "$evidence/delta.txt"
else
  printf 'declared base runtime NOT PRESENT at %s\n' "$base"
  printf 'the delta cannot be reconstructed on this box.\n'
fi

printf '\n===== (4) does the delta touch MTP, DFlash, or both =====\n'
if [[ -s "$evidence/delta.txt" ]]; then
  awk '{for(i=1;i<=NF;i++) if ($i ~ /\.py$/) print $i}' "$evidence/delta.txt" | sort -u > "$evidence/changed-files.txt"
  printf 'changed .py files: %s\n' "$(wc -l < "$evidence/changed-files.txt")"
  cat "$evidence/changed-files.txt"
  printf -- '\n--- per-file: how big is the change, and which arm does it sit on ---\n'
  while read -r f; do
    [[ -n "$f" ]] || continue
    rel="${f#$active/}"; rel="${rel#$base/}"
    a="$base/$rel"; b="$active/$rel"
    if [[ -f "$a" && -f "$b" ]]; then
      n="$(diff "$a" "$b" | grep -c '^[<>]' || true)"
      printf '%-64s changed_lines=%s\n' "$rel" "$n"
      diff -u "$a" "$b" > "$evidence/diff.$(echo "$rel" | tr '/' '_').txt" 2>&1 || true
    fi
  done < "$evidence/changed-files.txt"
fi

printf '\n===== (5) the actual diffs, bounded =====\n'
for f in "$evidence"/diff.*.txt; do
  [[ -f "$f" ]] || continue
  printf -- '\n########## %s (%s lines) ##########\n' "$(basename "$f")" "$(wc -l < "$f")"
  head -n 120 "$f"
done

printf '\n===== (6) is anything here a git checkout =====\n'
for d in "$active" "$base" "$root/vllm" "$root"; do
  [[ -d "$d" ]] || continue
  if [[ -d "$d/.git" ]]; then
    printf '%-70s GIT  branch=%s head=%s dirty=%s\n' "$d" \
      "$(cd "$d" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')" \
      "$(cd "$d" && git rev-parse HEAD 2>/dev/null || echo '?')" \
      "$(cd "$d" && git status --porcelain 2>/dev/null | wc -l)"
  else
    printf '%-70s no .git\n' "$d"
  fi
done

printf '\n===== (7) what does the image ship as stock, for reference =====\n'
docker run --rm --entrypoint /bin/bash vllm-ascend:pr14171-v026-runtime-20260821 -lc '
  for p in /vllm-workspace/vllm-ascend /usr/local/lib/python3*/site-packages/vllm_ascend; do
    [ -d "$p" ] && { echo "found $p"; ls "$p"/spec_decode/ 2>/dev/null | head -20; }
  done
  python3 -c "import vllm; print(\"vllm version:\", vllm.__version__)" 2>/dev/null || true
' > "$evidence/image-stock.txt" 2>&1 || true
head -n 40 "$evidence/image-stock.txt"

echo "EVIDENCE=$evidence"
echo BRANCH_IDENTITY_PASS
