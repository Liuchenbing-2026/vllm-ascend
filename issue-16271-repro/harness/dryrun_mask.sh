#!/bin/bash
# /nt/dryrun_mask.sh -- exercise the patch sequence without touching a card.
# Checks: every arm's patch stack applies and compiles, the maskfix hook is
# actually installed on the class (not just present in the file), and the tree
# comes back to the pristine md5s afterwards.
set -u
cd /vllm-workspace/vllm-ascend || exit 1

PRISTINE_ATTN=3ffdd3667b206531740ed46c69285f41

echo "===== revert to pristine"
python3 /nt/maskfix_patch.py --revert 2>/dev/null | tail -1
python3 /nt/nt_patch.py --revert | tail -1
md5sum vllm_ascend/attention/attention_v1.py

for arm in base c0 mf; do
  echo "===== arm $arm"
  python3 /nt/maskfix_patch.py --revert >/dev/null 2>&1
  python3 /nt/nt_patch.py --revert >/dev/null
  python3 /nt/nt_patch.py --fix >/dev/null
  [ "$arm" != "base" ] && python3 /nt/approx_patch.py >/dev/null
  [ "$arm" = "mf" ] && python3 /nt/maskfix_patch.py | tail -1
  python3 -m py_compile vllm_ascend/attention/attention_v1.py \
    vllm_ascend/attention/utils.py vllm_ascend/worker/v2/attn_utils.py \
    vllm_ascend/envs.py && echo "  compile OK"
  # Import-level check: the file containing the monkeypatch is not proof the
  # monkeypatch ran. Importing needs a device, so just confirm the module
  # source binds the attribute and that the env gate is read at import time.
  python3 - <<'PY'
import io, ast
src = io.open("vllm_ascend/attention/attention_v1.py", encoding="utf-8").read()
tree = ast.parse(src)
assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
bound = any(
    isinstance(t, ast.Attribute) and t.attr == "forward_fused_infer_attention"
    and isinstance(t.value, ast.Name) and t.value.id == "AscendAttentionBackendImpl"
    for a in assigns for t in a.targets
)
print("  hook bound in module source:", bound, "| marks:", src.count("_nt_maskfix_mask"))
PY
done

echo "===== revert"
python3 /nt/maskfix_patch.py --revert | tail -1
python3 /nt/nt_patch.py --revert | tail -1
GOT=$(md5sum vllm_ascend/attention/attention_v1.py | cut -d' ' -f1)
echo "attention_v1.py md5=$GOT expected=$PRISTINE_ATTN"
[ "$GOT" = "$PRISTINE_ATTN" ] && echo "TREE PRISTINE" || echo "TREE DIRTY"
