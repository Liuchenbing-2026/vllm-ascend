#!/bin/bash
# GLM-5.3-Flash -> w8a8 with msmodelslim.
#
#   b0829    branch glm5_next_quant_0829 as-is        (shared experts quantized)   <- recommended
#   b0829se  0829 + the one-line shared_experts exclude (shared experts stay bf16)
#
# Do NOT use branch glm5_next_quant_0830: it predates the ViT-export fix (4b73b71) and
# silently drops all 347 model.visual.* tensors. See docs/01-quantization.md.
#
# ~72 min on 8 cards. Peak ~20 GB HBM per card during the linear_quant DTS phase --
# the cards must be otherwise idle or it will OOM.
set -e

SRC=${SRC:-/data02/GLM-5.3-Flash-BF16}
WORK=${WORK:-/data02/glm53_quant}
VARIANT=${1:-b0829}        # b0829 | b0829se
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}

PIPENV="-e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
        -e PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn"   # the image's pip index is a k8s-internal cache

mkdir -p "$WORK"
TREE="$WORK/msmodelslim_$VARIANT"
if [ ! -d "$TREE" ]; then
  git clone -q https://gitcode.com/qq_46439621/msmodelslim.git "$TREE"
  git -C "$TREE" checkout -q glm5_next_quant_0829
  if [ "$VARIANT" = "b0829se" ]; then
    # the ONE line that branch glm5_next_quant_0830 adds, applied on top of 0829
    python3 - "$TREE" <<'PY'
import io, sys
p = sys.argv[1] + "/lab_practice/glm_5_next/glm_5_next_w8a8.yaml"
s = io.open(p, encoding="utf-8").read()
old = '      exclude:\n        - "*gate"\n'
new = '      exclude:\n        - "*gate"\n        - \'*shared_experts*\'\n'
assert old in s, "anchor not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched", p)
PY
  fi
fi

echo "=== installing msmodelslim from $TREE"
( cd "$TREE" && bash install.sh ) > "$WORK/install_$VARIANT.log" 2>&1
python3 -c "import msmodelslim,os;p=os.path.dirname(msmodelslim.__file__);print(open(os.path.join(p,'lab_practice/glm_5_next/glm_5_next_w8a8.yaml')).read())" \
  | grep -A4 'exclude:'

OUT_DIR=/data02/GLM-5.3-Flash-w8a8-$VARIANT
LOG=$WORK/quant_$VARIANT.log
rm -rf "$OUT_DIR"

export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
DEV_ARG="npu:$DEVICES"
echo "=== quantizing -> $OUT_DIR  (log $LOG)"
nohup setsid msmodelslim quant \
  --model_path "$SRC" \
  --save_path  "$OUT_DIR" \
  --device "$DEV_ARG" \
  --model_type GLM-5.3-Flash \
  --quant_type w8a8 \
  --trust_remote_code True \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid=$!"
echo "watch:  grep -a -oE 'Creating decoder layer [0-9]+' $LOG | tail -1"
echo "done when the log ends with '===========SUCCESS==========='"
