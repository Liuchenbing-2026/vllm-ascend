#!/bin/bash
# /nt/leg_clean.sh -- the tidied patch must reproduce the monkeypatch's result.
#
# "Refactored but never run" is exactly the failure this investigation kept
# hitting: the code compiles, the service starts, and the arm quietly behaves
# like the do-nothing arm. So the tidied form gets its own instance next to a
# fresh c0, and the judge is the same one that separated them before --
# acceptance. Two arms, interleaved cl / c0 / cl so a drift cannot land on one.
exec > /nt/logs/clean.out 2>&1
cd /vllm-workspace/vllm-ascend || exit 1

cleanup() {
  bash /nt/stop.sh >/dev/null 2>&1
  python3 /nt/maskfix_clean.py --revert | tail -2
  python3 /nt/nt_patch.py --revert | tail -1
  # approx_patch.py has no --revert and nt_patch.py's FILES list does not cover
  # envs.py, so its four added lines survived every previous leg. Harmless at
  # runtime (the flag defaults to 0) but it leaves a shared machine dirty, and a
  # later reader cannot tell it from someone else's edit. Take the tree back to
  # what git says it should be, and drop the backup files the patchers leave.
  git checkout -- vllm_ascend/envs.py 2>/dev/null
  find vllm_ascend -name '*.nt-*orig' -delete 2>/dev/null
  git status --porcelain | head -5
  md5sum vllm_ascend/attention/attention_v1.py vllm_ascend/envs.py
}
trap cleanup EXIT

export HCCL_DETERMINISTIC=true

run_arm() {
  local tag="$1" arm="$2"
  echo "##### arm $tag ($arm) $(date -Is)"
  bash /nt/stop.sh >/dev/null 2>&1
  python3 /nt/maskfix_clean.py --revert >/dev/null 2>&1
  python3 /nt/nt_patch.py --revert >/dev/null
  python3 /nt/nt_patch.py --fix >/dev/null
  python3 /nt/approx_patch.py >/dev/null
  export VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1
  export VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK=0
  if [ "$arm" = "cl" ]; then
    python3 /nt/maskfix_clean.py | tail -2
    export VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK=1
  fi
  python3 -m py_compile vllm_ascend/attention/attention_v1.py vllm_ascend/envs.py \
    || { echo "COMPILE_FAILED $tag"; return 1; }
  rm -rf /root/.cache/vllm/torch_compile_cache

  MRV2=1 bash /nt/serve.sh "$tag"
  bash /nt/wait_ready.sh "$tag" 1800 || { echo "SERVE_FAILED $tag"; tail -30 "/nt/logs/serve_$tag.log"; return 1; }
  python3 /nt/accept_probe.py "${tag}_p1" 256
  bash /nt/bench.sh "$tag" warm 48 8 | tail -2
  bash /nt/bench.sh "$tag" measure 96 8
  echo "--- pooled acceptance $tag"
  python3 /nt/pooled_acc.py "/nt/logs/serve_$tag.log"
  bash /nt/stop.sh >/dev/null 2>&1
  sleep 5
}

echo "##### clean-form leg $(date -Is)"
run_arm cl1 cl
run_arm c0x c0
run_arm cl2 cl

echo "##### text identity vs the monkeypatch leg"
python3 /nt/text_diff.py /nt/logs/accept_mf1_p1.json /nt/logs/accept_cl1_p1.json \
  /nt/logs/accept_c0x_p1.json /nt/logs/accept_cl2_p1.json
echo "##### DONE $(date -Is)"
