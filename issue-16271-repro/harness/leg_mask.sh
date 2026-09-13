#!/bin/bash
# /nt/leg_mask.sh -- does the device-side tail mask buy the acceptance back
# end to end, and what does it cost?
#
#   base  exact device path for the draft build: correct, but pays the blocking
#         seq_lens.tolist() every draft build. The accuracy reference.
#   c0    optimistic bound, no mask (the shipped opt-in switch): no sync, but the
#         kernel attends the rolled-back tail. ~6% acceptance lost.
#   mf    optimistic bound + per-request device mask, BSND + sparse_mode 0:
#         no sync AND no tail. Should land on base's acceptance at c0's dispatch.
#
# Unlike the earlier four-arm leg, these arms do NOT issue the same work -- mf
# changes the kernel and adds a mask build -- so throughput and ITL are part of
# the judge here, not just acceptance.
#
# Order is a palindrome (base c0 mf mf c0 base) so each arm gets two instances
# placed symmetrically in time: restart is the variance unit here, and a
# monotone drift over the leg would otherwise land entirely on one arm.
exec > /nt/logs/mask.out 2>&1
cd /vllm-workspace/vllm-ascend || exit 1

cleanup() {
  bash /nt/stop.sh >/dev/null 2>&1
  python3 /nt/maskfix_patch.py --revert | tail -1
  python3 /nt/nt_patch.py --revert | tail -1
  md5sum vllm_ascend/attention/attention_v1.py vllm_ascend/attention/utils.py \
         vllm_ascend/worker/v2/attn_utils.py vllm_ascend/envs.py
}
trap cleanup EXIT

export HCCL_DETERMINISTIC=true   # otherwise greedy decode is not reproducible

run_arm() {
  local tag="$1" arm="$2" verify="$3"
  echo "##### arm $tag ($arm) $(date -Is)"
  bash /nt/stop.sh >/dev/null 2>&1
  python3 /nt/maskfix_patch.py --revert >/dev/null 2>&1
  python3 /nt/nt_patch.py --revert >/dev/null
  python3 /nt/nt_patch.py --fix >/dev/null
  unset NT_MASKFIX NT_MASKFIX_VERIFY
  export VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=0
  if [ "$arm" != "base" ]; then
    python3 /nt/approx_patch.py >/dev/null
    export VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1
  fi
  if [ "$arm" = "mf" ]; then
    python3 /nt/maskfix_patch.py | tail -1
    export NT_MASKFIX=1 NT_MASKFIX_VERIFY="$verify" NT_MASKFIX_EVERY=2000
  fi
  python3 -m py_compile vllm_ascend/attention/attention_v1.py || { echo "COMPILE_FAILED $tag"; return 1; }
  rm -rf /root/.cache/vllm/torch_compile_cache

  MRV2=1 bash /nt/serve.sh "$tag"
  bash /nt/wait_ready.sh "$tag" 1800 || { echo "SERVE_FAILED $tag"; tail -30 "/nt/logs/serve_$tag.log"; return 1; }

  python3 /nt/accept_probe.py "${tag}_p1" 256
  # The first bench against a fresh server is a warm-up, not a measurement.
  bash /nt/bench.sh "$tag" warm 48 8 | tail -3
  bash /nt/bench.sh "$tag" measure 96 8
  echo "--- pooled acceptance $tag"
  python3 /nt/pooled_acc.py "/nt/logs/serve_$tag.log"
  if [ "$arm" = "mf" ]; then
    echo "--- NT_MASKFIX positive control (rank 0); hit=0 would mean this arm IS c0"
    grep -h "NT_MASKFIX\]" "/nt/logs/serve_$tag.log" | tail -6
  fi
  bash /nt/stop.sh >/dev/null 2>&1
  sleep 5
}

echo "##### mask leg $(date -Is)"
run_arm base1 base 0
run_arm c01   c0   0
run_arm mf1   mf   8
run_arm mf2   mf   0
run_arm c02   c0   0
run_arm base2 base 0

echo "##### text identity (every arm must match base byte for byte)"
python3 /nt/text_diff.py /nt/logs/accept_base1_p1.json /nt/logs/accept_base2_p1.json \
  /nt/logs/accept_c01_p1.json /nt/logs/accept_c02_p1.json \
  /nt/logs/accept_mf1_p1.json /nt/logs/accept_mf2_p1.json
echo "##### DONE $(date -Is)"
