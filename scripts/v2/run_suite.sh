#!/bin/bash
# Three-arm serve-level A/B: for each arm restart the serve, run the decode
# share probe, the prefill TTFT probe, and the sequential gsm8k eval.
set -u
cd /work
mkdir -p evals
for ARM in ascendc triton_v1 triton_v2; do
  echo "=== ARM $ARM $(date) ==="
  bash serve_only.sh $ARM || { echo "SERVE FAILED $ARM"; exit 1; }
  python3 probe_lora_share.py $ARM /work/evals/share_${ARM}.json 2>&1
  python3 ttft_probe.py $ARM /work/evals/ttft_${ARM}.json 2>&1
  python3 eval_gsm8k.py $ARM /work/evals/gsm8k_${ARM}.json 64 512 2>&1
done
echo "=== CROSS-ARM COMPARISON ==="
python3 cmp_evals.py /work/evals/gsm8k_ascendc.json /work/evals/gsm8k_triton_v2.json
python3 cmp_evals.py /work/evals/gsm8k_ascendc.json /work/evals/gsm8k_triton_v1.json
echo "SUITE DONE $(date)"
