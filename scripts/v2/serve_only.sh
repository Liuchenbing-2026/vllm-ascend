#!/bin/bash
set -u
ARM=$1
VA=$(python3 -c 'import os,vllm_ascend;print(os.path.dirname(vllm_ascend.__file__))' 2>/dev/null | tail -1)
LORA=$VA/lora
[ -f /work/stock_lora_ops.py ] || cp $LORA/lora_ops.py /work/stock_lora_ops.py
rm -f $LORA/lora_ops_triton.py $LORA/lora_ops_triton_kernels.py $LORA/lora_cpp_launcher.cpp $LORA/lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so
case "$ARM" in
  ascendc)   cp /work/stock_lora_ops.py $LORA/lora_ops.py ;;
  triton_v1) cp /work/branch/lora_ops.py /work/branch/lora_ops_triton.py /work/branch/lora_ops_triton_kernels.py /work/branch/lora_cpp_launcher.cpp /work/branch/lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so $LORA/ ;;
  triton_v2) cp /work/v2/lora_ops.py /work/v2/lora_ops_triton.py /work/v2/lora_ops_triton_kernels.py /work/v2/lora_cpp_launcher.cpp /work/v2/lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so $LORA/ ;;
esac
find $LORA -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
for p in $(pgrep -f "bin/vllm serve" 2>/dev/null); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "VLLM::" 2>/dev/null); do kill -9 $p 2>/dev/null; done
sleep 8
export ASCEND_RT_VISIBLE_DEVICES=0 TRITON_LORA_CPP=1
unset TRITON_LORA_TIME TRITON_LORA_CPP_DEBUG
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
nohup vllm serve /data1/Qwen3.6-27B --served-model-name base --host 127.0.0.1 --port 7519 \
  --max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 8 \
  --gpu-memory-utilization 0.93 --trust-remote-code \
  --enable-lora --max-loras 2 --max-lora-rank 16 \
  --lora-modules '{"name":"openscad","path":"/data1/lora-hf/q38-27b-r16-std"}' \
  > /work/serve_${ARM}.log 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 3 http://127.0.0.1:7519/v1/models >/dev/null 2>&1 && { echo READY; exit 0; }
  grep -qE "Traceback|Engine core initialization failed" /work/serve_${ARM}.log && { echo FAILED; tail -30 /work/serve_${ARM}.log; exit 1; }
  sleep 10
done
echo TIMEOUT; exit 1
