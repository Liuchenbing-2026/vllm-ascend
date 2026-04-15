# Ngram 投机解码 AscendC 算子性能对比指南

## 概述

将 ngram 投机解码中 3 个 PyTorch tensor ops 替换为 AscendC 融合算子，消除多次 kernel launch 开销：

| 算子 | 原实现 | 优化后 |
|------|--------|--------|
| `update_token_ids_ngram` | ~10+ PyTorch tensor ops | AscendC 单次 kernel |
| `ngram_match_extract` | ~20+ PyTorch tensor ops (unfold+argmax) | AscendC 单次 kernel |
| `copy_num_valid_draft_tokens` | `torch_npu.npu` 原生 stream | `torch_npu.npu` 原生 stream |

通过环境变量 `VLLM_NGRAM_USE_ASCENDC` 切换，`1` 为 AscendC（默认），`0` 回退 PyTorch。

## 环境准备

```bash
pkill -9 python
pkill -9 VLLM
source /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/set_env.sh
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/vllm-workspace/vllm:/disk1/lcb/vllm-ascend/vllm-ascend-0.12.0-1220_test/vllm-ascend/:${PYTHONPATH}
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/disk1/lcb/libjemalloc.so:$LD_PRELOAD
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
```

## 命令 A：启用 AscendC 融合算子（实验组）

```bash
export VLLM_NGRAM_USE_ASCENDC=1

vllm serve /disk1/lcb/qwen3-32B \
  --served-model-name qwen3 \
  --dtype bfloat16 \
  --max_model_len 23000 \
  --max-num-batched-tokens 40960 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable_expert_parallel \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8261 \
  --block-size 128 \
  --async-scheduling \
  --distributed_executor_backend "mp" \
  --speculative-model "[ngram]" \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 3 \
  --ngram-prompt-lookup-min 2 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,96,128],"enable_cpu_binding": "True"}'
```

## 命令 B：回退 PyTorch tensor ops（对照组）

```bash
export VLLM_NGRAM_USE_ASCENDC=0

vllm serve /disk1/lcb/qwen3-32B \
  --served-model-name qwen3 \
  --dtype bfloat16 \
  --max_model_len 23000 \
  --max-num-batched-tokens 40960 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable_expert_parallel \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8261 \
  --block-size 128 \
  --async-scheduling \
  --distributed_executor_backend "mp" \
  --speculative-model "[ngram]" \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 3 \
  --ngram-prompt-lookup-min 2 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,96,128],"enable_cpu_binding": "True"}'
```

## 压测命令

服务启动后，在另一个终端执行：

```bash
python benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --base-url http://localhost:8261 \
  --model qwen3 \
  --dataset-name sonnet --sonnet-input-len 512 --sonnet-output-len 256 \
  --num-prompts 200 --request-rate 10
```

## 对比指标

| 指标 | 含义 |
|------|------|
| median TTFT | 首 token 延迟 |
| median TPOT | 每 token 生成延迟 |
| throughput (tok/s) | 整体吞吐 |

## 微基准（单算子级别）

```bash
python benchmarks/bench_ngram_ops.py --batch_size 64 --max_seq_len 4096 --warmup 10 --repeat 100
```

## Ngram 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--speculative-model "[ngram]"` | - | 启用 ngram 投机解码 |
| `--num-speculative-tokens` | 5 | 每步猜测 token 数 |
| `--ngram-prompt-lookup-max` | 3 | 最大 n-gram 长度 |
| `--ngram-prompt-lookup-min` | 2 | 最小 n-gram 长度 |
| `VLLM_NGRAM_USE_ASCENDC` | 1/0 | 切换 AscendC / PyTorch |
