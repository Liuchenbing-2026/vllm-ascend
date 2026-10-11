# Qwen3-30B-A3B int8 原生 KV cache 路径（B2）128k 实测

> 日期：2026-10-11。环境：Atlas 800T A2，vllm-ascend 0.28.0rc，TP=2，enforce-eager，
> max_model_len=133120（YaRN x4 via `--hf-overrides`，原生 40960）。
> 模式：`VLLM_ASCEND_KVTQ_STORE=int8` —— KV cache 分配为 int8（NZ 排布 + per-channel scale），
> 写路径走自研算子散写，读路径走 FIA 原生 antiquant（mode 0），无任何 Python 解压。
> 输入 128k（131072），输出 1k（1024），`--ignore-eos`，random 数据集。

## 三种模式容量

| 指标 | baseline(bf16) | int8 原生 | TurboQuant 4bit store |
|---|---|---|---|
| GPU KV cache size | 351,616 tokens | 703,360 tokens (**2.00x**) | 1,364,096 tokens (**3.88x**) |
| 133,120 上下文最大并发 | 2.64 | 5.28 | 10.25 |

## TTFT（ms，越低越好）

| 并发 | base | int8 | tq4 | int8 vs base |
|---|---|---|---|---|
| 1 | 37,891 | 39,963 | 38,693 | +5.5% |
| 2 | 56,197 | 58,919 | 60,387 | +4.8% |
| 4 | 197,349 | 97,911 | 107,438 | **-50.4%** |
| 8 | 474,690 | 266,362 | 211,800 | **-43.9%** |
| 16 | 1,016,360 | 606,502 | 709,752 | **-40.3%** |
| 32 | 2,129,892 | 1,301,883 | 1,652,732 | **-38.9%** |

## TPOT（ms，越低越好）

| 并发 | base | int8 | tq4 | int8 vs base |
|---|---|---|---|---|
| 1 | 193.8 | 233.8 | 3,127.1 | +20.6% |
| 2 | 221.2 | 260.6 | 6,351.3 | +17.8% |
| 4 | 229.9 | 292.1 | 12,927.1 | +27.0% |
| 8 | 229.4 | 330.9 | 27,135.4 | +44.2% |
| 16 | 230.6 | 353.2 | 31,157.8 | +53.2% |
| 32 | 232.8 | 360.7 | 33,895.6 | +54.9% |

## 输出吞吐（tok/s）

| 并发 | base | int8 | 提升 |
|---|---|---|---|
| 4 | 7.3 | 10.3 | **+41%** |
| 8 | 7.4 | 10.3 | **+39%** |
| 16 | 7.4 | 10.3 | **+39%** |
| 32 | 7.4 | 11.1 | **+50%** |

## 结论

- **TTFT / 排队时延全面大幅改善**（c4+ 降低 39~50%）：容量 2x 意味着同样 32 条 128k 请求，
  baseline 要排 12 轮队，int8 只排 6 轮。这是容量红利的直接体现。
- **TPOT 劣化 18~55%**：写路径目前是每 token 每层约 7 个小算子的 eager 调度（~20-27ms/step），
  吃掉了 int8 读带宽收益（kernel 微基准 128k 单层读提速 c8 2.61x）。低并发读收益本来就小，
  净效果为负；**修复路径：写路径融合成单个 NPU kernel（自研算子加 int8 输出模式），
  预计写开销 <2ms，低并发 TPOT 转正，高并发读收益兑现**。
- tq4（Python 解压读路径）TPOT 劣化 16~146x，仅作容量上限参考，不可用。
- int8 路径容量 2.00x < tq4 的 3.88x，但时延可用性完全不同量级。

## 复现

```bash
VLLM_ASCEND_KVTQ_STORE=int8 ASCEND_RT_VISIBLE_DEVICES=6,7 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3-30B-A3B --served-model-name qwen3-30b \
  --tensor-parallel-size 2 --max-model-len 133120 --max-num-batched-tokens 133120 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}' \
  --gpu-memory-utilization 0.92 --no-enable-prefix-caching --enforce-eager --port 8379
# 启动日志应出现 GPU KV cache size: 703,360 tokens（base 为 351,616）
# 压测命令见 docs/qwen3_30b_128k.md 与 reproduce.md
```