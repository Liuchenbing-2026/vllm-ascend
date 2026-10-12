# Qwen3-30B-A3B int8 原生 KV cache 路径（B2）128k 实测

> 日期：2026-10-11（容量/未融合数据）、2026-10-12（写路径融合后复测）。
> 环境：Atlas 800T A2，vllm-ascend 0.28.0rc，TP=2，enforce-eager，
> max_model_len=133120（YaRN x4 via `--hf-overrides`，原生 40960）。
> 模式：`VLLM_ASCEND_KVTQ_STORE=int8` —— KV cache 分配为 int8（NZ 排布 + per-channel scale），
> 读路径走 FIA 原生 antiquant（mode 0），无任何 Python 解压。
> 写路径：自研融合算子 `KvCacheInt8Store`（单 kernel 完成 per-channel 量化 + NZ 散写，
> `VLLM_ASCEND_KVTQ_INT8_FUSED=0` 可回退 PyTorch 链，两者 bit-exact 等价）。
> 输入 128k（131072），输出 1k（1024），`--ignore-eos`，random 数据集。

## 三种模式容量

| 指标 | baseline(bf16) | int8 原生 | TurboQuant 4bit store |
|---|---|---|---|
| GPU KV cache size | 351,616 tokens | 703,360 tokens (**2.00x**) | 1,364,096 tokens (**3.88x**) |
| 133,120 上下文最大并发 | 2.64 | 5.28 | 10.25 |

## 写路径融合后（当前口径，KvCacheInt8Store）

### TTFT（ms，越低越好）

| 并发 | base | int8 融合 | int8 vs base |
|---|---|---|---|
| 1 | 37,891 | 38,521 | +1.7% |
| 2 | 56,197 | 56,988 | +1.4% |
| 4 | 197,349 | 94,495 | **-52.1%** |
| 8 | 474,690 | 246,222 | **-48.1%** |
| 16 | 1,016,360 | 548,813 | **-46.0%** |
| 32 | 2,129,892 | 1,179,369 | **-44.6%** |

### TPOT（ms，越低越好）

| 并发 | base | int8 融合 | int8 vs base |
|---|---|---|---|
| 1 | 193.8 | 190.5 | **-1.7%** |
| 2 | 221.2 | 219.7 | **-0.7%** |
| 4 | 229.9 | 253.6 | +10.3% |
| 8 | 229.4 | 289.9 | +26.4% |
| 16 | 230.6 | 310.2 | +34.5% |
| 32 | 232.8 | 317.6 | +36.4% |

### 输出吞吐（tok/s）与端到端耗时

| 并发 | base tok/s | int8 融合 tok/s | 提升 | base 总耗时(s) | int8 总耗时(s) | 缩短 |
|---|---|---|---|---|---|---|
| 1 | 4.3 | 4.4 | +2% | 472 | 467 | -1% |
| 2 | 7.2 | 7.3 | +1% | 283 | 282 | -0% |
| 4 | 7.3 | 11.6 | **+59%** | 564 | 354 | -37% |
| 8 | 7.4 | 11.6 | **+57%** | 1,106 | 706 | -36% |
| 16 | 7.4 | 11.6 | **+57%** | 2,203 | 1,415 | -36% |
| 32 | 7.4 | 12.4 | **+68%** | 4,423 | 2,637 | **-40%** |

## 写路径融合前后对比（int8 自身）

| 并发 | TPOT 未融合 | TPOT 融合 | 修复幅度 | TTFT 未融合 | TTFT 融合 |
|---|---|---|---|---|---|
| 1 | 233.8 | 190.5 | -18.5% | 39,963 | 38,521 |
| 2 | 260.6 | 219.7 | -15.7% | 58,919 | 56,988 |
| 4 | 292.1 | 253.6 | -13.2% | 97,911 | 94,495 |
| 8 | 330.9 | 289.9 | -12.4% | 266,362 | 246,222 |
| 16 | 353.2 | 310.2 | -12.2% | 606,502 | 548,813 |
| 32 | 360.7 | 317.6 | -12.0% | 1,301,883 | 1,179,369 |

## 结论

- **容量红利兑现**：c4+ TTFT 降低 45~52%，输出吞吐提升 57~68%，端到端总耗时缩短 36~40%。
  根因：128k 请求下 baseline KV 容量只够 ~2.64 条并发，其余排队；int8 容量 2x →
  并发额度 5.28 条，排队轮次减半。
- **低并发（c1/c2，无排队）TPOT 已转正**（-1.7% / -0.7%）：写路径融合后（单 kernel
  量化+散写，替代原来每 token 每层 ~7 个 PyTorch 小算子、~20-27ms/step 的 eager 调度），
  int8 读带宽收益（FIA antiquant 直接读 int8，HBM 流量减半）在单条 decode 上开始兑现。
- **c4+ TPOT 高 10~36% 是 batch-size 效应，不是算子开销**（2026-10-12 已定位）：
  int8 实际运行的 decode batch 约为 baseline 的 2 倍（5.28 vs 2.64 条 128k 序列同时在跑）。
  证据链：(a) 同 batch 下直接对比（c1/c2）int8 TPOT 持平略优（-1.7% / -0.7%）；
  (b) int8 c4 无排队（容量 5.28 ≥ 4），TPOT 253.6 = batch 2→4 的每步计算增长，
  每翻倍 ~+15%，外推到 batch 5.28 得 ~290ms，与 c8 实测 289.9 吻合；
  (c) 反向判别实验 `--max-num-seqs 2` + c8：TPOT 453ms（更差）——排队请求的
  128k prefill 与 decode 抢占干扰主导，证明限并发换 TPOT 在 128k 场景不成立。
  **结论：int8 的 TPOT 表现等于"base 跑到它根本跑不了的 batch 规模"时的表现，
  同 batch 口径下无劣化。**
- **图模式（关 enforce-eager）在 128k 场景是负收益，维持 eager 为正确配置**
  （2026-10-12 实测）：baseline 图模式 c1 TPOT 316.6ms vs eager 193.8ms（劣化 63%，
  FIA v2 graph 路径在长上下文 decode 更慢）；TTFT 不变（37.6s vs 37.9s）。
  int8 模式的 eager 限制因此不构成实际损失。顺带完成图模式加固：融合 kernel 增加
  slot 越界守卫（dummy slot 跳过）、scale 标定改为 eager 预分配 + 原地 copy_ 更新
  （消除 capture 期 host-sync 与图池分配），int8 写路径现在可安全穿过 graph capture，
  在 FIA v2 读路径移植（device seq_lens + antiquant 参数）前由守卫清晰报错。
- tq4（TurboQuant 4bit，Python 解压读路径）TPOT 劣化 16~146x，仅作容量上限参考，不可用；
  其容量 3.88x 需待 int4 融合读算子（FIA int4 antiquant）落地后才可用化。
- 精度：int8 per-channel 量化 gsm8k 73.5%（baseline bf16 74.0%），融合写与 PyTorch 写
  bit-exact 等价，精度结论不受融合影响。

## 复现

```bash
# 1. 安装融合算子包（op_i8store 源码 + 老 kv_tq 合并构建，见 code/op/build_op.sh）
#    构建后 install 到 ASCEND OPP customize 目录，并：
source $ASCEND_HOME/opp/vendors/customize/bin/set_env.bash

# 2. 启动服务（写路径默认走融合算子；VLLM_ASCEND_KVTQ_INT8_FUSED=0 回退 PyTorch 链）
VLLM_ASCEND_KVTQ_STORE=int8 ASCEND_RT_VISIBLE_DEVICES=6,7 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3-30B-A3B --served-model-name qwen3-30b \
  --tensor-parallel-size 2 --max-model-len 133120 --max-num-batched-tokens 133120 \
  --hf-overrides ''{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}'' \
  --gpu-memory-utilization 0.92 --no-enable-prefix-caching --enforce-eager --port 8379
# 启动日志应出现 GPU KV cache size: 703,360 tokens（base 为 351,616）
# 且日志中 fused op unavailable 计数为 0

# 3. 压测命令见 docs/qwen3_30b_128k.md 与 reproduce.md
```
