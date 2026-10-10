# Qwen3.8-27B(hybrid, head_dim=256)128k 实测与 hybrid 页对齐修复

> 日期:2026-10-10。环境:Atlas 800T A2,vllm-ascend 0.28.0rc(dspark 分支镜像),TP=2,enforce-eager,
> max_model_len=133120,store 模式(VLLM_ASCEND_KVTQ_STORE=1, mse_bits=4)。

## 一、head_dim=256 移植

- kernel/host 全部维度参数化:`HALF_ELEMS=8192` 维度不变,行数随 head_dim 反比;bits 2/3/4 ST 全 PASS(128 维回归 bit-exact)。
- glue 9 用例 + store roundtrip 双维度(128/256)全过。
- Qwen3.6-27B 冒烟:影子输出与基线逐 token 一致;store 写路径正常。

## 二、hybrid 页对齐问题与修复(容量收益的关键)

**问题**:Qwen3.5/3.6/3.8 为 hybrid 架构(3/4 GDN 线性注意力 + 1/4 全注意力)。vllm-ascend
`vllm_ascend/patch/platform/patch_mamba_config.py` 的 `verify_and_update_config`
(文件末尾赋给 `HybridAttentionMambaModelConfig`)要求 **attention 页 == mamba 页**,并用
`ModelConfig.get_head_size()`(真实 256)反推 attention block_size。它完全绕过
`Attention.get_kv_cache_spec` 的 head_size 收缩 hook,导致压缩省出的字节全部变成页内
padding:两侧 KV cache size 均为 449,649 tokens,容量收益为 0。注意这不是报错——服务正常、
推理正确,只是容量账没有变化(多留显存不会崩)。

**修复**(`code/integration/kvtq_store.py` 的 `_patch_hybrid_page_alignment`):
包装 `HybridAttentionMambaModelConfig.verify_and_update_config`(注意:不是 `MambaModelConfig`,
且必须先 import `vllm_ascend.patch.platform.patch_mamba_config` 让其末尾赋值先执行,再包装),
在页大小计算期间临时让 `ModelConfig.get_head_size()` 返回打包后的等效 head_size
(hybrid 256 维:行 pad 到 192 B/head,即 96 个 bf16 元素,满足 `h*kv_heads*2 | ssm_page`
的整除断言)。block_size 由 1536 变为 4096,attn 页恰好等于 mamba 页,无浪费。

**效果**(同机同配置,仅算子开关不同):

| 指标 | baseline | store(修复后) | 倍数 |
|---|---|---|---|
| GPU KV cache size | 449,649 tokens | 1,124,124 tokens | **2.50x** |
| 133,120 上下文最大并发 | 3.38x | 8.44x | **2.50x** |

hybrid 模型只有 1/4 层是全注意力,外加行 pad(130→192 B),故倍数低于纯 GQA 的 3.88x。

## 三、128k 输入 / 1k 输出 A/B(TTFT / TPOT)

base 输出 1k tokens 全量;tq 侧因 store v1 读路径慢做了截断(c1/c2 输出 128,c4+ 输出 32),
TPOT 取稳态均值,口径可比。

| 并发 | TTFT base | TTFT tq | ΔTTFT | TPOT base | TPOT tq | TPOT 倍数 |
|---|---|---|---|---|---|---|
| 1 | 42.1 s | 42.7 s | +1.4% | 207.6 ms | 1910.6 ms | 9.2x |
| 2 | 62.5 s | 64.8 s | +3.8% | 240.1 | 3811.3 | 15.9x |
| 4 | 159.8 s | 151.0 s | -5.5% | 266.4 | 6224.8 | 23.4x |
| 8 | 382.9 s | 343.0 s | -10.4% | 275.6 | 6762.1 | 24.5x |
| 16 | 843.0 s | 736.0 s | -12.7% | 282.5 | 7275.8 | 25.8x |
| 32 | 1776.7 s | 1528.1 s | -14.0% | 285.7 | 7414.5 | 26.0x |

**结论口径**:
- 写路径(prefill 量化)开销极小:TTFT ±1~4%,并发 ≥4 时 tq 因排队效应反而更快。
- 收益在**容量**(2.50x),不在时延。
- store v1 读路径(Python 分段 dequant + FIA)在 128k decode 是主要瓶颈,TPOT 随并发线性放大
  (~1.9 s × 并发);生产化需要把 dequant+attention 做成融合 kernel 或常驻 dequant 缓冲。

## 四、复现要点

```bash
# 服务(容器内),开启算子
VLLM_ASCEND_KVTQ_STORE=1 VLLM_ASCEND_KVTQ_BITS=4 \
ASCEND_RT_VISIBLE_DEVICES=4,5 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3.8-27B --served-model-name qwen38-27b \
  --tensor-parallel-size 2 --max-model-len 133120 \
  --max-num-batched-tokens 133120 --gpu-memory-utilization 0.92 \
  --no-enable-prefix-caching --enforce-eager --port 8377
# 启动日志应出现:
#   [KVTQ-STORE] hybrid page-alignment patch installed
#   patch_mamba_config.py: Setting attention block size to 4096 ...
#   GPU KV cache size: 1,124,124 tokens ... 8.44x
# 若 block size 仍为 1536 / KV size 449,649,说明页对齐 patch 未生效。

# 基准(容器内)
vllm bench serve --backend openai-chat --model /home/models/Qwen3.8-27B \
  --served-model-name qwen38-27b --host 127.0.0.1 --port 8377 \
  --endpoint /v1/chat/completions --dataset-name random \
  --random-input-len 131072 --random-output-len 1024 \
  --num-prompts <N> --max-concurrency <C> --ignore-eos --save-result ...
```

baseline 侧去掉两个 `VLLM_ASCEND_KVTQ_*` 环境变量即可,其余参数相同。