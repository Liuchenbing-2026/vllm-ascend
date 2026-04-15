# Ngram 投机解码 AscendC 算子优化方案

## 1. 背景与问题

### 1.1 Ngram 投机解码简介

Ngram 投机解码（Speculative Decoding）是一种无需额外草稿模型的加速策略：利用已有 token 历史中的 n-gram 模式来预测下一批 token，再由主模型一次性验证。这种方法不需要加载额外模型，只依赖历史 token 匹配。

### 1.2 性能瓶颈

在 vllm-ascend 异步调度模式下，ngram 草稿阶段有 3 个关键算子占总耗时 57%（3.58ms / 6.27ms），均为 PyTorch tensor ops 实现：

| 算子 | 原始耗时 | 问题 |
|------|---------|------|
| `update_token_ids_ngram` | 734us | ~10 次 PyTorch tensor ops，10+ 次 kernel launch |
| `ngram_match_extract` (含 `_find_first_and_extract_all_n_parallel`) | 1556us | ~20 次 PyTorch tensor ops（unfold、argmax、gather 等），20+ 次 kernel launch |
| `copy_num_valid_draft_tokens` | 1293us | `torch.cuda.Stream/Event` PTA 封装层开销 |

**根因**：每个 PyTorch tensor op 独立发起 NPU kernel launch，launch 开销（主机端排队、同步）在小 tensor 场景下占主导。且 PyTorch 动态 tensor ops 无法被 aclgraph（图模式）捕获。

### 1.3 优化目标

1. 将算子 1、2 融合为 AscendC 单 kernel，消除 kernel launch 开销
2. 算子 3 用 `torch_npu.npu` 原生 stream API 替换 `torch.cuda` PTA 封装
3. 注册 meta kernel，使算子支持 aclgraph 图模式捕获

## 2. 调用链路分析

### 2.1 整体调用链

```
model_runner_v1.py: execute_model()
  └─ spec_decode_worker: generate_draft_tokens()
       ├─ update_token_ids_ngram()     ← 算子 1
       ├─ propose()                     ← 算子 2 (内部调用 ngram_match_extract)
       └─ copy_num_valid_draft_tokens() ← 算子 3
```

### 2.2 调用入口

- `model_runner_v1.py` 导入 `copy_num_valid_draft_tokens_npu`（L78）
- `AscendNgramProposerNPU`（`ngram_proposer_npu.py`）继承 `NgramProposerGPU`，override `update_token_ids_ngram()` 和 `propose()`

### 2.3 原始 Python 算法概要

**算子 1: `update_token_ids_ngram`**（`ngram_proposer_gpu.py:385-458`）

```
输入: sampled_token_ids[B, max_new], token_ids_gpu[B, max_len],
      num_tokens_no_spec[B], discard_mask[B], vocab_size
逻辑:
  1. backup = token_ids_gpu[r, max(0, num_tokens_no_spec[r] - 1)]  // 备份上一个有效 token
  2. 对 discarded request 将 sampled_token_ids 全部置 -1
  3. valid_mask = (token != -1) & (token < vocab_size)
  4. valid_count = valid_mask.sum(dim=1)
  5. next_token = valid_sampled[last_valid_idx] if count > 0 else backup
输出: next_token_ids[B], valid_count[B], valid_sampled_token_ids[B, max_new]
```

**算子 2: `_find_first_and_extract_all_n_parallel` + valid count**（`ngram_proposer_gpu.py:46-158`）

```
输入: token_ids[B, max_len], seq_lengths[B], combined_mask[B],
      min_n, max_n, k
逻辑:
  1. 对每个 n ∈ [min_n, max_n]:
     a. suffix = token_ids 的最后 n 个 token
     b. 用 unfold 创建滑动窗口，与 suffix 逐一比较
     c. argmax 找最早匹配位置
  2. 选最长匹配的 n-gram（优先 max_n）
  3. 从匹配位置 + n 处提取 k 个 draft token
  4. 统计前导连续有效 token 数
输出: draft_tokens[B, k], num_valid_draft_tokens[B]
```

**算子 3: `copy_num_valid_draft_tokens`**（`ngram_proposer_gpu.py:637-661`）

```
原始: torch.cuda.Stream + torch.cuda.Event 异步 D2H 拷贝
问题: torch.cuda 在 NPU 上走 PTA 兼容层，额外封装开销
优化: 直接用 torch_npu.npu.Stream + torch_npu.npu.Event
```

## 3. 修改方案

### 3.1 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│  Python 层 (ngram_proposer_npu.py)                          │
│  ┌──────────────────────┐  ┌──────────────────────────────┐  │
│  │ update_token_ids_ngram│  │ propose()                    │  │
│  │   ↓ torch.ops._C_ascend│  │   ↓ scatter (PyTorch)       │  │
│  │   .npu_update_token_  │  │   ↓ torch.ops._C_ascend     │  │
│  │    ids_ngram()        │  │   .npu_ngram_match_extract() │  │
│  └──────────┬───────────┘  └──────────────┬───────────────┘  │
│             │                              │                  │
│  ┌──────────┴──────────────────────────────┴───────────────┐  │
│  │  copy_num_valid_draft_tokens_npu()                      │  │
│  │  torch_npu.npu.Stream / Event                           │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐    ┌──────────────────────────┐
│  torch_binding.cpp  │    │  torch_binding_meta.cpp   │
│  EXEC_NPU_CMD(      │    │  Meta kernels (shape-only │
│    aclnn*...)       │    │  inference for graph mode) │
│  TORCH_LIBRARY_EXPAND│    │  TORCH_LIBRARY_IMPL_EXPAND│
└─────────┬───────────┘    └───────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────┐
│  AscendC Kernels (csrc/)                      │
│  ┌─────────────────────┐ ┌──────────────────┐ │
│  │ UpdateTokenIdsNgram │ │ NgramMatchExtract│ │
│  │  op_host/ (tiling,  │ │  op_host/ (tiling,│ │
│  │   def, infershape)  │ │   def, infershape)│ │
│  │  op_kernel/ (.cpp)  │ │  op_kernel/ (.cpp)│ │
│  └─────────────────────┘ └──────────────────┘ │
└───────────────────────────────────────────────┘
```

### 3.2 文件变更清单

#### 新增文件（12 个）

**算子 1: UpdateTokenIdsNgram**

```
csrc/update_token_ids_ngram/
├── op_host/
│   ├── CMakeLists.txt                          # 构建配置
│   ├── update_token_ids_ngram_def.cpp          # OP_ADD 注册（输入/输出/属性）
│   ├── update_token_ids_ngram_infershape.cpp   # IMPL_OP_INFERSHAPE（输出 shape 推断）
│   ├── update_token_ids_ngram_tiling.cpp       # IMPL_OP_OPTILING（Tiling 参数计算）
│   └── update_token_ids_ngram_tiling.h         # TilingData 定义
└── op_kernel/
    └── update_token_ids_ngram.cpp              # AscendC kernel 实现
```

**算子 2: NgramMatchExtract**

```
csrc/ngram_match_extract/
├── op_host/
│   ├── CMakeLists.txt
│   ├── ngram_match_extract_def.cpp
│   ├── ngram_match_extract_infershape.cpp
│   ├── ngram_match_extract_tiling.cpp
│   └── ngram_match_extract_tiling.h
└── op_kernel/
    └── ngram_match_extract.cpp
```

#### 修改文件（4 个）

| 文件 | 修改内容 |
|------|---------|
| `csrc/torch_binding.cpp` | 新增 2 个 C++ wrapper 函数 + `EXEC_NPU_CMD` 调用 + `TORCH_LIBRARY_EXPAND` 注册 |
| `csrc/torch_binding_meta.cpp` | 新增 2 个 meta kernel（shape 推断）+ `TORCH_LIBRARY_IMPL_EXPAND` 注册 |
| `vllm_ascend/spec_decode/ngram_proposer_npu.py` | override `update_token_ids_ngram` / `propose`，调用 AscendC 算子；新增 `copy_num_valid_draft_tokens_npu` |
| `vllm_ascend/worker/model_runner_v1.py` | import 从 `vllm` 切换到 `vllm_ascend` 的 `copy_num_valid_draft_tokens_npu` |

### 3.3 算子 1: UpdateTokenIdsNgram 详细设计

#### TilingData

```cpp
BEGIN_TILING_DATA_DEF(UpdateTokenIdsNgramTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);    // 实际使用核数
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);         // batch size
    TILING_DATA_FIELD_DEF(uint32_t, reqsPerCore);     // 每核处理请求数
    TILING_DATA_FIELD_DEF(uint32_t, remainderReqs);   // 余数核多处理 1 个
    TILING_DATA_FIELD_DEF(uint32_t, maxNewTokens);    // sampled_token_ids 列数
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);       // token_ids_gpu 列数
    TILING_DATA_FIELD_DEF(int32_t, vocabSize);        // 词表大小
END_TILING_DATA_DEF;
```

#### Kernel 算法

```
对每个 AI Core（coreId = GetBlockIdx()）：
  计算 myStartReq, myNumReqs（均分 + 余数策略）
  预加载本核 metadata: num_tokens_no_spec, discard_mask → UB

  FOR each request r in [myStartReq, myStartReq + myNumReqs):
    1. GM → UB: backup_token = token_ids_gpu[r, max(0, numTok-1)]
    2. GM → UB: sampled_tokens[0..maxNewTokens)
    3. 读 discard_mask[r]
    4. 标量循环: 若 discarded 则置 -1; 统计 valid_count; 记录 lastValidToken
    5. next_token = (validCount > 0) ? lastValidToken : backupToken
    6. UB → GM: next_token_ids[r], valid_count[r], valid_sampled[r,:]
```

#### UB 内存分配

| Buffer | 大小 | 用途 |
|--------|------|------|
| numTokBuf_ | myNumReqs * 4B, 32B 对齐 | 预加载 num_tokens_no_spec |
| discardBuf_ | myNumReqs * 1B, 32B 对齐 | 预加载 discard_mask |
| sampledBuf_ | maxNewTokens * 4B, 32B 对齐 | 当前请求的 sampled tokens |
| backupBuf_ | 32B | 单个 backup token |
| outSampledBuf_ | maxNewTokens * 4B, 32B 对齐 | 输出 valid_sampled |
| scalarBuf_ | 32B | 输出标量（next_token, valid_count）|

### 3.4 算子 2: NgramMatchExtract 详细设计

#### TilingData

```cpp
BEGIN_TILING_DATA_DEF(NgramMatchExtractTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, reqsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderReqs);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);       // token_ids 列数
    TILING_DATA_FIELD_DEF(uint32_t, minN);             // 最小 n-gram 长度
    TILING_DATA_FIELD_DEF(uint32_t, maxN);             // 最大 n-gram 长度
    TILING_DATA_FIELD_DEF(uint32_t, k);                // draft token 数
END_TILING_DATA_DEF;
```

#### Kernel 算法（分块扫描）

序列可能长达数万 token，无法一次放入 UB（192KB），因此采用分块扫描策略：

```
常量: CHUNK_SIZE = 4096, MAX_N_CAP = 16, MAX_K_CAP = 32
      SCAN_BUF_ELEMS = CHUNK_SIZE + MAX_N_CAP = 4112

对每个 request r：
  1. 检查 combined_mask[r], 若 false → 输出全 -1, 跳过
  2. 检查 seqLen, 若 < minN → 输出全 -1, 跳过
  3. GM → UB: suffix = token_ids[r, seqLen-maxN : seqLen]

  4. FOR n = maxN downto minN:          // 优先最长匹配
       maxSearchPos = seqLen - n - 1
       FOR chunkStart = 0, step CHUNK_SIZE:
         loadEnd = min(chunkStart + CHUNK_SIZE + n - 1, seqLen)
         GM → UB: scanBuf[0..loadEnd-chunkStart)  // 带 n-1 overlap
         FOR pos = chunkStart to min(chunkStart+CHUNK_SIZE, maxSearchPos+1):
           比较 scanBuf[pos-chunkStart : +n] vs suffix
           匹配 → bestMatchPos = pos, bestN = n, break all
       IF found → break

  5. 若无匹配 → 输出全 -1
  6. draftStart = bestMatchPos + bestN
     GM → UB: draft_tokens[0..min(k, seqLen-draftStart))
     填充 -1 到 k
  7. 统计前导连续有效 token 数 (numValid)
  8. UB → GM: draft_tokens[r,:], num_valid_draft_tokens[r]
```

#### UB 内存分配

| Buffer | 大小 | 用途 |
|--------|------|------|
| seqLenBuf_ | myNumReqs * 4B, 32B 对齐 | 预加载 seq_lengths |
| maskBuf_ | myNumReqs * 1B, 32B 对齐 | 预加载 combined_mask |
| suffixBuf_ | MAX_N_CAP * 4B = 64B, 32B 对齐 | 后缀 n-gram |
| scanBuf_ | SCAN_BUF_ELEMS * 4B = 16448B, 32B 对齐 | 分块扫描缓冲（含 overlap）|
| draftBuf_ | MAX_K_CAP * 4B = 128B, 32B 对齐 | 输出 draft tokens |
| scalarBuf_ | 32B | 输出 num_valid |

**总 UB 占用** ≈ 17KB（远小于 192KB 限制），即使 myNumReqs 较大（metadata 缓冲）也不会溢出。

### 3.5 算子 3: copy_num_valid_draft_tokens NPU Stream 优化

无需 AscendC kernel，仅替换 Python 层 stream API：

| 项目 | 原始 | 优化后 |
|------|------|--------|
| Stream 创建 | `torch.cuda.Stream()` | `torch_npu.npu.Stream()` |
| Event 创建 | `torch.cuda.Event()` | `torch_npu.npu.Event()` |
| 当前 stream | `torch.cuda.current_stream()` | `torch_npu.npu.current_stream()` |
| context manager | `torch.cuda.stream(s)` | `torch_npu.npu.stream(s)` |

### 3.6 Torch 注册

#### torch_binding.cpp（PrivateUse1 dispatch key）

```cpp
// 函数签名注册
ops.def(
    "npu_update_token_ids_ngram(Tensor sampled_token_ids, Tensor token_ids_gpu, "
    "Tensor num_tokens_no_spec, Tensor discard_mask, int vocab_size) -> "
    "(Tensor next_token_ids, Tensor valid_count, Tensor valid_sampled_token_ids)"
);
ops.impl("npu_update_token_ids_ngram", torch::kPrivateUse1,
         &vllm_ascend::npu_update_token_ids_ngram);

ops.def(
    "npu_ngram_match_extract(Tensor token_ids, Tensor seq_lengths, "
    "Tensor combined_mask, int min_n, int max_n, int k) -> "
    "(Tensor draft_tokens, Tensor num_valid_draft_tokens)"
);
ops.impl("npu_ngram_match_extract", torch::kPrivateUse1,
         &vllm_ascend::npu_ngram_match_extract);
```

C++ wrapper 内部调用 `EXEC_NPU_CMD(aclnn*, ...)` 发起算子执行。

#### torch_binding_meta.cpp（Meta dispatch key, 图模式支持）

```cpp
// Meta kernel 只做 shape 推断，不实际计算
ops.impl("npu_update_token_ids_ngram",
         &vllm_ascend::meta::npu_update_token_ids_ngram_meta);
ops.impl("npu_ngram_match_extract",
         &vllm_ascend::meta::npu_ngram_match_extract_meta);
```

注册 Meta dispatch key 后，`torch.compile` / aclgraph 在 tracing 阶段可以推断输出 shape 而无需实际执行算子。

### 3.7 Python 集成层

`ngram_proposer_npu.py` 中 `AscendNgramProposerNPU` 继承 `NgramProposerGPU`，override 两个方法：

```python
# update_token_ids_ngram: 直接调用 AscendC 算子
next_token_ids, valid_count, valid_sampled = (
    torch.ops._C_ascend.npu_update_token_ids_ngram(
        sampled_token_ids,
        token_ids_gpu[:num_reqs],
        num_tokens_no_spec[:num_reqs],
        discard_request_mask[:num_reqs].to(torch.int8),  # bool → int8
        gpu_input_batch.vocab_size,
    )
)

# propose: scatter 保持 PyTorch 实现，ngram 匹配调用 AscendC 算子
draft_tokens, num_valid_draft_tokens = (
    torch.ops._C_ascend.npu_ngram_match_extract(
        token_ids_gpu,
        num_tokens_tmp.to(torch.int32),
        combined_mask.to(torch.int8),  # bool → int8
        self.min_n, self.max_n, self.k,
    )
)
```

**类型转换注意**：`discard_mask` 和 `combined_mask` 在 Python 层为 `torch.bool`，AscendC kernel 接收 `INT8`，在调用前用 `.to(torch.int8)` 转换。

### 3.8 A/B 开关

通过环境变量 `VLLM_NGRAM_USE_ASCENDC` 控制：

```python
_USE_ASCENDC = os.environ.get("VLLM_NGRAM_USE_ASCENDC", "1") != "0"
```

- `VLLM_NGRAM_USE_ASCENDC=1`（默认）：使用 AscendC 融合算子
- `VLLM_NGRAM_USE_ASCENDC=0`：回退到父类 `NgramProposerGPU` 的 PyTorch tensor ops

### 3.9 图模式说明

当前 ngram proposer 运行在异步调度路径（`spec_decode_worker`），**不在** aclgraph 捕获范围内，以 eager 模式执行。但：

1. Meta kernel 已注册，算子具备图模式捕获能力
2. 融合后单次 kernel launch 已消除主要 launch 开销
3. 未来若 proposer 路径支持 aclgraph 捕获，无需额外改动

## 4. 性能对比方法

### 4.1 环境准备

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

### 4.2 命令 A：启用 AscendC 融合算子（实验组）

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

### 4.3 命令 B：回退 PyTorch tensor ops（对照组）

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

### 4.4 压测命令

服务启动后，在另一个终端执行：

```bash
python benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --base-url http://localhost:8261 \
  --model qwen3 \
  --dataset-name sonnet --sonnet-input-len 512 --sonnet-output-len 256 \
  --num-prompts 200 --request-rate 10
```

### 4.5 对比指标

| 指标 | 含义 |
|------|------|
| median TTFT | 首 token 延迟（Time To First Token）|
| median TPOT | 每 token 生成延迟（Time Per Output Token）|
| throughput (tok/s) | 整体吞吐量 |

### 4.6 微基准（单算子级别）

```bash
python benchmarks/bench_ngram_ops.py --batch_size 64 --max_seq_len 4096 --warmup 10 --repeat 100
```

报告每个算子的 median / mean / p90 延迟（微秒），分别对比 PyTorch 和 AscendC 实现。

## 5. Ngram 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--speculative-model "[ngram]"` | - | 启用 ngram 投机解码 |
| `--num-speculative-tokens` | 5 | 每步猜测 token 数（k）|
| `--ngram-prompt-lookup-max` | 3 | 最大 n-gram 长度（max_n）|
| `--ngram-prompt-lookup-min` | 2 | 最小 n-gram 长度（min_n）|
| `VLLM_NGRAM_USE_ASCENDC` | 1/0 | 切换 AscendC（默认）/ PyTorch tensor ops |

## 6. 优化效果预期

| 算子 | 原始 (PyTorch) | 优化后 (AscendC) | 加速比 |
|------|---------------|-----------------|--------|
| update_token_ids_ngram | ~734us (10+ launches) | ~50-150us (1 launch) | 5-15x |
| ngram_match_extract | ~1556us (20+ launches) | ~100-300us (1 launch) | 5-15x |
| copy_num_valid_draft_tokens | ~1293us (PTA 封装) | ~200-400us (原生 API) | 3-6x |
| **草稿阶段总计** | **~3583us** | **~350-850us** | **4-10x** |
