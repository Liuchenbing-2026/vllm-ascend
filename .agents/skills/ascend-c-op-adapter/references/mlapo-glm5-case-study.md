# Case Study: MLAPO 算子适配 GLM5 模型

## 背景

MLA Preprocess (MLAPO) 是 vllm-ascend 中最复杂的 AscendC 融合算子，将 MLA attention 的 decode 路径中 Q/KV 投影、RMSNorm、RoPE、量化、EinSum 等操作融合为单个 kernel launch，大幅减少 HBM 访问和 kernel launch 开销。

MLAPO 最初为 DeepSeek V3 硬编码了 MLA 维度常量。适配 GLM5 需要将这些常量参数化为运行时从 tensor shape 推导的值。

## 模型维度对比

| 参数 | DeepSeek V3 | GLM5 | 差异 |
|------|------------|------|------|
| q_lora_rank | 1536 | 2048 | +512 |
| qk_nope_head_dim | 128 | 192 | +64 |
| qk_rope_head_dim | 64 | 64 | 相同 |
| kv_lora_rank | 512 | 512 | 相同 |
| v_head_dim | 128 | 256 | +128 |
| num_heads | 128 | 64 | -64 |

关键影响的派生值：

| 派生值 | 公式 | DSV3 | GLM5 |
|--------|------|------|------|
| mm1OutSize | q_lora_rank + kv_lora_rank + qk_rope_head_dim | 2112 | 2624 |
| splitSizeTwo | q_lora_rank | 1536 | 2048 |
| hiddenStrideRope | qk_nope_head_dim + qk_rope_head_dim | 192 | 256 |
| avgFactor | 1.0 / q_lora_rank | 0.000651 | 0.000488 |

## 适配步骤详解

### Step 1: 识别所有硬编码位置

**搜索命令**：
```bash
grep -rn "2112\|1536\b" csrc/mla_preprocess/
grep -rn "128\b" csrc/mla_preprocess/op_kernel/*.hpp  # 注意区分 qk_nope_head_dim vs 其他 128
```

**发现的硬编码位置**：

1. **`op_host/tiling/mla_preprocess_tiling.h`** — 无（最终要在这里新增字段）
2. **`op_host/mla_preprocess.h`** (Tiling 计算)
   - `mm1OutSize = 2112` (多处)
   - `splitSizeOne = 576`, `splitSizeTwo = 1536`
   - `splitRmsNormSizeOne = 512`, `splitRmsNormSizeTwo = 64`
   - `ropeSplitSizeOne = 64`, `ropeSplitSizeTwo = 128`
   - `hiddenStrideRope = 192`
   - `avgFactor = 0.000651041666f`
3. **`op_kernel/*.hpp`** (4 个 kernel 变体)
   - `2112`, `1536`, `576`, `128` 等分散在计算逻辑中
   - 辅助类 `Quant`, `RmsNormQuant` 中也有硬编码

### Step 2: 扩展 MlaTilingData 结构体

文件: `csrc/mla_preprocess/op_host/tiling/mla_preprocess_tiling.h`

新增的字段及其语义：

```cpp
struct MlaTilingData {
    // ... 已有字段保持不变 ...

    // Model-specific MLA dimensions (derived from tensor shapes)
    uint32_t mm1OutSize{2112};        // q_lora_rank + kv_lora_rank + qk_rope_head_dim
    uint32_t splitSizeOne{576};        // kv_lora_rank + qk_rope_head_dim
    uint32_t splitSizeTwo{1536};       // q_lora_rank
    uint32_t splitRmsNormSizeOne{512}; // kv_lora_rank
    uint32_t splitRmsNormSizeTwo{64};  // qk_rope_head_dim
    uint32_t ropeSplitSizeOne{64};     // qk_rope_head_dim
    uint32_t ropeSplitSizeTwo{128};    // qk_nope_head_dim
    uint32_t hiddenStrideRope{192};    // qk_nope_head_dim + qk_rope_head_dim
    uint32_t qkNopeHeadDim{128};       // for RoPE offset calculation
    float avgFactor{0.000651041666f};  // 1/splitSizeTwo (1/qLoraRank)
};
```

**设计要点**：
- 所有新字段的默认值 = DeepSeek V3 的值 → 不传新参数时行为不变
- 字段名与计算公式一一对应 → 方便 kernel 代码替换
- `avgFactor` 是 `float` 而非 `uint32_t` → RMSNorm 均值因子

### Step 3: 从 tensor shape 推导维度

文件: `csrc/mla_preprocess/op_host/mla_preprocess.h`

在 `MlaPreprocessTiling::Init()` 中新增推导逻辑：

```cpp
// 推导 OpParam 中的 MLA 维度
OpParam opParam;
opParam.qkNopeHeadDim = wuk.sizes()[1];           // wuk: [headNum, qkNopeHeadDim, kvLoraRank]
opParam.kvLoraRank = wuk.sizes()[2];
opParam.qLoraRank = gamma1.sizes()[0];             // gamma1: [qLoraRank]
opParam.qkRopeHeadDim = kv_cache_rope.sizes().back(); // kv_cache_rope: [..., qkRopeHeadDim]

// 填充 TilingData
tilingData->mm1OutSize = opParam.qLoraRank + opParam.kvLoraRank + opParam.qkRopeHeadDim;
tilingData->splitSizeOne = opParam.kvLoraRank + opParam.qkRopeHeadDim;
tilingData->splitSizeTwo = opParam.qLoraRank;
tilingData->splitRmsNormSizeOne = opParam.kvLoraRank;
tilingData->splitRmsNormSizeTwo = opParam.qkRopeHeadDim;
tilingData->ropeSplitSizeOne = opParam.qkRopeHeadDim;
tilingData->ropeSplitSizeTwo = opParam.qkNopeHeadDim;
tilingData->hiddenStrideRope = opParam.qkNopeHeadDim + opParam.qkRopeHeadDim;
tilingData->qkNopeHeadDim = opParam.qkNopeHeadDim;
tilingData->avgFactor = 1.0f / static_cast<float>(opParam.qLoraRank);
```

**Tensor shape 到维度的映射关系**：

| Tensor | Shape | 维度推导 |
|--------|-------|---------|
| `wuk` (W_UK 转置) | `[headNum, qkNopeHeadDim, kvLoraRank]` | `sizes()[1]` → qkNopeHeadDim, `sizes()[2]` → kvLoraRank |
| `gamma1` (Q LayerNorm weight) | `[qLoraRank]` | `sizes()[0]` → qLoraRank |
| `kv_cache_rope` | `[..., qkRopeHeadDim]` | `sizes().back()` → qkRopeHeadDim |
| `wdqkv` | `[hiddenStateDim, mm1OutSize]` | `sizes()[0]` → hiddenStateDim |

### Step 4: Torch 适配层透传

文件: `csrc/mla_preprocess/mla_preprocess_torch_adpt.h`

Tiling 函数签名更新以接收推导所需 tensor：

```cpp
auto [workspace_tensor, tiling, block_dim] = mlapo::mla_preprocess_tiling(
    hiddenState,
    wdqkv,
    wuk,        // 用于推导 qkNopeHeadDim, kvLoraRank
    gamma1,     // 用于推导 qLoraRank
    kv_cache_rope,  // 用于推导 qkRopeHeadDim
    cache_mode,
    quant_mode,
    enableInnerOut
);
```

**注意**：`mla_preprocess_torch_adpt.h` 使用 `SetCustomHandler` 模式而非 `EXEC_NPU_CMD`，因为 MLAPO 需要自定义 tiling 计算（包含 3 个 PpMatmul tiling + RMSNorm + RoPE + EinSum tiling）。

### Step 5: Kernel 中替换硬编码

以 `mla_preprocess_mix_bf16.hpp` (MLAPO_BF16 变体) 为例：

```cpp
// Before: 硬编码
constexpr uint32_t MM1_OUT_SIZE = 2112;
constexpr uint32_t SPLIT_SIZE_TWO = 1536;

// After: 从 tiling 读取
uint32_t mm1OutSize = mlaTilingData.mm1OutSize;
uint32_t splitSizeTwo = mlaTilingData.splitSizeTwo;
```

**4 个 kernel 变体都需要修改**：
- `mla_preprocess_mix_bf16.hpp` (MLAPO_BF16)
- `mla_preprocess_mix_bf16_qdown.hpp` (MLAPO_BF16_INNER)
- `mla_preprocess_mix_bf16_nq.hpp` (MLAPO_BF16_NQ)
- `mla_preprocess_mix_fp16.hpp` (MLAPO_FP16)

**辅助类也需修改**：

```cpp
class RmsNormQuant {
    // Before
    void Process(...) {
        constexpr uint32_t SPLIT = 1536;
        // ...
    }

    // After
    uint32_t mm1OutSize_;
    void Init(const MlaTilingData& tiling) {
        mm1OutSize_ = tiling.mm1OutSize;
    }
    void Process(...) {
        uint32_t split = mm1OutSize_;
        // ...
    }
};
```

### Step 6: Kernel entry point 同步

文件: `csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp`

确保新 tiling 字段在 kernel 入口被正确拷贝：

```cpp
// 从 GM 拷贝 tiling 数据到 local
MlaTilingData mlaTilingData;
__gm__ MlaTilingData *tilingData = reinterpret_cast<__gm__ MlaTilingData *>(tiling);

// 需要拷贝新增字段
mlaTilingData.mm1OutSize = tilingData->mm1OutSize;
mlaTilingData.splitSizeOne = tilingData->splitSizeOne;
mlaTilingData.splitSizeTwo = tilingData->splitSizeTwo;
mlaTilingData.splitRmsNormSizeOne = tilingData->splitRmsNormSizeOne;
mlaTilingData.splitRmsNormSizeTwo = tilingData->splitRmsNormSizeTwo;
mlaTilingData.ropeSplitSizeOne = tilingData->ropeSplitSizeOne;
mlaTilingData.ropeSplitSizeTwo = tilingData->ropeSplitSizeTwo;
mlaTilingData.hiddenStrideRope = tilingData->hiddenStrideRope;
mlaTilingData.qkNopeHeadDim = tilingData->qkNopeHeadDim;
mlaTilingData.avgFactor = tilingData->avgFactor;
```

## Python 调用点（不需改动）

MLAPO 的 Python 调用点在 `vllm_ascend/attention/mla_v1.py` 的 `_mla_preprocess_only_decode()` 方法中。由于维度信息已编码在 tensor shape 中（`W_UK_T`, `gamma1` 等），Python 层**不需要传递额外维度参数**。这是维度参数化设计的核心优势。

```python
# mla_v1.py — 调用不变
torch.ops._C_ascend.mla_preprocess(
    hidden_states,
    self.wd_qkv,         # [hidden_dim, mm1OutSize]
    self.deq_scale_qkv,
    self.gamma1,          # [q_lora_rank] ← 推导 qLoraRank
    self.beta1,
    self.wu_q,
    self.qb_deq_scl,
    self.gamma2,
    cos, sin,
    self.W_UK_T,          # [headNum, qkNopeHeadDim, kvLoraRank] ← 推导维度
    decode_k_nope,
    decode_k_pe,          # [..., qkRopeHeadDim] ← 推导 qkRopeHeadDim
    slot_mapping,
    cache_mode="nzcache",
    quant_mode="per_tensor_quant_asymm",
    q_out0=q_out0, kv_cache_out0=kv_cache_out0,
    q_out1=q_out1, kv_cache_out1=kv_cache_out1,
    enable_inner_out=False,
    inner_out=torch.tensor([]),
)
```

## 踩坑记录

### 1. TilingData 字段追加位置

**问题**：在结构体中间插入新字段会导致偏移量变化，kernel 读取到错误数据。
**解决**：**始终在结构体末尾追加新字段**。已有字段的偏移量不变，确保后向兼容。

### 2. Kernel 辅助类遗漏

**问题**：只修改了主 kernel 函数中的硬编码，忘了修改 `Quant`, `RmsNormQuant` 等辅助类。
**表现**：GLM5 推理输出乱码（RMSNorm 使用了错误的 split 大小）。
**解决**：用 `grep` 搜索所有硬编码值的位置，包括 `.hpp` 文件中的类定义。

### 3. 默认值选择

**问题**：新字段没有默认值，DSV3 路径 tiling 计算未填充新字段 → kernel 读到 0。
**解决**：所有新 TilingData 字段必须带默认值 = 原模型（DSV3）的值。

### 4. Workspace 大小不足

**问题**：GLM5 的 mm1OutSize (2624) > DSV3 (2112)，workspace 按旧值分配 → buffer overflow。
**解决**：`SetMlapoWorkSpace()` 中的 workspace 计算必须使用参数化的维度值，不能硬编码。

### 5. 4 个 kernel 变体不一致

**问题**：只改了 MLAPO_BF16 变体，忘了改 MLAPO_FP16 / MLAPO_BF16_INNER / MLAPO_BF16_NQ。
**解决**：所有变体必须同步修改。建议创建 checklist 逐一确认。

## npu_rms_norm_bias 算子对比（补充案例）

`npu_rms_norm_bias` 是另一个 AscendC 算子集成案例，展示了不同于 MLAPO 的适配模式：

| 对比维度 | MLAPO | npu_rms_norm_bias |
|---------|-------|-------------------|
| 调用模式 | `SetCustomHandler` | `EXEC_NPU_CMD` |
| Tiling | 自定义（PpMatmul + 多子步骤） | CANN 框架自动 |
| Kernel 变体 | 4 个（dtype + quant mode） | 5 个（分 N/D 策略） |
| 维度参数化 | 需要（多模型 MLA 维度） | 不需要（通用 RMSNorm） |
| 复杂度 | 极高（20+ tensor，3 个 MatMul tiling） | 中等（3 tensor，单步计算） |

**关键经验**：`EXEC_NPU_CMD` 模式更简单可靠，如果 CANN 框架能自动处理 tiling 和调度，优先使用此模式。只有当需要自定义 tiling 逻辑（如多步 MatMul 联合 tiling）时才使用 `SetCustomHandler`。

**npu_rms_norm_bias 调用模式修复记录**：
- 最初错误使用 `OpCommand` + `Input/Output/Attr` 模式
- 运行时报 `aclnnRmsNormBias not found`
- 根因：`OpCommand` 模式在仓内无先例且行为未验证
- 修复为 `EXEC_NPU_CMD(aclnnRmsNormBias, ...)` 后运行正常

## 验证结果

### DeepSeek V3（回归测试）
- Tiling 字段使用默认值 → 行为与修改前完全一致
- 端到端推理输出精度无退化

### GLM5（新支持）
- Tiling 字段从 tensor shape 正确推导
- mm1OutSize=2624, splitSizeTwo=2048, hiddenStrideRope=256
- 端到端推理输出与 GPU 基准对齐（rtol=1e-3）
