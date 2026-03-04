# CANN 算子库知识库

## 概述

CANN（Compute Architecture for Neural Networks）是昇腾 AI 异构计算架构，提供四大算子库：

| 算子库 | 全称 | 覆盖领域 | 算子数量级 |
|--------|------|----------|-----------|
| **ops-nn** | Neural Network | 激活函数、卷积、池化、归一化、损失函数 | 1400+ |
| **ops-math** | Mathematics | 线性代数、矩阵运算、基础数学函数 | 数百 |
| **ops-transformer** | Transformer | 注意力机制、MLA、MoE、位置编码 | 数十（高度融合） |
| **ops-cv** | Computer Vision | 图像处理、色彩空间、几何变换 | 数百 |

**仓库地址**: https://gitcode.com/cann/

---

## 1. 架构分层

```
┌─────────────────────────────────┐
│   应用接口层 (ACLNN API)         │  ← 开发者调用入口
├─────────────────────────────────┤
│   核心算子层 (AscendC Kernels)   │  ← 1400+ 优化算子
├─────────────────────────────────┤
│   硬件抽象层 (Cube/Vector/Scalar)│  ← 达芬奇架构单元调度
└─────────────────────────────────┘
```

---

## 2. ops-nn（神经网络算子库）

### 算子分类

| 类别 | 典型算子 | 说明 |
|------|---------|------|
| **激活函数** | ReLU, GELU, SiLU, Swish, Mish | 逐元素非线性变换 |
| **归一化** | RMSNorm, LayerNorm, BatchNorm, GroupNorm | 特征归一化 |
| **卷积** | Conv2D, DepthwiseConv, DeformableConv | 卷积操作 |
| **池化** | MaxPool, AvgPool, AdaptivePool | 下采样 |
| **损失函数** | CrossEntropy, NLLLoss, SmoothL1 | 训练损失计算 |
| **融合算子** | AddRmsNormBias, MatmulAllReduceAddRmsNorm | 多步融合减少 HBM 访问 |

### vllm-ascend 中使用的 ops-nn 算子

| 算子 | 调用方式 | 用途 |
|------|---------|------|
| `npu_rms_norm` | `torch.ops.npu.npu_rms_norm` | QK LayerNorm |
| `npu_add_rms_norm` | `torch.ops.npu.npu_add_rms_norm` | 残差 + RMSNorm |
| `npu_add_rms_norm_bias` | 自定义 AscendC（csrc/add_rms_norm_bias） | 残差 + RMSNorm + bias |
| `npu_apply_top_k_top_p` | 自定义 AscendC（csrc/apply_top_k_top_p_custom） | 采样过滤 |

---

## 3. ops-math（数学算子库）

### 算子分类

| 类别 | 典型算子 | 说明 |
|------|---------|------|
| **矩阵运算** | MatMul, BatchMatMul, GroupedMatmul | BLAS 级矩阵乘 |
| **逐元素** | Add, Mul, Div, Pow, Rsqrt, Clamp | 基础算术 |
| **规约** | ReduceSum, ReduceMax, ReduceMean | 维度规约 |
| **排序** | TopK, Sort, ArgMax | 排序与索引 |
| **类型转换** | Cast, Quantize, Dequantize | 数据类型转换 |

### vllm-ascend 中使用的 ops-math 算子

| 算子 | 调用方式 | 用途 |
|------|---------|------|
| `batch_matmul_transpose` | 自定义 AscendC（csrc/batch_matmul_transpose） | V 上投影 |
| `matmul_allreduce_add_rmsnorm` | 自定义 AscendC（csrc/matmul_allreduce_add_rmsnorm） | TP 融合 |
| `grouped_matmul_swiglu_quant` | 自定义 AscendC | MoE Expert FFN |

---

## 4. ops-transformer（Transformer 算子库）

### 算子分类

| 类别 | 典型算子 | 说明 |
|------|---------|------|
| **注意力** | FlashAttentionScore, IncreFlashAttention, PromptFlashAttention | 高效注意力计算 |
| **推理注意力** | FusedInferAttentionScore (V1/V2), SparseFlashAttention | 推理专用融合注意力 |
| **MLA** | MlaPreprocess (MlaProlog) | Multi-head Latent Attention 预处理 |
| **位置编码** | ApplyRotaryPosEmb, RoPE | 旋转位置编码 |
| **MoE** | MoeInitRoutingQuant, MoeGatingTopK, DispatchFFNCombine | MoE 路由与调度 |
| **GMM** | GroupedMatmul (V1-V4) | 分组矩阵乘（MoE Expert） |

### cann-ops-adv 融合算子完整清单

| 算子 | 功能 | 备注 |
|------|------|------|
| ApplyRotaryPosEmb | Q/K 旋转位置编码融合 | 推理用 |
| FFN / FFNV2 / FFNV3 | Transformer FFN 层融合 | 渐进优化 |
| FlashAttentionScore | 自注意力（训练） | FlashAttention 算法 |
| FlashAttentionScoreGrad / V2 | FlashAttention 反向传播 | 训练用 |
| FusedInferAttentionScore / V2 | 推理注意力 + 量化 + 分页 | PagedAttention |
| IncreFlashAttention V1-V4 | 增量 FlashAttention | Decode 阶段 |
| PromptFlashAttention V1-V3 | Prompt 阶段注意力 | Prefill 阶段 |
| GroupedMatmul V1-V4 | 分组矩阵乘 | 变长维度支持 |
| MoeInitRoutingQuant / V2 | MoE 路由 + 量化 | Token 分发 |
| MoeFinalizeRoutingV2Grad | MoE 路由反向 | 训练用 |
| Sinkhorn | Sinkhorn 距离 Expert 路由 | Switch Transformer |
| GroupedBiasAddGrad | GroupedBiasAdd 反向 | 训练用 |

### vllm-ascend 中的 Transformer 自定义算子

| 算子 | 目录 | 功能 |
|------|------|------|
| `mla_preprocess` | csrc/mla_preprocess/ | MLA decode: Q RMSNorm→量化→BMM→RoPE, KV RMSNorm→RoPE→写缓存 |
| `sparse_flash_attention` | csrc/sparse_flash_attention/ | Top-K 稀疏注意力 |
| `dispatch_ffn_combine` | csrc/dispatch_ffn_combine/ | Prefill MoE: dispatch→FFN→combine (INT8) |
| `dispatch_ffn_combine_bf16` | csrc/dispatch_ffn_combine_bf16/ | 同上 BF16 版本 |
| `dispatch_gmm_combine_decode` | csrc/dispatch_gmm_combine_decode/ | Decode MoE: dispatch→GMM→combine |
| `moe_gating_top_k` | csrc/moe_gating_top_k/ | Expert 选择: Top-K gating |
| `moe_init_routing_custom` | csrc/moe_init_routing_custom/ | MoE 路由初始化 |
| `dispatch_layout` | csrc/dispatch_layout/ | Token dispatch 布局计算 |
| `moe_dispatch_normal` | csrc/moe_dispatch_normal/ | 通用 MoE dispatch |
| `moe_combine_normal` | csrc/moe_combine_normal/ | 通用 MoE combine |
| `lightning_indexer_vllm` | csrc/lightning_indexer_vllm/ | SFA Top-K 稀疏索引 |

---

## 5. ops-cv（计算机视觉算子库）

| 类别 | 典型算子 | 说明 |
|------|---------|------|
| **图像处理** | Resize, Crop, Pad, Flip | 几何变换 |
| **色彩空间** | RGB2YUV, YUV2RGB, ColorJitter | 色彩转换 |
| **目标检测** | NMS, ROIAlign, ROIPool | 检测后处理 |
| **图像增强** | GaussianBlur, Normalize | 预处理 |

vllm-ascend 当前不涉及 ops-cv 算子。

---

## 6. AscendC 算子标准目录结构

```
<operator_name>/
├── <operator_name>_torch_adpt.h      # PyTorch↔ACLNN 桥接
├── op_host/                           # Host 端: Tiling + 注册
│   ├── <op>_def.cpp                  # OP_ADD 算子注册
│   ├── <op>_tiling.cpp               # Tiling 策略计算
│   ├── <op>_tiling.h                 # TilingData 结构体
│   ├── <op>_infershape.cpp           # 输出 shape 推导
│   ├── aclnn_<op>.h                  # ACLNN C API（CANN 自动生成）
│   └── CMakeLists.txt
└── op_kernel/                         # Device 端: AscendC Kernel
    ├── <op>.h                        # 主 Kernel 实现
    ├── <op>_*.h/.hpp                 # 变体 Kernel（dtype/tiling 策略）
    ├── <op>_kernel.cpp               # 入口分发
    └── kernel/                       # 可选: 公共工具
        ├── common_func.h
        ├── iterator.h
        └── mma.h
```

### 关键文件说明

| 文件 | 职责 | 调用模式 |
|------|------|---------|
| `*_torch_adpt.h` | `EXEC_NPU_CMD(aclnn<Op>, ...)` 调用 CANN 算子 | PyTorch → ACLNN |
| `*_def.cpp` | `OP_ADD` + `IMPL_OP_OPTILING` 注册 | CANN 框架发现算子 |
| `*_tiling.cpp` | 根据 tensor shape 和硬件约束计算 tile 维度 | Host CPU 执行 |
| `*_kernel.cpp` | AscendC 三段式流水: CopyIn → Compute → CopyOut | Device NPU 执行 |

### torch_binding.cpp 注册模式

```cpp
// 声明
TORCH_LIBRARY_EXPAND(_C_ascend, ops) {
    ops.def("operator_name(Tensor x, ...) -> Tensor");
    ops.impl("operator_name", torch::kPrivateUse1, &impl_func);
}
```

---

## 7. 算子调用两种可靠模式

### 模式 1: ACLNN 标准算子（推荐）

```cpp
// torch_adpt.h 中
EXEC_NPU_CMD(aclnnOperatorName, input, output, ...);
```

适用于: `OP_ADD` + `IMPL_OP_OPTILING` 注册的 CANN 算子。

### 模式 2: SetCustomHandler 原始指针

```cpp
// 绕过 CANN 框架，直接操作 kernel
SetCustomHandler(stream, kernel_func, args...);
```

适用于: 需要直接控制 kernel launch 的场景。

**注意**: 不要使用 `OpCommand` + `Input/Output/Attr` 模式（仓内无先例，行为未验证）。

---

## 8. 编译与构建

### 前置依赖

- CANN Toolkit（与 NPU 驱动版本匹配）
- Python >= 3.7, GCC >= 7.3, CMake >= 3.16
- Protobuf <= 3.20.x

### vllm-ascend 编译

```bash
cd vllm-ascend
pip install -e . -v --no-build-isolation
```

### 独立算子编译（cann-ops-adv）

```bash
mkdir build && cd build
cmake .. && make package -j$(nproc)
# 输出: CANN-custom_ops-<version>-linux.<arch>.run
```

### 新增 kernel 后必须完整重编译

确保 aclnn wrapper 自动生成:
```bash
pip install -e . -v --no-build-isolation  # 完整重编译
```

---

## 9. vllm-ascend 自定义算子完整清单（22 个）

### 按推理流水线阶段分类

**注意力模块（5 个）**:
- `mla_preprocess` — MLA decode 预处理（RMSNorm+量化+BMM+RoPE 融合）
- `sparse_flash_attention` — Top-K 稀疏注意力
- `batch_matmul_transpose` — V 上投影矩阵乘
- `lightning_indexer_vllm` — SFA Top-K 索引
- `lightning_indexer_quant` — SFA 量化索引

**MoE 模块（11 个）**:
- `moe_gating_top_k` — Expert 选择
- `moe_init_routing_custom` — 路由初始化
- `dispatch_layout` — Token dispatch 布局
- `dispatch_ffn_combine` — Prefill MoE dispatch+FFN+combine (INT8)
- `dispatch_ffn_combine_bf16` — 同上 BF16
- `dispatch_gmm_combine_decode` — Decode MoE dispatch+GMM+combine
- `moe_dispatch_normal` — 通用 dispatch
- `moe_combine_normal` — 通用 combine
- `notify_dispatch` — 同步通知
- `moe_grouped_matmul` — 分组矩阵乘
- `grouped_matmul_swiglu_quant_weight_nz_tensor_list` — NZ 格式 Expert FFN

**归一化模块（1 个）**:
- `add_rms_norm_bias` — 残差+RMSNorm+bias

**融合通信模块（1 个）**:
- `matmul_allreduce_add_rmsnorm` — MatMul+AllReduce+Add+RMSNorm

**采样模块（1 个）**:
- `apply_top_k_top_p_custom` — Top-K/Top-P 采样

**序列模型（1 个）**:
- `causal_conv1d` — 因果 1D 卷积（Mamba/SSM）

**KV Cache（1 个）**:
- `transpose_kv_cache_by_block` — 块级 KV Cache 转置

**推测解码（1 个）**:
- `copy_and_expand_eagle_inputs` — EAGLE 输入扩展

---

## 参考资源

- CANN 算子库: https://gitcode.com/cann/
- cann-ops（基础算子）: https://gitee.com/ascend/cann-ops
- cann-ops-adv（融合算子）: https://gitee.com/ascend/cann-ops-adv
- ops-nn 深度解读: https://ascendai.csdn.net/698616480a2f6a37c59063b9.html
- ops-transformer MlaProlog 指南: https://ascendai.csdn.net/693a12b82087ae0db7a0d963.html
- Ascend C 开发入门: https://www.hiascend.com/
