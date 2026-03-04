---
name: ascend-c-op-adapter
description: "将 AscendC 自定义算子适配并集成到 vllm-ascend 代码仓。涵盖算子目录结构、Tiling 参数化、Torch 注册、Python 调用、多模型维度适配的全流程。"
---

# AscendC 自定义算子适配 Skill

## Overview

指导 AI Agent 将 AscendC 自定义算子（如 MLA Preprocess、RMSNorm+Bias 等融合算子）适配到 vllm-ascend 代码仓，并支持多模型维度参数化。适用于：

- **新算子集成**：将已有 AscendC kernel 注册到 vllm-ascend 的 `torch.ops._C_ascend` 命名空间
- **多模型适配**：将硬编码维度参数化，使同一算子支持 DeepSeek V3、GLM5 等不同 MLA 模型
- **算子调试**：排查 `aclnn*` 调用失败、Tiling 错误、精度不对等问题

## Read order

1. 先读本文件，理解全流程架构和硬约束
2. 开发时参考 `references/workflow-checklist.md`（步骤命令和代码模板）
3. 首次适配参考 `references/mlapo-glm5-case-study.md`（MLAPO 算子适配 GLM5 完整实战）
4. 遇到问题参考 `references/troubleshooting.md`（常见报错与解决方案）

## Hard constraints

- **算子调用模式**：仓内只有两种可靠调用模式
  - ① `EXEC_NPU_CMD(aclnn*, ...)` — 用于 `OP_ADD` + `IMPL_OP_OPTILING` 注册的标准 CANN 算子
  - ② `SetCustomHandler` + 原始指针 — 用于需要自定义 tiling + kernel launch 的算子（如 MLAPO）
  - **禁止使用** `OpCommand` + `Input/Output/Attr` 模式（行为未验证，会报 `aclnn* not found`）

- **文件结构规范**：每个 AscendC 算子必须遵循标准目录结构
  ```
  csrc/<op_name>/
  ├── op_host/                    # Host 端（tiling + infershape）
  │   ├── <op_name>.h             # Tiling 主逻辑
  │   └── tiling/
  │       └── <op_name>_tiling.h  # TilingData 结构体
  ├── op_kernel/                  # Device 端（AscendC kernel）
  │   ├── <op_name>_kernel.cpp    # Kernel entry point
  │   ├── <op_name>.h             # Kernel 头文件
  │   └── <variant>.hpp           # 各 kernel 变体
  └── <op_name>_torch_adpt.h      # PyTorch 适配层
  ```

- **Torch 注册三件套**：
  - `torch_binding.cpp` — `ops.def()` + `ops.impl()` NPU 实现
  - `torch_binding_meta.cpp` — `ops.impl()` Meta 实现（torch.compile shape inference）
  - `CMakeLists.txt` — kernel `.cpp` 加入编译列表

- **维度参数化**：从 tensor shape 推导维度，不硬编码模型常量
  - 通过 `wuk.sizes()`, `gamma1.sizes()`, `kv_cache_rope.sizes()` 等获取
  - `MlaTilingData` 结构体中用带默认值的字段，确保后向兼容

- **编译依赖**：新增 kernel 后必须完整重编译，确保 aclnn wrapper 自动生成

## Execution playbook

### 1) 需求分析

- 确认算子的计算公式和 I/O tensor 列表
- 确认是新建算子还是适配已有算子到新模型
- 确认调用路径（decode / prefill / 两者都有）
- 确认是否需要量化变体（W8A8、per-tensor、per-channel 等）
- 收集所有目标模型的维度参数（参考 `configuration_*.py`）

### 2) 选择适配策略

| 场景 | 策略 |
|------|------|
| 全新 AscendC 算子集成 | 按标准目录结构创建，参考 `add_rms_norm_bias` |
| 已有算子适配新模型 | 维度参数化：硬编码 → TilingData 字段 + tensor shape 推导 |
| 仅需 Python wrapper 变更 | 修改 `_torch_adpt.h` 和 Python 调用点 |
| 从 torch_npu 原生算子切换 | 用 `EXEC_NPU_CMD(aclnn*, ...)` 模式 |

### 3) 实现算子 Host 端

#### 3.1 定义 TilingData 结构体

在 `op_host/tiling/<op_name>_tiling.h` 中定义所有 tiling 字段：

```cpp
struct MyOpTilingData {
    uint32_t numCore{0};
    uint32_t n{0};
    // ... 模型特定维度，带默认值确保后向兼容
    uint32_t kvLoraRank{512};    // DeepSeek V3 默认值
    uint32_t qkNopeHeadDim{128}; // DeepSeek V3 默认值
};
```

#### 3.2 实现 Tiling 计算

在 `op_host/<op_name>.h` 中实现维度推导和 tiling 计算：

```cpp
// 从 tensor shape 推导维度
uint32_t qkNopeHeadDim = wuk.sizes()[1];
uint32_t kvLoraRank = wuk.sizes()[2];
uint32_t qLoraRank = gamma1.sizes()[0];

// 填充 tiling 字段
tilingData->kvLoraRank = kvLoraRank;
tilingData->qkNopeHeadDim = qkNopeHeadDim;
```

### 4) 实现 Torch 适配层

在 `<op_name>_torch_adpt.h` 中选择正确的调用模式：

**模式 A — `EXEC_NPU_CMD`（简单算子）**：
```cpp
EXEC_NPU_CMD(aclnnMyOp, input, weight, gamma, epsilon, output, rstd);
return std::tuple<at::Tensor, at::Tensor>(output, rstd);
```

**模式 B — `SetCustomHandler`（自定义 tiling 算子）**：
```cpp
auto [workspace, tiling, block_dim] = my_op_tiling(inputs...);
aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
at_npu::native::OpCommand cmd;
cmd.Name("my_op");
cmd.SetCustomHandler([stream, ptrs..., tiling_ptr, block_dim]() -> int {
    my_op_impl(stream, ptrs..., tiling_ptr, block_dim);
    return 0;
});
cmd.Run();
```

### 5) Torch 注册

#### 5.1 `torch_binding.cpp`

```cpp
// 头部添加 include
#include "<op_name>/<op_name>_torch_adpt.h"

// ops.def() — 声明算子 schema
ops.def("my_op(Tensor input, Tensor weight, ...) -> (Tensor output, Tensor rstd)");
// ops.impl() — 注册 NPU 实现
ops.impl("my_op", torch::kPrivateUse1, &vllm_ascend::my_op);
```

#### 5.2 `torch_binding_meta.cpp`

```cpp
// Meta 实现（仅返回 shape，不做计算）
at::Tensor output = at::empty(input.sizes(), input.options());
at::Tensor rstd = at::empty(rstd_shape, input.options().dtype(at::kFloat));
return {output, rstd};
```

#### 5.3 `CMakeLists.txt`

```cmake
set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/<op_name>/op_kernel/<op_name>_kernel.cpp
)
```

### 6) Python 调用点

```python
# 在 vllm_ascend/attention/ 或 vllm_ascend/ops/ 中调用
torch.ops._C_ascend.my_op(
    input, weight, gamma,
    epsilon=1e-6,
    output=pre_allocated_output,
    rstd=pre_allocated_rstd,
)
```

### 7) 多模型维度适配

当同一算子需要支持多个模型（如 DeepSeek V3 → GLM5）时：

1. **对比维度差异**：列出所有硬编码常量及各模型值
2. **找到 tensor shape 到维度的映射**：确认哪些输入 tensor 的 shape 可推导出目标维度
3. **在 TilingData 中新增字段**：带默认值（=原模型值）确保后向兼容
4. **修改 Tiling 计算**：从 tensor shape 推导填充新字段
5. **修改 Kernel**：用 `tilingData->newField` 替换硬编码常量
6. **修改 Kernel 辅助类**：如 `Quant`, `RmsNormQuant` 等类中的成员变量

### 8) 验证

#### 功能验证

```bash
# 单算子测试
python3 test_my_op.py  # 对比 torch 参考实现

# 端到端推理
vllm serve <model> --dtype bfloat16 --max-model-len 4096
# 发送请求验证输出正确性
```

#### 精度验证

```python
# 对比优化前后的中间 tensor
ref_output = torch_reference_impl(input, weight, gamma)
npu_output = torch.ops._C_ascend.my_op(input, weight, gamma)
torch.testing.assert_close(ref_output, npu_output, rtol=1e-3, atol=1e-3)
```

### 9) 交付 Checklist

#### 文件完整性
- [ ] `csrc/<op_name>/` 目录结构完整（op_host + op_kernel + torch_adpt）
- [ ] `torch_binding.cpp` 已注册 `ops.def()` + `ops.impl()`
- [ ] `torch_binding_meta.cpp` 已注册 Meta 实现
- [ ] `CMakeLists.txt` 已添加 kernel 编译路径
- [ ] Python 调用点已更新

#### 功能正确性
- [ ] 单算子精度对齐（rtol=1e-3, atol=1e-3）
- [ ] 端到端推理输出正确
- [ ] 所有目标模型（DSV3/GLM5/...）均验证通过

#### 多模型兼容
- [ ] TilingData 新字段带默认值（后向兼容）
- [ ] 维度从 tensor shape 推导，不硬编码
- [ ] Kernel 中所有硬编码常量已替换为 tiling 字段

#### 代码规范
- [ ] 调用模式正确（`EXEC_NPU_CMD` 或 `SetCustomHandler`，不用 `OpCommand`）
- [ ] 完整重编译后 aclnn wrapper 自动生成
- [ ] optional tensor 使用 `c10::optional<at::Tensor>` 并处理无值情况

## Key file locations

| 文件 | 路径 |
|------|------|
| MLAPO 算子（参考） | `csrc/mla_preprocess/` |
| add_rms_norm_bias（参考） | `csrc/add_rms_norm_bias/` |
| rms_norm_bias（参考） | `csrc/rms_norm_bias/` |
| Torch 注册 | `csrc/torch_binding.cpp` |
| Torch Meta 注册 | `csrc/torch_binding_meta.cpp` |
| CMake 构建 | `CMakeLists.txt` |
| MLA attention（Python 调用点） | `vllm_ascend/attention/mla_v1.py` |
| SFA attention（Python 调用点） | `vllm_ascend/attention/sfa_v1.py` |
| LayerNorm（Python 调用点） | `vllm_ascend/ops/layernorm.py` |

## Reference documents

- `references/workflow-checklist.md` — 步骤命令和代码模板速查
- `references/mlapo-glm5-case-study.md` — MLAPO 算子适配 GLM5 完整实战（含代码 diff 和踩坑记录）
- `references/troubleshooting.md` — 常见报错与解决方案
