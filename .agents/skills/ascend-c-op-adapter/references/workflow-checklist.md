# AscendC 算子适配 Workflow Checklist

逐步命令和代码模板，配合 `SKILL.md` 使用。

## Phase 1: 环境与信息收集

### 1.1 确认代码仓目录

```bash
# vllm-ascend 根目录
ls vllm-ascend/csrc/
ls vllm-ascend/csrc/torch_binding.cpp
ls vllm-ascend/CMakeLists.txt
```

### 1.2 收集目标模型维度

从 `configuration_*.py` 获取 MLA 相关维度：

```python
# DeepSeek V3
q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64
kv_lora_rank=512, v_head_dim=128, num_heads=128

# GLM5
q_lora_rank=2048, qk_nope_head_dim=192, qk_rope_head_dim=64
kv_lora_rank=512, v_head_dim=256, num_heads=64
```

### 1.3 确认已有算子的目录结构

```bash
# 以 mla_preprocess 为例
find csrc/mla_preprocess/ -type f | sort
# 预期输出:
# csrc/mla_preprocess/mla_preprocess_torch_adpt.h
# csrc/mla_preprocess/op_host/mla_preprocess.h
# csrc/mla_preprocess/op_host/tiling/mla_preprocess_tiling.h
# csrc/mla_preprocess/op_kernel/mla_preprocess.h
# csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
# csrc/mla_preprocess/op_kernel/*.hpp  (kernel 变体)
```

## Phase 2: 新建算子文件结构

### 2.1 创建目录

```bash
OP_NAME=my_new_op
mkdir -p csrc/${OP_NAME}/op_host/tiling
mkdir -p csrc/${OP_NAME}/op_kernel
```

### 2.2 TilingData 结构体模板

文件: `csrc/${OP_NAME}/op_host/tiling/${OP_NAME}_tiling.h`

```cpp
#ifndef MY_NEW_OP_TILING_H
#define MY_NEW_OP_TILING_H

#include <cstdint>

struct MyNewOpTilingData {
    // 通用字段
    uint32_t tilingKey{0};
    uint64_t userWorkspaceSize{0};
    uint32_t numCore{0};
    uint32_t n{0};           // batch size (num_tokens)
    uint32_t perTaskNum{0};
    uint32_t resTaskNum{0};

    // 模型特定维度 — 带默认值确保后向兼容
    uint32_t hiddenSize{7168};
    uint32_t kvLoraRank{512};
    uint32_t qkNopeHeadDim{128};
    float epsilon{1e-6f};
};

#endif
```

### 2.3 Torch 适配层模板

文件: `csrc/${OP_NAME}/${OP_NAME}_torch_adpt.h`

**模式 A — EXEC_NPU_CMD（推荐用于简单算子）**：

```cpp
#ifndef MY_NEW_OP_TORCH_ADPT_H
#define MY_NEW_OP_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> my_new_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    double epsilon)
{
    // 处理 optional tensor
    at::Tensor Bias = bias.has_value()
        ? bias.value()
        : at::empty({1}, input.options());

    // 分配输出
    at::Tensor output = at::empty(input.sizes(), input.options());
    std::vector<int64_t> rstd_shape = {input.size(0), 1};
    at::Tensor rstd = at::empty(rstd_shape, input.options().dtype(at::kFloat));

    // 调用 CANN aclnn 接口
    EXEC_NPU_CMD(aclnnMyNewOp, input, weight, Bias, epsilon, output, rstd);

    return std::tuple<at::Tensor, at::Tensor>(output, rstd);
}

}  // namespace vllm_ascend
#endif
```

**模式 B — SetCustomHandler（用于自定义 tiling 算子）**：

```cpp
#ifndef MY_NEW_OP_TORCH_ADPT_H
#define MY_NEW_OP_TORCH_ADPT_H

#include "op_host/my_new_op.h"

namespace vllm_ascend {

at::Tensor my_new_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor& output)
{
    // 1. Tiling 计算
    auto [workspace_tensor, tiling, block_dim] = my_new_op_tiling(input, weight);

    // 2. 提取原始指针
    void* input_ptr = input.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* output_ptr = output.data_ptr();
    void* workspace_ptr = workspace_tensor.data_ptr();
    void* tiling_ptr = tiling.data_ptr();

    // 3. 通过 SetCustomHandler 提交 kernel
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("my_new_op");
    cmd.SetCustomHandler(
        [stream, input_ptr, weight_ptr, output_ptr,
         workspace_ptr, tiling_ptr, block_dim]() -> int {
            my_new_op_impl(stream, input_ptr, weight_ptr,
                           output_ptr, workspace_ptr, tiling_ptr, block_dim);
            return 0;
        });
    cmd.Run();

    return output;
}

}  // namespace vllm_ascend
#endif
```

## Phase 3: Torch 注册

### 3.1 `torch_binding.cpp` 添加

```cpp
// 1. 顶部添加 include
#include "<op_name>/<op_name>_torch_adpt.h"

// 2. 在 TORCH_LIBRARY_EXPAND 块中添加 (与已有算子对齐)
ops.def(
    "my_new_op(Tensor input, Tensor weight, Tensor? bias=None, "
    "          float epsilon=1e-6) -> (Tensor output, Tensor rstd)"
);
ops.impl("my_new_op", torch::kPrivateUse1, &vllm_ascend::my_new_op);
```

### 3.2 `torch_binding_meta.cpp` 添加

```cpp
// 在 meta namespace 中添加
std::tuple<at::Tensor, at::Tensor> my_new_op_meta(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    double epsilon)
{
    at::Tensor output = at::empty_symint(input.sym_sizes(), input.options());
    std::vector<c10::SymInt> rstd_shape = {input.sym_size(0), c10::SymInt(1)};
    at::Tensor rstd = at::empty_symint(rstd_shape, input.options().dtype(at::kFloat));
    return {output, rstd};
}

// 在 TORCH_LIBRARY_EXPAND 块中添加
ops.impl("my_new_op", &vllm_ascend::meta::my_new_op_meta);
```

### 3.3 `CMakeLists.txt` 添加

```cmake
set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/<op_name>/op_kernel/<op_name>_kernel.cpp
    # ... 已有算子
)

# 如需排除特定硬件（如 310P）
set(VLLM_ASCEND_CUSTOM_OP_EXCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/<op_name>/op_kernel/<op_name>_kernel.cpp
    # ...
)
```

## Phase 4: 维度参数化（已有算子适配新模型）

### 4.1 识别硬编码常量

```bash
# 搜索 kernel 中的硬编码值
grep -rn "1536\|2112\|128\b" csrc/<op_name>/op_kernel/*.hpp
grep -rn "1536\|2112\|128\b" csrc/<op_name>/op_host/*.h
```

### 4.2 新增 TilingData 字段

```cpp
// 在 tiling_data.h 中新增，带默认值=原模型值
uint32_t mm1OutSize{2112};        // q_lora_rank + kv_lora_rank + qk_rope_head_dim
uint32_t splitSizeTwo{1536};       // q_lora_rank
uint32_t qkNopeHeadDim{128};       // qk_nope_head_dim
```

### 4.3 从 tensor shape 推导

```cpp
// 在 Tiling Init() 中
uint32_t qkNopeHeadDim = wuk.sizes()[1];   // wuk: [headNum, qkNopeHeadDim, kvLoraRank]
uint32_t kvLoraRank = wuk.sizes()[2];
uint32_t qLoraRank = gamma1.sizes()[0];    // gamma1: [qLoraRank]

tilingData->mm1OutSize = qLoraRank + kvLoraRank + qkRopeHeadDim;
tilingData->splitSizeTwo = qLoraRank;
tilingData->qkNopeHeadDim = qkNopeHeadDim;
```

### 4.4 替换 kernel 硬编码

```cpp
// Before
constexpr uint32_t MM1_OUT = 2112;
// After
uint32_t mm1Out = tilingData.mm1OutSize;
```

### 4.5 修改辅助类

```cpp
// kernel 内辅助类（如 Quant, RmsNormQuant）需同步修改
class RmsNormQuant {
    uint32_t mm1OutSize_;  // 新增成员变量
public:
    void Init(const MlaTilingData& tiling) {
        mm1OutSize_ = tiling.mm1OutSize;
    }
};
```

## Phase 5: Python 调用集成

### 5.1 调用方式

```python
# 直接调用已注册算子
torch.ops._C_ascend.my_new_op(
    input_tensor,
    weight_tensor,
    bias=bias_tensor,      # optional
    epsilon=1e-6,
    output=output_tensor,  # pre-allocated mutable output
    rstd=rstd_tensor,      # pre-allocated mutable output
)
```

### 5.2 Wrapper 函数（推荐）

```python
# vllm_ascend/ops/<op_name>.py
import torch

def my_new_op(input, weight, bias=None, epsilon=1e-6):
    output = torch.empty_like(input)
    rstd = torch.empty(input.shape[0], 1, dtype=torch.float32, device=input.device)
    torch.ops._C_ascend.my_new_op(input, weight, bias, epsilon, output, rstd)
    return output, rstd
```

## Phase 6: 编译与验证

### 6.1 完整重编译

```bash
# 必须完整编译以生成 aclnn wrapper
cd vllm-ascend
pip install -e . --no-build-isolation
```

### 6.2 单算子测试

```python
import torch
import torch_npu

# 准备测试数据
x = torch.randn(128, 7168, dtype=torch.bfloat16).npu()
w = torch.randn(7168, dtype=torch.bfloat16).npu()

# NPU 执行
output_npu = torch.ops._C_ascend.my_new_op(x, w)

# 参考实现
output_ref = torch_reference(x.cpu().float(), w.cpu().float())

# 精度对比
torch.testing.assert_close(
    output_npu.cpu().float(),
    output_ref,
    rtol=1e-3, atol=1e-3
)
```

### 6.3 端到端验证

```bash
# 启动推理服务
vllm serve <model_path> --dtype bfloat16 --max-model-len 4096

# 发送测试请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "Hello"}]}'
```
