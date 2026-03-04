---
name: npu-triton-dev
description: "开发高性能 NPU Triton 算子的完整指导。涵盖从零编写、性能分析到优化极限判定的全流程。"
---

# NPU Triton 高性能算子开发 Skill

## Overview

指导 AI Agent 在 Ascend NPU 上开发、调优和验证 Triton 算子。适用于新算子开发、GPU 算子迁移、性能优化三种场景。

## Read order

1. 先读本文件，理解全流程
2. 开发时参考 `references/templates.md`（代码模板）
3. 性能分析参考 `references/perf-analysis.md`（Roofline + msprof）
4. 遇到编译问题参考 `references/troubleshooting.md`（常见报错与解决）

## Hard constraints

- **UB 容量 192KB**：单核片上内存上限，tile 数据量不能超出
- **Grid ≤ 物理核数**：纯向量用 `get_vectorcore_num()`，含 `tl.dot` 用 `get_aicore_num()`
- **不支持 uint64 / float64**：必须用 int32 / float32 替代
- **不支持链式布尔**：`a or b or c` 改为 `(a or b) or c`
- **对齐要求**：VV 32 字节，CV 512 字节
- **核内循环必须**：`tl.range(pid, total, num_programs)` 分批处理

## Execution playbook

### 1) 需求分析

- 确认算子的数学定义（输入/输出/计算公式）
- 确认调用场景（decode hot path? prefill? 融合候选?）
- 确认数据类型（bf16/fp16/fp32/int8）和典型 shape
- 判断瓶颈类型：
  ```
  计算强度 = FLOPs / Bytes
  Vector 平衡点 ≈ 6 FLOPs/Byte
  Cube 平衡点 ≈ 178 FLOPs/Byte
  大多数激活/归一化/RoPE 算子是 Memory-bound
  ```

### 2) 选择开发策略

| 场景 | 策略 |
|------|------|
| 全新算子 | 从模板开始，按 Step 3 编写 |
| GPU Triton 迁移 | 保留计算逻辑，重写 grid/loop/tiling/dtype |
| 已有算子优化 | 跳到 Step 5 性能分析 |
| 多算子融合 | 合并计算逻辑到一个 kernel，中间结果留 UB |

### 3) 编写 Kernel（从模板出发）

#### 3.1 Wrapper 函数模板

```python
import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

def my_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert x.is_contiguous(), "Input must be contiguous"
    num_tokens, hidden_size = x.shape
    pad_hidden = triton.next_power_of_2(hidden_size)
    output = torch.empty_like(x)

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)

    _my_kernel[grid](
        x, x.stride(0),
        weight,
        output, output.stride(0),
        num_tokens, hidden_size,
        pad_hidden,
        eps,
    )
    return output
```

#### 3.2 Kernel 模板（标准循环模式）

```python
@triton.jit
def _my_kernel(
    x_ptr, x_stride,
    w_ptr,
    out_ptr, out_stride,
    num_tokens, hidden_size,
    PAD_HIDDEN: tl.constexpr,
    eps,
):
    pid = tl.program_id(0).to(tl.int64)
    num_cores = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, num_cores):
        cols = tl.arange(0, PAD_HIDDEN)
        mask = cols < hidden_size

        x = tl.load(x_ptr + row_idx * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # 计算逻辑
        mean_sq = tl.sum(x * x) / hidden_size
        rrms = tl.rsqrt(mean_sq + eps)
        result = (x * rrms * w).to(x_ptr.dtype.element_ty)

        tl.store(out_ptr + row_idx * out_stride + cols, result, mask=mask)
```

#### 3.3 UB 容量预检

```python
# 编写前预估 UB 占用
tile_elements = PAD_HIDDEN  # 或 BLOCK_M × BLOCK_N
bytes_per_element = 2  # bf16
num_tiles_in_flight = 3  # 输入 + 中间 + 输出（至少）
ub_usage = tile_elements * bytes_per_element * num_tiles_in_flight
assert ub_usage <= 192 * 1024, f"UB overflow: {ub_usage} > {192*1024}"
```

如果超出 192KB，必须引入二级分块：
```python
for sub_offset in range(0, PAD_HIDDEN, SUB_BLOCK):
    sub_cols = sub_offset + tl.arange(0, SUB_BLOCK)
    # 处理子块
```

### 4) 功能验证

```bash
# Step 1: CPU 解释器模式（精度基准）
export TRITON_INTERPRET=1
python3 test_my_op.py

# Step 2: NPU 执行
unset TRITON_INTERPRET
python3 test_my_op.py

# Step 3: 精度对比
# torch.testing.assert_close(result_cpu, result_npu, rtol=1e-3, atol=1e-3)
```

必须测试的边界条件：
- `num_tokens = 1`（单 token decode）
- `hidden_size` 非 2 的幂
- 大 batch（`num_tokens > vectorcore_num`，验证循环正确性）

### 5) 性能分析（Roofline + msprof）

#### 5.1 计算理论极限

```python
def calc_theoretical_time(data_bytes, flops, gm_bw=1.8e12, vec_peak=11.06e12):
    t_memory = data_bytes / gm_bw
    t_compute = flops / vec_peak
    t_theory = max(t_memory, t_compute)
    bound = "Memory-bound" if t_memory > t_compute else "Compute-bound"
    return t_theory * 1e6, bound  # 返回 μs

# 示例: RMSNorm (batch=128, hidden=7168, bf16)
data_bytes = 128 * 7168 * 2 * 2 + 7168 * 2  # 输入+输出+权重
flops = 128 * 7168 * 5                        # sq+sum+rsqrt+mul+mul
t_us, bound = calc_theoretical_time(data_bytes, flops)
print(f"理论极限: {t_us:.2f} μs ({bound})")
```

#### 5.2 实测采集

```bash
# On-device profiling
msprof op --kernel-name=_my_kernel python3 test_my_op.py

# Simulation profiling（无需硬件）
msprof op simulator --kernel-name=_my_kernel --soc-version=Ascend910B1 python3 test_my_op.py
```

#### 5.3 性能比判定

```
性能比 = 实际耗时 / 理论极限

≤ 1.2  → 已达极限，停止优化
1.2~2.0 → 有空间，分析 PipeUtilization
> 2.0   → 必须优化
```

#### 5.4 PipeUtilization 瓶颈定位

| 现象 | 诊断 | 行动 |
|------|------|------|
| mte2 >> vec | 搬运瓶颈，计算空等 | `multibuffer=True`、算子融合 |
| vec >> mte2 | 计算瓶颈 | 低精度、减少计算 |
| scalar 高 | 标量瓶颈 | 增大 BLOCK、向量化 |
| 所有都低 | 流水空泡 | 检查 tiling、multibuffer |
| vec ≈ mte2 且都高 | 理想状态 | 已接近极限 |

### 6) 优化迭代

按优先级尝试：

1. **multibuffer=True** — 搬运与计算流水并行
2. **增大 BLOCK_SIZE** — 减少循环次数降低标量开销
3. **算子融合** — 合并相邻操作减少 HBM 访问
4. **heuristics** — 编译时分支消除（`@triton.heuristics`）
5. **do_not_specialize** — 避免不必要的 JIT 重编译
6. **向量化标量** — `i64→f32` 后用向量比较替代标量比较
7. **调整 num_warps/num_stages** — 常见搭配：warps=4/stages=3

每次优化后必须重新执行 Step 5 验证效果。

### 7) 集成到 vllm-ascend

#### 文件放置

```
vllm_ascend/ops/triton/
├── my_new_op.py          # kernel + wrapper
├── triton_utils.py       # 已有，get_vectorcore_num() 等
```

#### 注册（如需 torch.compile 兼容）

```python
# 在 ops/__init__.py 或相应位置注册
torch.library.custom_op("_C_ascend::my_op", my_op_impl)

# Fake 实现（图模式 tracing）
@torch.library.register_fake("_C_ascend::my_op")
def my_op_fake(x, weight, eps=1e-6):
    return torch.empty_like(x)
```

#### 调用点

```python
# 在模型代码中调用
from vllm_ascend.ops.triton.my_new_op import my_op
output = my_op(x, weight, eps)
```

### 8) 交付 Checklist

#### 功能
- [ ] TRITON_INTERPRET=1 精度正确
- [ ] NPU 执行结果与 CPU 基准对齐（rtol=1e-3, atol=1e-3）
- [ ] 边界条件通过（num_tokens=1, 非对齐维度, 大 batch）
- [ ] Fake 实现已注册（torch.compile 兼容）

#### 性能
- [ ] 理论极限已计算（含瓶颈类型判定）
- [ ] msprof 数据已采集
- [ ] 性能比 ≤ 1.5（可接受）或 ≤ 1.2（优秀）
- [ ] PipeUtilization 无明显异常

#### 代码规范
- [ ] Grid ≤ 物理核数
- [ ] 核内 `tl.range` 循环
- [ ] 无 uint64/float64/链式布尔
- [ ] UB 用量预估 ≤ 192KB
- [ ] `mask` + `other` 处理边界
- [ ] 中间计算 float32，存储转回原 dtype

## Key file locations

| 文件 | 路径 |
|------|------|
| Triton 工具函数 | `vllm_ascend/ops/triton/triton_utils.py` |
| RoPE 算子（参考） | `vllm_ascend/ops/triton/rope.py` |
| RMSNorm 算子（参考） | `vllm_ascend/ops/triton/batch_invariant/rmsnorm.py` |
| SwiGLU+量化融合（参考） | `vllm_ascend/ops/triton/activation/swiglu_quant.py` |
| QKV 分割+RMSNorm+RoPE 融合（参考） | `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py` |
| FLA 内核（高级参考） | `vllm_ascend/ops/triton/fla/` |
| Mamba 算子（启发式 tiling 参考） | `vllm_ascend/ops/triton/mamba/causal_conv1d.py` |

## Reference documents

- `NPU_TRITON_BEST_PRACTICES.md` — 开发最佳实践完整版
- `NPU_TRITON_PERF_ANALYSIS.md` — 性能分析与优化极限判定完整版
- `TRITON_CUDA_VS_NPU.md` — CUDA Triton 与 NPU Triton 差异分析
- [triton-ascend 官方文档](https://ascend.github.io/triton-ascend/)
- [msprof 性能分析指南](https://github.com/Ascend/triton-ascend/blob/master/docs/sources/mindstudio-guide/01-msProf_op.md)
