# NPU Triton 代码模板

## 1. 标准循环模式（逐行处理）

适用于：RMSNorm、Activation、RoPE 等逐 token 操作。

```python
import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


def my_elementwise_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Wrapper: 输入校验 + grid 计算 + kernel 启动"""
    assert x.is_contiguous()
    num_tokens, hidden_size = x.shape
    pad_hidden = triton.next_power_of_2(hidden_size)
    output = torch.empty_like(x)

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)

    _elementwise_kernel[grid](
        x, x.stride(0),
        weight,
        output, output.stride(0),
        num_tokens, hidden_size,
        pad_hidden,
        eps,
    )
    return output


@triton.jit
def _elementwise_kernel(
    x_ptr, x_stride,
    w_ptr,
    out_ptr, out_stride,
    num_tokens, hidden_size,
    PAD_HIDDEN: tl.constexpr,
    eps,
):
    pid = tl.program_id(0).to(tl.int64)
    num_cores = tl.num_programs(0)

    # 核内循环：每个核处理多行
    for row_idx in tl.range(pid, num_tokens, num_cores):
        cols = tl.arange(0, PAD_HIDDEN)
        mask = cols < hidden_size

        # 加载（带 mask 和 other）
        x = tl.load(x_ptr + row_idx * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # 计算（float32 精度）
        mean_sq = tl.sum(x * x) / hidden_size
        rrms = tl.rsqrt(mean_sq + eps)
        result = (x * rrms * w).to(x_ptr.dtype.element_ty)

        # 存储
        tl.store(out_ptr + row_idx * out_stride + cols, result, mask=mask)
```

## 2. 2D Grid 模式（多维并行）

适用于：多 head 处理（如 split_qkv_rmsnorm_rope）。

```python
def my_2d_op(x: torch.Tensor, n_heads: int, head_dim: int):
    num_tokens = x.shape[0]
    num_vectorcore = get_vectorcore_num()

    n_cols = n_heads  # 第二维度大小
    n_rows = num_vectorcore // n_cols  # 第一维度分配的核数

    grid = (n_rows, n_cols)
    _2d_kernel[grid](x, num_tokens, n_heads, head_dim)


@triton.jit
def _2d_kernel(x_ptr, num_tokens, n_heads, head_dim):
    row_pid = tl.program_id(0).to(tl.int64)
    col_pid = tl.program_id(1).to(tl.int64)  # head index
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_pid, num_tokens, row_step):
        head_offset = col_pid * head_dim
        cols = tl.arange(0, head_dim)
        # 处理 (row_idx, col_pid) 位置的数据
        ...
```

## 3. 持久化内核模式（常驻核）

适用于：小数据量但频繁调用的算子（如 batch_invariant RMSNorm）。

```python
@triton.jit
def _persistent_kernel(
    x_ptr, out_ptr,
    num_tokens, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_cores = tl.num_programs(0)

    # Grid 固定为 vectorcore 数，核内循环处理所有数据
    for token_idx in tl.range(pid, num_tokens, num_cores):
        for block_start in range(0, hidden_size, BLOCK_SIZE):
            cols = block_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < hidden_size
            x = tl.load(x_ptr + token_idx * hidden_size + cols, mask=mask, other=0.0)
            # 处理...
            tl.store(out_ptr + token_idx * hidden_size + cols, x, mask=mask)
```

## 4. 二级分块模式（UB 受限时）

适用于：tile 超 192KB 的场景。

```python
@triton.jit
def _tiled_kernel(
    x_ptr, out_ptr,
    num_tokens, hidden_size,
    XBLOCK: tl.constexpr,      # 每核处理的元素总数
    XBLOCK_SUB: tl.constexpr,  # 每次迭代的子块大小（适配 UB）
):
    pid = tl.program_id(0).to(tl.int64)
    xoffset = pid * XBLOCK

    for sub_offset in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + sub_offset + tl.arange(0, XBLOCK_SUB)
        mask = x_index < num_tokens * hidden_size
        x = tl.load(x_ptr + x_index, mask=mask, other=0.0)
        result = x * x  # 计算
        tl.store(out_ptr + x_index, result, mask=mask)
```

## 5. Block Pointer 模式（2D 矩阵访问）

适用于：含 `tl.dot` 的矩阵运算（FLA 内核等）。

```python
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointer（自动边界检查）
    p_a = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    p_b = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN),
        block_shape=(BK, BN),
        order=(1, 0),
    )

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(p_a, boundary_check=(0, 1))
        b = tl.load(p_b, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        p_a = tl.advance(p_a, (0, BK))
        p_b = tl.advance(p_b, (BK, 0))

    # 存储结果
    p_c = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    tl.store(p_c, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))
```

## 6. 算子融合模式（多操作合一）

适用于：连续操作的中间结果可留在 UB。

```python
@triton.jit
def _fused_rmsnorm_rope_kernel(
    x_ptr, x_stride,
    w_ptr,
    cos_ptr, sin_ptr,
    out_ptr, out_stride,
    num_tokens, hidden_size, rope_dim,
    PAD_HIDDEN: tl.constexpr,
    eps,
):
    pid = tl.program_id(0).to(tl.int64)
    num_cores = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, num_cores):
        cols = tl.arange(0, PAD_HIDDEN)
        mask = cols < hidden_size

        # Op 1: Load
        x = tl.load(x_ptr + row_idx * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Op 2: RMSNorm（中间结果留在寄存器/UB）
        mean_sq = tl.sum(x * x) / hidden_size
        x_norm = x * tl.rsqrt(mean_sq + eps) * w

        # Op 3: RoPE（直接对 UB 中的 x_norm 操作）
        rope_cols = tl.arange(0, rope_dim // 2)
        cos = tl.load(cos_ptr + row_idx * (rope_dim // 2) + rope_cols)
        sin = tl.load(sin_ptr + row_idx * (rope_dim // 2) + rope_cols)
        # ... apply rotation ...

        # Op 4: Store（仅一次 HBM 写入）
        tl.store(out_ptr + row_idx * out_stride + cols, result, mask=mask)
```

## 7. Fake 实现模板（torch.compile 兼容）

```python
# 注册自定义算子
torch.library.custom_op("_C_ascend::my_op", my_op_impl)

# Fake 实现（仅返回 shape，不计算）
@torch.library.register_fake("_C_ascend::my_op")
def my_op_fake(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    return torch.empty_like(x)
```

## 8. Autotune 配置模板

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, multibuffer=True, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, multibuffer=True, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, multibuffer=True, num_warps=8, num_stages=3),
    ],
    key=["hidden_size"],  # 按 hidden_size 选择最优配置
)
@triton.jit
def _autotuned_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

## 9. Heuristics + do_not_specialize 模板

```python
@triton.heuristics({
    "HAS_BIAS": lambda args: args["bias_ptr"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["num_tokens"])  # 避免对变化频繁的参数重编译
def _flexible_kernel(
    ...,
    bias_ptr,
    cu_seqlens,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_tokens,
):
    if HAS_BIAS:  # 编译时消除
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        x = x + bias
    if IS_VARLEN:  # 编译时消除
        bos = tl.load(cu_seqlens + batch_idx)
```
