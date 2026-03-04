# NPU Triton 常见问题排查

## 1. 编译错误

### UB Overflow

```
error: ub overflow, requires XXXX bits while 1572864 bits available
```

**原因**：tile 数据超过 192KB UB 容量。
**解决**：
1. 减小 `PAD_HIDDEN` / `BLOCK_SIZE`
2. 引入二级分块 `XBLOCK_SUB`
3. 减少同时在 UB 中的 tensor 数量
4. 如果无法拆分，fallback 到 NPU 原生算子

**预估公式**：
```python
ub_bytes = tile_elements * bytes_per_element * num_concurrent_tiles
# num_concurrent_tiles ≈ 输入tile数 + 输出tile数 + 中间tile数
# 需要 ub_bytes ≤ 192 * 1024
```

### CoreDim 超限

```
coreDim=XXXX can't be greater than UINT16_MAX
```

**原因**：grid 大小 > 65535。
**解决**：
1. 增大 `BLOCK_SIZE` 使得 `grid = ceil(N / BLOCK_SIZE) ≤ 65535`
2. 设置 `export TRITON_ALL_BLOCKS_PARALLEL=1`

### 不支持的数据类型

```
error: unsupported type: uint64 / float64
```

**解决**：
```python
# uint64 → int32
pos = tl.load(pos_ptr + idx).to(tl.int32)

# float64 → float32
r = tl.load(r_ptr + idx).to(tl.float32)
```

### 链式布尔运算

```
error: chained boolean operator not supported
```

**解决**：
```python
# 错误
use = a or b or c

# 正确
use = (a or b) or c
```

## 2. 运行时错误

### Kernel 启动失败

**检查项**：
1. grid 大小是否 ≤ 物理核数
2. 输入 tensor 是否在 NPU 上（`.npu()`）
3. 输入 tensor 是否 contiguous
4. 数据类型是否匹配 kernel 参数

### 精度不对

**排查步骤**：
1. `export TRITON_INTERPRET=1` 在 CPU 上跑，对比 PyTorch 参考实现
2. 检查 `float32` 中间精度是否丢失（特别是 sum/mean/rsqrt）
3. 检查 `mask` 和 `other` 是否正确处理了边界
4. 检查 `tl.arange` 范围和 stride 计算
5. 用 `tl.device_print`（需 `export TRITON_DEVICE_PRINT=1`）打印中间值

### tl.device_print 不输出

**原因**：kernel 的标量参数数量（含隐藏参数 gridx/gridy/gridz）必须为偶数。
**解决**：添加一个 dummy 参数使总数为偶数。

## 3. 性能问题

### 性能远低于预期（性能比 > 3）

**排查顺序**：
1. **Grid 太小？** 检查是否用了 `get_vectorcore_num()` 而非固定小数
2. **无循环？** NPU 必须有 `tl.range` 核内循环
3. **BLOCK 太小？** scalar 开销占比大 → 增大 BLOCK
4. **数据不连续？** 非 contiguous tensor 导致离散搬运 → `.contiguous()`
5. **未对齐？** 检查地址是否 32B/512B 对齐

### multibuffer 无效果

**可能原因**：
1. tile 太大，double buffer 后超 UB → 减小 tile
2. 算子本身计算量极少，搬运时间也很短 → 融合多个算子
3. 编译器已自动优化 → 查看 pipeline diagram 确认

### Autotuning 选择了次优配置

```bash
export TRITON_PRINT_AUTOTUNING=1
# 查看选择了哪个 config
```

检查 `key` 参数是否正确覆盖了影响性能的维度。

## 4. 调试环境变量速查

| 变量 | 值 | 用途 |
|------|-----|------|
| `TRITON_INTERPRET` | 1 | CPU 执行（精度基准） |
| `TRITON_DEVICE_PRINT` | 1 | 启用 `tl.device_print` |
| `MLIR_ENABLE_DUMP` | 1 | 输出 MLIR IR（定位 90% 编译问题） |
| `TRITON_DEBUG` | 1 | 详细调试 dump |
| `TRITON_DISABLE_CACHE` | 1 | 禁用编译缓存（确保重新编译） |
| `TRITON_ALWAYS_COMPILE` | 1 | 强制重新编译 |
| `TRITON_ALL_BLOCKS_PARALLEL` | 1 | 解决 coreDim 超限 |
| `TRITON_PRINT_AUTOTUNING` | 1 | 打印 autotuning 结果 |
| `TRITON_BENCH_METHOD` | npu | NPU benchmark |
| `TRITON_DISABLE_LINE_INFO` | 0 | 启用代码级热点（profiling 需要） |
| `MLIR_ENABLE_TIMING` | 1 | 编译耗时统计 |
| `TRITON_KERNEL_DUMP` | 1 | 保存 kernel 代码到磁盘 |
| `TRITON_DUMP_DIR` | 路径 | kernel dump 输出路径 |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | 1 | 失败时跳过编译阶段 |

## 5. 编译产物位置

| 文件 | 路径 | 用途 |
|------|------|------|
| 编译缓存 | `~/.triton/cache/` | 避免重复编译 |
| `.ttadapter` | `~/.triton/cache/` | Triton→Ascend 适配 |
| `.ttir` | `~/.triton/cache/` | 适配后 IR |
| `.so` | `~/.triton/cache/` | 最终可执行库 |
| Dump | `~/.triton/dump/` | MLIR 和 C++ 源码 |
