# AscendC 算子适配常见问题排查

## 1. 编译与链接错误

### `aclnn* not found` 运行时错误

```
RuntimeError: aclnnMyNewOp not found
```

**原因 A — 使用了错误的调用模式**：
- `OpCommand` + `Input/Output/Attr` 模式在仓内无先例，行为未验证
- **解决**：改用 `EXEC_NPU_CMD(aclnnMyNewOp, ...)` 或 `SetCustomHandler` 模式

**原因 B — 未完整重编译**：
- 新增 kernel 后 aclnn wrapper 需要自动生成，增量编译可能遗漏
- **解决**：
  ```bash
  pip install -e . --no-build-isolation --force-reinstall
  # 或清理后重编译
  rm -rf build/ && pip install -e . --no-build-isolation
  ```

**原因 C — CMakeLists.txt 未更新**：
- kernel `.cpp` 未加入 `VLLM_ASCEND_CUSTOM_OP` 列表
- **解决**：检查 `CMakeLists.txt` 中是否包含 kernel 路径

### `undefined symbol` 链接错误

```
ImportError: undefined symbol: _ZN11vllm_ascend...
```

**原因**：`torch_binding.cpp` 中 `ops.impl()` 引用的函数签名与实际实现不匹配。
**解决**：
1. 检查 `torch_adpt.h` 中函数签名与 `torch_binding.cpp` 中 `ops.def()` schema 是否一致
2. 注意 `Tensor?` (optional) vs `Tensor` 的区别
3. 注意 `Tensor!` (mutable) 用于 in-place 输出

### `CMake Error: No ASCEND_CANN_PACKAGE_PATH`

```
CMake Error: ascendc_kernel_cmake does not exist
```

**原因**：CANN 包未安装或环境变量未设置。
**解决**：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 验证
echo $ASCEND_HOME_PATH
```

## 2. Tiling 错误

### Kernel 输出全零或乱码

**排查步骤**：
1. 检查 TilingData 新字段是否在 kernel entry point 中被拷贝
2. 检查新字段是否在结构体**末尾**追加（中间插入会打乱偏移）
3. 打印 tiling 值验证：
   ```cpp
   // 在 host 端 tiling 函数中
   printf("mm1OutSize=%u, splitSizeTwo=%u\n",
          tilingData->mm1OutSize, tilingData->splitSizeTwo);
   ```

### Workspace buffer overflow

```
EE9999: Inner Error (Workspace too small)
```

**原因**：新模型的维度更大导致 workspace 不够。
**解决**：检查 `SetWorkSpace()` 中的计算是否使用参数化维度而非硬编码值：
```cpp
// 错误
uint64_t ws = 2112 * sizeof(float);
// 正确
uint64_t ws = tilingData->mm1OutSize * sizeof(float);
```

### TilingKey 路由错误

**原因**：`tilingKey` bitfield 计算未覆盖新模型的 dtype/quant mode 组合。
**解决**：检查 `tilingKey` 构建逻辑中每个 bit 的含义，确保新模型的参数组合能正确路由到对应 kernel 变体。

## 3. Torch 注册错误

### Schema 不匹配

```
RuntimeError: schema mismatch for "_C_ascend::my_op"
```

**原因**：`ops.def()` 声明的参数列表与实际函数签名不匹配。
**检查项**：
1. 参数数量和顺序
2. `Tensor?` vs `Tensor`（optional 用 `c10::optional<at::Tensor>`）
3. `Tensor!` 用于 mutable 输出参数
4. 默认值（`float epsilon=1e-6`）
5. 返回类型元组格式

### Meta 实现 shape 错误

```
RuntimeError: Sizes of tensors must match
```

**原因**：`torch_binding_meta.cpp` 中 meta 函数返回的 shape 与实际输出不一致。
**解决**：meta 函数必须精确计算输出 shape：
```cpp
// 确保 shape 计算与实际实现一致
auto output = at::empty_symint(input.sym_sizes(), input.options());
```

### `torch.compile` graph break

**原因**：算子未注册 meta 实现 → torch.compile 无法 trace。
**解决**：确保 `torch_binding_meta.cpp` 中已注册 meta 实现。

## 4. 精度问题

### 新模型输出偏差大

**排查步骤**：
1. **逐步对比**：在 host 端打印 tiling 参数，确认维度推导正确
2. **单步验证**：隔离各子步骤（RMSNorm、RoPE、MatMul）单独验证
3. **对比参考实现**：用 PyTorch 实现各子步骤的参考值
4. **检查辅助类**：kernel 中的辅助类是否也使用了参数化维度

```python
# Python 端逐步对比
# 1. 验证 Q 投影
q_ref = torch.matmul(hidden_states, wd_qkv)
# 2. 验证 RMSNorm
rms_ref = rms_norm(q_ref[:, :q_lora_rank], gamma1)
# 3. 验证 RoPE
rope_ref = apply_rope(q_pe, cos, sin)
```

### 原模型（DSV3）精度退化

**原因**：参数化修改引入了 bug，影响了默认值路径。
**排查**：
1. 确认 TilingData 默认值完全等于 DSV3 硬编码值
2. 确认 kernel 中所有 `constexpr` → `tilingData->field` 替换无遗漏
3. 运行 DSV3 端到端推理，对比修改前后输出

## 5. 运行时错误

### Kernel launch 失败

```
EE9999: aicore kernel launch failed
```

**排查**：
1. `block_dim` 是否超过物理核数
2. 输入 tensor 是否在 NPU 上且 contiguous
3. 数据类型是否匹配 kernel 预期

### Stream 同步问题

**表现**：结果偶尔不对，重跑又正确。
**原因**：异步 kernel launch 后未同步就读取输出。
**解决**：
```python
# 调试时强制同步
torch.npu.synchronize()
```

## 6. 多模型兼容问题

### 新模型可用但原模型挂了

**原因**：TilingData 字段追加顺序或默认值有误。
**排查**：
1. 确认新字段在结构体**末尾**追加
2. 确认默认值 = 原模型值
3. 确认 Tiling Init() 对所有模型都能正确推导维度
4. 确认原模型的 tensor shape 也能被正确解析

### 部分 kernel 变体失败

**原因**：多个 `.hpp` 变体中只修改了部分。
**解决**：
```bash
# 搜索所有硬编码位置
grep -rn "OLD_VALUE" csrc/<op_name>/op_kernel/*.hpp
# 确保所有变体都已修改
```

## 7. 调试环境变量速查

| 变量 | 用途 |
|------|------|
| `ASCEND_LAUNCH_BLOCKING=1` | 同步执行，方便定位出错 kernel |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | 日志输出到 stdout |
| `TASK_QUEUE_ENABLE=0` | 禁用任务队列，逐个执行 |
| `ACL_OP_DEBUG_LEVEL=3` | 详细算子调试信息 |

## 8. 快速自检 Checklist

- [ ] `torch_binding.cpp` 中 `ops.def()` schema 与 `torch_adpt.h` 函数签名一致
- [ ] `torch_binding_meta.cpp` 中 meta 实现返回正确 shape
- [ ] `CMakeLists.txt` 中包含 kernel `.cpp` 路径
- [ ] TilingData 新字段在结构体末尾追加且带默认值
- [ ] Kernel entry point 中新 tiling 字段被正确拷贝
- [ ] 所有 kernel 变体（`.hpp`）都已同步修改
- [ ] 辅助类中的硬编码也已替换
- [ ] 完整重编译通过
- [ ] 原模型回归测试通过
- [ ] 新模型端到端推理正确
