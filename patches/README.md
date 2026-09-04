# 这两个补丁不需要 —— 留作记录

排查图捕获失败时，我按驱动报错的字面意思

```
Not_Supported(EE1016): ... The current thread is in the capture state and the current
operation cannot be performed ... This operation is supported only in the RELAXED mode.
rtMemcpy execution failed, reason=operation not permitted when a stream is capturing...
```

判断成「在 ACL graph 捕获期间分配 pinned host 内存是非法的」，改了 vllm-ascend 两处：

### 补丁 A —— `vllm_ascend/worker/block_table.py:283-288`

```python
def commit_block_table(self, num_reqs: int) -> None:
    self.block_table.gpu[:num_reqs].copy_(
        self.block_table.cpu[:num_reqs].clone().pin_memory(),   # 每次调用都新分配 pinned
        non_blocking=True,
    )
```
改成 `self.block_table.copy_to_gpu(num_reqs)`
（`CpuGpuBuffer.cpu` 构造时就已经是 pinned，`.clone().pin_memory()` 是多余的）。

### 补丁 B —— `vllm_ascend/worker/utils.py:15-18`

```python
def copy_snapshot_to_gpu(buffer: CpuGpuBuffer) -> torch.Tensor:
    cpu_snapshot = buffer.cpu.clone().pin_memory()
    return buffer.gpu.copy_(cpu_snapshot, non_blocking=True)
```
改成用一个挂在 buffer 上的持久 pinned 暂存区（保留 snapshot 语义，只分配一次）。
该 helper 有 15 处调用，含 `_dummy_run` 与 `_pad_query_start_loc_for_fia`。

---

## 为什么说不需要

做了单变量 A/B：

| 补丁 | memlock | 默认 PIECEWISE 捕获 |
|---|---|---|
| 打上 | unlimited | ✓ 40 s |
| **还原** | unlimited | **✓ 9 s** |
| 打上 | 64 KB (默认) | ✗ |
| 还原 | 64 KB (默认) | ✗ |

**决定性的只有 `--ulimit memlock=-1`。** 那条 EE1016 是驱动在
`aclrtMallocHostWithCfg` 失败时一并吐出的次要信息，真正的失败是撞了容器 64 KB 的
max-locked-memory rlimit。

## 补丁 B 仍有一个理论上的论点（未验证 [U]）

在**被捕获的图**里，host 源地址必须稳定 —— 图记录的是地址，重放时若该 pinned 缓冲区
已被回收，读到的就是垃圾。`.clone().pin_memory()` 每次给一个新地址，理论上不适合捕获路径。

但：CachingHostAllocator 会把释放推迟到拷贝完成，实际是否会出问题**没有证据**，
而且实测未打补丁一切正常（正确性、APC 命中、MTP 接受率都对）。
**在拿到反例之前不要上这两个补丁。**
