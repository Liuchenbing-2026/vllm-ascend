# GLM5 PP2 + KV Pool 修复方案

> **Bug**：`P 侧 PP=2 + KV Pool（AscendStoreConnector producer-put）` 在 prefix warmup 时报 `TRANSFER_FAIL / INTERNAL_ERROR`，所有失败 key 含 `@pp_rank:1`
> **Bug 文档**：[`GLM5/bugfix/PP_KVPOOL/bug.md`](https://github.com/FutureSkyFly/Model_test/blob/main/GLM5/bugfix/PP_KVPOOL/bug.md)
> **base**：`vllm-project/vllm-ascend:main` @ `44312516`（"Revert Layerwise KV Pooling" 之后的状态）
> **vllm**：`0.23.0`
> **工作分支**：`FutureSkyFly/vllm-ascend:pp_kvpool_fix`

---

## 一、根因（grep 验证后的确定结论）

### 1.1 `partitions` **只在消费者侧**被构造

`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:153-178`：

```python
partitions = None
if self.kv_role == "kv_consumer" and self.consumer_is_to_put:
    # 只有 kv_consumer + consumer_is_to_put 才会读
    #   prefill_pp_size / prefill_pp_layer_partition
    # 构造 partitions
    ...
```

**P 侧 `kv_role == "kv_producer"` 完全走不进这个分支**，所以 `token_database.partitions = None`，PP 视角信息丢失。

### 1.2 PP 适配函数 **只在消费者侧**被调用

`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:381-391`：

```python
if self.kv_role == "kv_consumer":
    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
        keys, addrs, sizes, kv_cache_group_id=group_id,
    )

if current_event is not None:
    current_event.synchronize()
self.m_store.put(keys, addrs, sizes)
```

`_decode_adaptor_prefill_pp` 的语义（`config_data.py:412-434`）：**把一个 `@pp_rank:0` 的 key 拆成 N 个 `@pp_rank:0..N-1` 的 key，addr/size 按 partitions 切 N 段**。

P 侧拿不到这条 PP-aware 路径，所以每个 PP rank 的 worker 各自 put 各自的局部 KV，但 **没有任何机制让 D 侧能把 P 侧的 PP=2 切片重新合并回 D 侧 PP=1 的视角**。

### 1.3 失败时序（结合 bug.md 报错链）

```
P 侧 PP1 worker
  ├─ register_kv_caches()
  │   └─ 把 layers[40:78] 的 KV 注册到 Mooncake，segment name 与 PP0 不同
  ├─ KVCacheStoreSendingThread._handle_request()
  │   ├─ keys 含 "@pp_rank:1"
  │   ├─ kv_role == "kv_producer" → 不走 _decode_adaptor_prefill_pp
  │   └─ m_store.put(keys=[@pp_rank:1 ...], addrs, sizes)
  └─ Mooncake TRANSFER_FAIL ← 这里炸
```

报错是 Mooncake C++ 层的 `client_service.cpp:1181 / 1320`，含义是 **transfer engine 在 put 时无法把 buffer / segment 关系建立起来**。

### 1.4 P 侧 lookup 假设了「所有 pp_rank 都已 put」

`pool_worker.py:1056-1062`（lookup 路径已经这么写了）：

```python
pp_base_keys = multi_tp_keys.copy()
for i in range(1, self.pp_size):
    for item in pp_base_keys:
        new_str = item.replace("@pp_rank:0", f"@pp_rank:{i}", 1)
        multi_tp_keys.append(new_str)
```

意思是 lookup 时 **会去查 `@pp_rank:0..N-1` 全部 pp 视角的 key**。如果 put 路径不保证这些 key 都存在，lookup 就会假阴/失败。

---

## 二、修复方向（按 bug.md 指引 + 代码现状）

> Bug 文档原话："Keep P-side AscendStoreConnector producer-put semantics, and add PP-aware key/layer/address adaptation before MooncakeBackend.put()."

把消费者侧的 PP-aware 适配**对称化到生产者侧**。

### 2.1 三层改动

| 层 | 改动 | 文件 |
|---|---|---|
| ① 配置 | 让 P 侧也构造 `partitions`（从 `prefill_pp_size` + `prefill_pp_layer_partition` 读）| `pool_worker.py:153-178` |
| ② 生产者侧 put 适配 | 新增 `prefill_adaptor_producer_pp()`，put 前按 partition 切 key/addr/size，**只 put 自己 PP rank 对应的那段** | `config_data.py` + `kv_transfer.py:381` |
| ③ 注册路径校核 | 确保 P 侧每个 PP worker 注册到 Mooncake 的 segment 与 put 时的 key 一致 | `pool_worker.py:register_kv_caches` |

### 2.2 改动 ① 代码骨架

`pool_worker.py:153-178` 改成：

```python
partitions = None
need_pp_partitions = (
    (self.kv_role == "kv_consumer" and self.consumer_is_to_put)
    or (self.kv_role == "kv_producer" and self.pp_size > 1)   # ← 新加这一支
)
if need_pp_partitions:
    num_hidden_layers = model_config.hf_text_config.num_hidden_layers
    partition_list_str = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
        "prefill_pp_layer_partition", None
    )
    # P 侧 prefill_pp_size 等于 self.pp_size；D 侧从 extra_config 读
    if self.kv_role == "kv_producer":
        prefill_pp_size = self.pp_size
    else:
        prefill_pp_size = int(
            vllm_config.kv_transfer_config.kv_connector_extra_config.get("prefill_pp_size", 1)
        )
    # 后续解析 partitions 不变
    ...
```

**注意**：现在没有 `VLLM_PP_LAYER_PARTITION` 与 `prefill_pp_layer_partition` 的强一致校验——bug.md 的启动脚本里两者都设 `40,38`，但应当在 `pool_worker` 启动时 assert 一次，防止用户漏配。

### 2.3 改动 ② 代码骨架

`config_data.py` 新增（与 `decode_adaptor_prefill_pp` 对偶）：

```python
def prefill_adaptor_producer_pp(
    self,
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    kv_cache_group_id: int = 0,
    cache_role: str = "kv",
    pp_rank: int = 0,
):
    """P 侧 PP > 1 时的 producer-put 适配。

    当前 worker 仅持有 layers[start : end]，其中
        start = sum(partitions[:pp_rank])
        end   = start + partitions[pp_rank]
    所以 put 时:
      - key 改成 `@pp_rank:{pp_rank}`（已经是当前 rank，正常情况下无需改）
      - addr/size 只保留属于本 pp_rank 的那段 layer
      - 让 D 侧的 `_decode_adaptor_prefill_pp` 能按 partitions 把这些片拼回完整 KV
    """
    if self.partitions is None or len(self.partitions) == 1:
        return keys, addrs, sizes

    group_num_layers = self.group_num_layers.get(cache_role, {}).get(kv_cache_group_id, 0)
    if not group_num_layers:
        return keys, addrs, sizes

    new_keys, new_addrs, new_sizes = [], [], []
    layer_start = sum(self.partitions[:pp_rank])
    layer_end = layer_start + self.partitions[pp_rank]

    for i, (addr_list, size_list) in enumerate(zip(addrs, sizes)):
        caches_per_layer = max(len(addr_list) // group_num_layers, 1)
        a_start = layer_start * caches_per_layer
        a_end = layer_end * caches_per_layer
        new_keys.append(keys[i])               # key 已是本 rank 视角，不改
        new_addrs.append(addr_list[a_start:a_end])
        new_sizes.append(size_list[a_start:a_end])

    return new_keys, new_addrs, new_sizes
```

`kv_transfer.py:381-391` 改成：

```python
if self.kv_role == "kv_consumer":
    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
        keys, addrs, sizes, kv_cache_group_id=group_id,
    )
elif self.kv_role == "kv_producer" and getattr(self, "pp_size", 1) > 1:
    keys, addrs, sizes = self._prefill_adaptor_producer_pp(
        keys, addrs, sizes,
        kv_cache_group_id=group_id,
        pp_rank=self.pp_rank,
    )

if current_event is not None:
    current_event.synchronize()
self.m_store.put(keys, addrs, sizes)
```

这要求 `KVCacheStoreSendingThread` 多两个字段 `pp_size`、`pp_rank`，从 `KVPoolWorker` 传进去。

### 2.4 改动 ③ 校验

`register_kv_caches()` 路径中验证：

1. **PP1 worker 的 segment 名是否与 PP0 冲突**：Mooncake `mooncake.json` 里 `device_name=""`，segment 名通常由 IP + 进程 PID 派生。两个 PP worker 在同一台机器跑，应该 PID 不同 → segment 不同。**需要打 1 行 log 确认**：

   ```python
   logger.info(
       "KV pool register: pp_rank=%d local_segment=%s ptrs=%d total_len=%d",
       self.pp_rank, self.m_store.get_local_segment_name(), len(ptrs), sum(lengths),
   )
   ```

2. **PP1 注册的 buffer 总量是否超过 `global_segment_size=15GB`**：GLM-5.1 80 层，每层 KV cache 大小取决于 `num_blocks × block_size × num_kv_heads × head_dim`。**Phase 0 必须先实测**。

---

## 三、Phase 化实施

| Phase | 内容 | 改动估算 | 依赖 |
|---|---|---:|---|
| **P0** | 实测确认 segment 大小 + 加 register log | +10 | 无 |
| **P1** | `pool_worker.py` 让 P 侧也构造 `partitions` | +30 | P0 |
| **P2** | `config_data.py` 加 `prefill_adaptor_producer_pp` | +50 | P1 |
| **P3** | `kv_transfer.py` 接入 producer-side PP adaptation | +20 | P2 |
| **P4** | `pool_worker.py:get_send_thread` 透传 `pp_size/pp_rank` 给 sending thread | +15 | P3 |
| **P5** | 启动校验：`VLLM_PP_LAYER_PARTITION` 与 `prefill_pp_layer_partition` 一致性 assert | +20 | P4 |
| **P6** | UT + e2e 复现/修复验证 | +200 | P5 |
| **合计** | | **+345** | 1.5–2 周 |

---

## 四、验证矩阵（按 bug.md 实测配置）

| 用例 | 配置 | 期望 |
|---|---|---|
| 复现原始 bug | P=DP4 TP4 PP2 / D=DP8 TP4 PP1，prefix cache + indexcache + dsa_cp + kvpool | TRANSFER_FAIL（修复前）|
| 修复后基线 | 同上，应用 P1-P4 | prefix warmup 不报错，aisbench 64K 输入 64 prompt 完整 RECV=64 |
| 退化检查 1：PP=1 | P=DP4 TP4 PP1 | 行为与修复前一致（new code path 不触发）|
| 退化检查 2：纯 KV Pool 无 PP | 任何 PP=1 + KV Pool | 不变 |
| 退化检查 3：consumer-side | 不变更 D 侧逻辑 | `decode_adaptor_prefill_pp` 路径正常 |
| 边界：partition 不均 | `pp_layer_partition="40,38"` 这种非均分 | 切片正确，sum 校验通过 |
| 边界：MTP draft layer | `--speculative-config` 带 mtp | MTP draft 层的 KV 也应被正确 put |

---

## 五、风险与回滚

| 风险 | 缓解 |
|---|---|
| Mooncake C++ 侧仍存在 buffer-segment 注册问题（不是 Python 层 PP 适配能解决的）| Phase 0 必须先做 Mooncake 侧日志 + segment 大小核实 |
| `prefill_adaptor_producer_pp` 切片错位 → put 了错误的 addr | 加 UT：对一个已知 shape 的 KV，按 partition 切完手算 vs 函数输出对比 |
| `prefill_pp_layer_partition` 用户漏配 | Phase 5 加启动 assert，直接 fail-fast |
| D 侧 `_decode_adaptor_prefill_pp` 与 P 侧改动语义不对偶 | 加 UT：跑 `prefill_adaptor_producer_pp` → `decode_adaptor_prefill_pp` round-trip，验证 key+addr+size 能合回 |
| 旧 bug：P 侧不开 KV Pool 的客户 | 改动加分支判断 `kv_role == "kv_producer" and pp_size > 1`，PP=1 走原路径 |

回滚：所有改动都在 `if pp_size > 1` 守卫下，PP=1 时与原逻辑等价。

---

## 六、决策点（先回答这 3 个才能动手）

1. **PP1 worker 直接 put `@pp_rank:1` 是不是 Mooncake C++ 层的真 bug？**（Phase 0 必须确定）
   - 如果是 Mooncake bug → 上 Mooncake 修；vllm-ascend 这边的 `prefill_adaptor_producer_pp` 改动只能算「workaround」
   - 如果不是 Mooncake bug，仅是 vllm-ascend 没传 partition 信息 → 上面的 3 层改动就够

2. **是让 PP rank > 0 worker 各自 put 自己持有的层，还是统一由 PP rank 0 收集后 put？**
   - **A**（推荐）：各 PP rank 自己 put（避免跨进程 KV 传输开销）
   - B：PP rank 0 收集（语义干净但慢且重构大）

3. **`prefill_pp_size` 是从 vllm-ascend 自动推断（= `self.pp_size`）还是要求 `kv_connector_extra_config` 显式传？**
   - 自动推断少一处配置，但 P/D 配置分离时容易写错
   - **建议自动推断 + 显式覆盖**：默认 `self.pp_size`，可由 `kv_connector_extra_config.prefill_pp_size` 覆盖

---

## 七、改动文件清单速查

| 文件 | 主要改动 |
|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `partitions` 构造扩 P 侧；register log；透传 `pp_size/pp_rank` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py` | 新增 `prefill_adaptor_producer_pp` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVCacheStoreSendingThread` 加 producer-side PP adaptation；构造函数加 `pp_size/pp_rank` |
| `tests/ut/distributed/ascend_store/test_config_data.py` | 加 `test_prefill_adaptor_producer_pp_*` 用例 |
| `tests/ut/distributed/ascend_store/test_kv_transfer.py` | 加 `test_sending_thread_producer_pp_*` 用例 |
| `docs/source/user_guide/feature_guide/kv_pool.md`（如有）| 文档：`prefill_pp_layer_partition` P/D 两侧都要配 |

---

*文档版本*：v1
*作者*：FutureSkyFly
*分支*：`pp_kvpool_fix`（自 `vllm-project/vllm-ascend:main @ 44312516` 派生）
