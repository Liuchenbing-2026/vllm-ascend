# F2.1：A2 EP 走 all2all

- 父 spec：`2026-05-25-vllm-ascend-a2-adapt-umbrella-design.md`
- 候选选定：A（复用 A3 的 `AlltoAllCommImpl`，先信任复用，运行期报错再 fork）
- Commit：`feat(a2/moe): add A2 all2all comm method`
- User 估收益：+20%

## 1. 目标

A2 上的 MoE EP 模式当前 `select_moe_comm_method` 只产 `MC2` 或 `ALLGATHER`（`ascend_forward_context.py:272-282`）。Feature 2.1 让 A2 在 `a2_adapt_config.moe_comm = "alltoall"` 或 `auto` + 大 prefill 时返回 `MoECommType.ALLTOALL`，并验证现有 `AlltoAllCommImpl` 系列在 A2 上可跑。

## 2. 当前 main 的相关入口

| 文件:行 | 内容 |
|---|---|
| `vllm_ascend/ascend_forward_context.py:233-319` | `select_moe_comm_method` —— 入口，按 device 4 路 elif |
| `vllm_ascend/ascend_forward_context.py:272-282` | A2 elif（当前只产 MC2/ALLGATHER） |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py:55-62` | `setup_moe_comm_method` —— `ep_size > 1` 时已注册 ALLTOALL → `AlltoAllCommImpl`（A2 已注册） |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py:PrepareAndFinalizeWithAll2All` | 现 ALLTOALL 用的 prepare_finalize（device 假设待验证） |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py:TokenDispatcherWithAll2AllV` | 现 ALLTOALL 用的 token dispatcher |

## 3. 候选方案

| 候选 | 实施 | 取舍 |
|---|---|---|
| A. **复用现有 ALLTOALL 类**（选定） | A2 elif 增 alltoall 决策；setup 不动；运行期遇 A3-only API 报错时再 fork 到 `comm_a2.py` | 代码量最小、复用 main 演进；前提是 `AlltoAllCommImpl` 是 device-agnostic（实施时 grep 验证） |
| B. 新写 `AlltoAllA2CommImpl` + `TokenDispatcherWithAll2AllVA2` | 完全 A2 fork 到 `comm_a2.py` / `token_dispatcher_a2.py` | 物理隔离；代码 + 维护翻倍 |

候选 A 选定。

## 4. 实施细节

### 4.1 `select_moe_comm_method` A2 elif 改写

`vllm_ascend/ascend_forward_context.py:272-282` 改为：

```python
elif soc_version in {AscendDeviceType.A2}:
    a2_moe = get_ascend_config().a2_adapt_config.moe_comm
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size = (
        vllm_config.parallel_config.world_size_across_dp
        // vllm_config.parallel_config.pipeline_parallel_size
    )
    num_experts_per_device = num_experts // ep_world_size

    if a2_moe == "alltoall":
        moe_comm_type = MoECommType.ALLTOALL
    elif a2_moe == "auto" and num_tokens > mc2_tokens_capacity and ep_world_size >= 8:
        # prefill 大 batch：A2 上 all2all 比 allgather+per-expert 切分快
        moe_comm_type = MoECommType.ALLTOALL
    elif num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
        moe_comm_type = MoECommType.MC2
    else:
        moe_comm_type = MoECommType.ALLGATHER
```

后续 F2.2 / F2.3 / F2.4 在本分支之上 append 各自的 elif（不破坏顺序）。

### 4.2 setup_moe_comm_method 不动

`vllm_ascend/ops/fused_moe/moe_comm_method.py:56` 已在 `ep_size > 1` 时给所有 device 注册 ALLTOALL → `AlltoAllCommImpl(moe_config)`。A2 已包含其中。

### 4.3 device-agnostic 验证

实施时 grep 以下 3 个类是否有 `soc_version != A2` 或 `assert A3` 类 device assertion：

- `AlltoAllCommImpl`（`vllm_ascend/ops/fused_moe/moe_comm_method.py`）
- `TokenDispatcherWithAll2AllV`（`vllm_ascend/ops/fused_moe/token_dispatcher.py`）
- `PrepareAndFinalizeWithAll2All`（`vllm_ascend/ops/fused_moe/prepare_finalize.py`）

若有：第一次 fork 到 `comm_a2.py`（class `AlltoAllA2CommImpl`），把 device guard 改成 A2 白名单。
若无：复用，0 改动。

### 4.4 UT 占位

`tests/ut/ops/fused_moe/test_a2_alltoall_select.py`：

- monkey-patch `get_ascend_device_type()` → `AscendDeviceType.A2`
- monkey-patch `get_ascend_config().a2_adapt_config.moe_comm = "alltoall"`
- 构造 vllm_config（ep_world_size=8, num_tokens=1024, mc2_tokens_capacity=512）
- 调 `select_moe_comm_method(num_tokens=1024, vllm_config)` → 断言 `MoECommType.ALLTOALL`
- 同 fn `a2_moe = "auto"` + `num_tokens=200`（小于 capacity） → 断言不返回 ALLTOALL

## 5. Error handling

- `a2_moe` 取值非法 → `AscendConfig.__init__` 在 `A2AdaptConfig.__post_init__` 抛 `ValueError`（spec 4.4 已定义）。
- 运行期 HCCL `all_to_all_single` 失败 → 不接管；用户在 192 实测时看 log 反馈。
- 4.3 grep 发现 device assertion：本 commit 内 fork 到 `comm_a2.py`，spec 不预先决定。

## 6. 改动文件清单

- `vllm_ascend/ascend_forward_context.py`（A2 elif 改写）
- 可能 `vllm_ascend/ops/fused_moe/comm_a2.py`（fork 备用，4.3 决定）
- 可能 `vllm_ascend/ops/fused_moe/token_dispatcher_a2.py`（fork 备用）
- `tests/ut/ops/fused_moe/test_a2_alltoall_select.py`（新）

## 7. 验收

- AST + ruff 通过
- UT 通过（pytest -sv tests/ut/ops/fused_moe/test_a2_alltoall_select.py）
- commit message 附 dry-run 结果
