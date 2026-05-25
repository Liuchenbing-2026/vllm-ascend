# F2.2：A2 上 dispatch & combine

- 父 spec：`2026-05-25-vllm-ascend-a2-adapt-umbrella-design.md`
- 候选选定：A（新增 enum + 独立 impl）
- Commit：`feat(a2/moe): add A2 dispatch_combine comm method`
- User 估收益：+30%
- 备注：本 feature 暴露 `dispatch_ffn_combine` HCCL primitive 单独路径，不是完整 FUSED_MC2（那是 F2.4）；user 标 "算子 bs 有约束"

## 1. 目标

将 A3 上 `FUSED_MC2 mode 1` 内部用的 `dispatch_ffn_combine` HCCL primitive 在 A2 上作为独立 comm 路径暴露，受 bs 约束。

## 2. 当前 main 的相关入口

| 文件:行 | 内容 |
|---|---|
| `vllm_ascend/ascend_forward_context.py:286-298` | A3 elif 的 `fused_mc2_enable` 选择 + `dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model)` |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py:PrepareAndFinalizeWithMC2` | A3 上 FUSED_MC2 用的 prepare_finalize；接近参考实现 |
| `vllm_ascend/ops/fused_moe/comm_utils.py` | 实施时 grep 查 torch_npu API（`npu_dispatch_ffn_combine` 或同义） |

## 3. 候选方案

| 候选 | 实施 | 取舍 |
|---|---|---|
| A. **新增 enum + 独立 impl**（选定） | `MoECommType.DISPATCH_COMBINE`；`comm_a2.py:DispatchCombineA2CommImpl` | 语义清晰；和 F2.4 的 FUSED 解耦；bs 约束在 select 集中校验 |
| B. 复用 FUSED_MC2 + A2 mode 1 | 不增 enum；select A2 elif 加 fused_mc2 mode 1；setup 给 A2 注册 FusedMC2CommImpl | 代码少；语义模糊（2.2 / 2.4 共 enum，难做 ablation） |

候选 A 选定。

## 4. 实施细节

### 4.1 MoECommType.DISPATCH_COMBINE 已在 infra commit 加

参考 umbrella 4.4：`MoECommType.DISPATCH_COMBINE = 4`。

### 4.2 setup_moe_comm_method 增 A2 注册

`vllm_ascend/ops/fused_moe/moe_comm_method.py:55-62` 在 `ep_size > 1` 分支末尾加：

```python
if get_ascend_device_type() == AscendDeviceType.A2:
    _MoECommMethods[MoECommType.DISPATCH_COMBINE] = DispatchCombineA2CommImpl(moe_config)
```

### 4.3 新写 `comm_a2.py:DispatchCombineA2CommImpl`

```python
class DispatchCombineA2CommImpl(MoECommMethod):
    """A2-only: expose dispatch_ffn_combine HCCL primitive as a standalone comm method.

    Mirrors the dispatch+combine portion of A3 FUSED_MC2 mode 1 without the
    full fused-MC2 packaging.
    """

    def _get_token_dispatcher(self):
        # MC2 dispatcher 处理的就是 dispatch+combine 的 token 路由；复用
        return TokenDispatcherWithMC2(self.moe_config)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeDispatchCombineA2(self.moe_config)

    # prepare / mlp_compute / finalize 方法见 PrepareAndFinalizeDispatchCombineA2
```

新写 `PrepareAndFinalizeDispatchCombineA2(PrepareAndFinalize)`：

- `prepare` 调 `torch_npu.npu_dispatch_ffn_combine`（**API 名待 user 在 191 实测确认**；spec 落地时按 `getattr(torch_npu, "npu_dispatch_ffn_combine", None)` 做安全 import + RuntimeError fall back）。
- `finalize` 调反向 combine 部分。
- 实现参考 `PrepareAndFinalizeWithMC2`（同文件），但去掉 fused-MC2 packaging，只用 dispatch + combine 两步。

### 4.4 select_moe_comm_method A2 elif 扩

在 F2.1 写过的 A2 elif 上再 append（保持 F2.1 / F2.2 / F2.3 / F2.4 在同 elif 内顺序判断）：

```python
elif soc_version in {AscendDeviceType.A2}:
    a2_moe = get_ascend_config().a2_adapt_config.moe_comm
    ...
    bs_min = get_ascend_config().a2_adapt_config.dispatch_combine_bs_min
    bs_max = get_ascend_config().a2_adapt_config.dispatch_combine_bs_max
    if bs_max is None:
        bs_max = mc2_tokens_capacity   # 默认按 mc2 capacity

    if a2_moe == "alltoall":
        moe_comm_type = MoECommType.ALLTOALL
    elif a2_moe == "dispatch_combine":
        moe_comm_type = MoECommType.DISPATCH_COMBINE
    elif a2_moe == "auto" and ep_world_size <= 32 and bs_min < num_tokens <= bs_max and not is_draft_model:
        moe_comm_type = MoECommType.DISPATCH_COMBINE
    elif a2_moe == "auto" and num_tokens > mc2_tokens_capacity and ep_world_size >= 8:
        moe_comm_type = MoECommType.ALLTOALL
    elif num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
        moe_comm_type = MoECommType.MC2
    else:
        moe_comm_type = MoECommType.ALLGATHER
```

注：auto 决策顺序刻意把 dispatch_combine（小-中 bs）排在 alltoall（大 prefill）前 —— 因为 dispatch_combine 的 bs 上界 = mc2_tokens_capacity，alltoall 的判定条件是 num_tokens > mc2_tokens_capacity，两者互补不重叠。

### 4.5 UT 占位

`tests/ut/ops/fused_moe/test_a2_dispatch_combine_select.py`：

- device=A2 + `a2_moe="dispatch_combine"` → 断言 `DISPATCH_COMBINE`
- device=A2 + `a2_moe="auto"` + `num_tokens=256`（在 1 ~ capacity 内）+ `ep_world_size=16` → 断言 `DISPATCH_COMBINE`
- device=A3 + `a2_moe="dispatch_combine"`（即使设了也忽略）→ 走 A3 elif，不返回 `DISPATCH_COMBINE`

## 5. Error handling

- `torch_npu.npu_dispatch_ffn_combine` 不存在（API 名不匹配 / 旧版 torch_npu）：`PrepareAndFinalizeDispatchCombineA2.__init__` 抛 `RuntimeError("A2 dispatch_combine requires torch_npu.npu_dispatch_ffn_combine; not found. Check torch_npu version or set a2_adapt_config.moe_comm='alltoall'.")`。
- bs 越界 + user 强制 `a2_moe="dispatch_combine"`：log warning + 兜底 MC2 / ALLGATHER 的逻辑由 select 函数控制（强制设取直接返回 DISPATCH_COMBINE，越界由 impl 内部 raise → user log）。

## 6. 改动文件清单

- `vllm_ascend/ascend_forward_context.py`（A2 elif 加 dispatch_combine 决策）
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`（setup_moe_comm_method 加 A2 注册）
- `vllm_ascend/ops/fused_moe/comm_a2.py`（新文件，`DispatchCombineA2CommImpl` + `PrepareAndFinalizeDispatchCombineA2`）
- `tests/ut/ops/fused_moe/test_a2_dispatch_combine_select.py`（新文件）

## 7. 验收

- AST + ruff 通过
- UT 通过
- `python -c "from vllm_ascend.ops.fused_moe.comm_a2 import DispatchCombineA2CommImpl"` 不报错（无 torch_npu 时跳过初始化）
- commit message 附 dry-run 结果

## 8. Known unknown

- `torch_npu.npu_dispatch_ffn_combine` 实际 API 名 / 签名：user 在 191 上实测时验证；spec 用 `getattr` 安全 import 留接口。
- A2 上 dispatch_ffn_combine 的 bs 上界精确值：默认 = `mc2_tokens_capacity`；user 后续按 HCCL 文档 / 实测调 `a2_adapt_config.dispatch_combine_bs_max`。
