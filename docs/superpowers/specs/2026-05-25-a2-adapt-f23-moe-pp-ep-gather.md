# F2.3：A2 上 PP + ep_gather

- 父 spec：`2026-05-25-vllm-ascend-a2-adapt-umbrella-design.md`
- 候选选定：A（调 vllm 主仓 `deep_gemm_utils.ep_gather`，failure path = RuntimeError）
- Commit：`feat(a2/moe): add A2 PP+ep_gather comm method`
- User 估收益：+20%
- 备注：A3 跨超节点 PP+MoE 大融合算子的前置 dry-run

## 1. 目标

A2 上 PP > 1 时让 MoE EP 走 `ep_gather`（vllm 主仓 `model_executor/layers/fused_moe/deep_gemm_utils.ep_gather`），替原 all_gather + permute 组合。本 feature 是 F2.4 的依赖前置（F2.4 内组合 ep_gather + dispatch_combine）。

## 2. 当前 main 的相关入口

| 文件:行 | 内容 |
|---|---|
| `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | vllm 主仓 `ep_gather` 函数（vllm-ascend 当前无调用） |
| `vllm_ascend/ascend_forward_context.py:select_moe_comm_method` | 当前选择层不区分 PP；F2.3 在 A2 elif 内按 `pp_size > 1` 触发 |
| `vllm.config.ParallelConfig.pipeline_parallel_size` | 入口判断 |

## 3. 候选方案

| 候选 | 实施 | 取舍 |
|---|---|---|
| A. **调 vllm 主仓 `deep_gemm_utils.ep_gather`**（选定） | `comm_a2.py:PpEpGatherA2CommImpl` import + 调；deep_gemm 不可用时 RuntimeError | 用上游既有 API；和 F2.4 衔接自然 |
| B. 手写 all_gather + permute 等价 | 自己拼 torch.distributed.all_gather + index permute | 不依赖 deep_gemm；丢 fusion 收益；偏离 user 字面意图 |

候选 A 选定。

## 4. 实施细节

### 4.1 setup_moe_comm_method 增 A2 + PP 注册

`vllm_ascend/ops/fused_moe/moe_comm_method.py`：

```python
if get_ascend_device_type() == AscendDeviceType.A2:
    pp_size = get_current_vllm_config().parallel_config.pipeline_parallel_size
    if pp_size > 1:
        _MoECommMethods[MoECommType.PP_EP_GATHER] = PpEpGatherA2CommImpl(moe_config)
```

（注：`moe_config` 是否带 pp_size 字段实施时确认；若无，按上述 `get_current_vllm_config()` 取。）

### 4.2 新写 `comm_a2.py:PpEpGatherA2CommImpl`

```python
class PpEpGatherA2CommImpl(MoECommMethod):
    """A2-only: PP-aware MoE comm using deep_gemm_utils.ep_gather."""

    def __init__(self, moe_config):
        super().__init__(moe_config)
        try:
            from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather
            self._ep_gather = ep_gather
        except ImportError as e:
            raise RuntimeError(
                "A2 PP+ep_gather requires vllm.model_executor.layers.fused_moe."
                f"deep_gemm_utils.ep_gather; import failed: {e}. "
                "Use a2_adapt_config.moe_comm='alltoall' as fallback."
            )

    def _get_token_dispatcher(self):
        return TokenDispatcherWithEpGatherA2(self.moe_config)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithEpGatherA2(self.moe_config, ep_gather=self._ep_gather)
```

新写 `TokenDispatcherWithEpGatherA2(MoETokenDispatcher)` 和 `PrepareAndFinalizeWithEpGatherA2(PrepareAndFinalize)`（参考 `TokenDispatcherWithAllGather` + `PrepareAndFinalizeWithAllGather` 结构，把 all_gather 替为 `ep_gather`）。

### 4.3 select_moe_comm_method A2 elif 扩

在 F2.2 写过的 A2 elif 上 append（保持 F2.1/F2.2/F2.3 顺序）：

```python
elif soc_version in {AscendDeviceType.A2}:
    ...
    pp_size = vllm_config.parallel_config.pipeline_parallel_size

    if a2_moe == "alltoall":
        moe_comm_type = MoECommType.ALLTOALL
    elif a2_moe == "dispatch_combine":
        moe_comm_type = MoECommType.DISPATCH_COMBINE
    elif a2_moe == "pp_ep_gather":
        if pp_size <= 1:
            logger.warning("a2_moe='pp_ep_gather' requires PP>1; falling back to ALLGATHER")
            moe_comm_type = MoECommType.ALLGATHER
        else:
            moe_comm_type = MoECommType.PP_EP_GATHER
    elif a2_moe == "auto" and pp_size > 1 and num_tokens <= mc2_tokens_capacity and ep_world_size > 1:
        # F2.3 在 PP>1 + 小-中 batch 时优先（F2.4 的 PP_FUSED 在 F2.4 commit 时会插到本判断之前）
        moe_comm_type = MoECommType.PP_EP_GATHER
    elif a2_moe == "auto" and ep_world_size <= 32 and bs_min < num_tokens <= bs_max and not is_draft_model:
        moe_comm_type = MoECommType.DISPATCH_COMBINE
    elif a2_moe == "auto" and num_tokens > mc2_tokens_capacity and ep_world_size >= 8:
        moe_comm_type = MoECommType.ALLTOALL
    elif num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
        moe_comm_type = MoECommType.MC2
    else:
        moe_comm_type = MoECommType.ALLGATHER
```

### 4.4 UT 占位

`tests/ut/ops/fused_moe/test_a2_pp_ep_gather_select.py`：

- device=A2 + `a2_moe="auto"` + `pp_size=2` + `ep_world_size=4` + `num_tokens=256`（<= capacity）→ 断言 `PP_EP_GATHER`
- device=A2 + `a2_moe="pp_ep_gather"` + `pp_size=1` → 断言 `ALLGATHER` + log warning（用 pytest `caplog`）
- device=A3 + `a2_moe="pp_ep_gather"` → 不返回 `PP_EP_GATHER`

## 5. Error handling

- vllm 主仓缺 `deep_gemm_utils.ep_gather`（旧版 vllm）：`PpEpGatherA2CommImpl.__init__` 抛 RuntimeError，提示 user 用 alltoall 兜底。
- pp_size = 1 但强制 `a2_moe="pp_ep_gather"`：log warning + 返回 ALLGATHER（select 函数兜底，避免实际跑出 PP_EP_GATHER）。

## 6. 改动文件清单

- `vllm_ascend/ascend_forward_context.py`（A2 elif 加 pp_ep_gather 决策）
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`（setup_moe_comm_method 加 A2 + PP 注册）
- `vllm_ascend/ops/fused_moe/comm_a2.py`（追加 `PpEpGatherA2CommImpl` + `PrepareAndFinalizeWithEpGatherA2`）
- `vllm_ascend/ops/fused_moe/token_dispatcher_a2.py`（新文件，`TokenDispatcherWithEpGatherA2`）
- `tests/ut/ops/fused_moe/test_a2_pp_ep_gather_select.py`（新）

## 7. 验收

- AST + ruff 通过
- UT 通过
- `python -c "from vllm_ascend.ops.fused_moe.comm_a2 import PpEpGatherA2CommImpl"` 不报错（不实例化）
- commit message 附 dry-run 结果

## 8. Known unknown

- A2 上 deep_gemm 后端可用性：deep_gemm 通常优先 CUDA；A2 上 vllm-ascend 是否 stub 处理 user 待在 191 实测。spec 假设可用，失败 path 走 RuntimeError → user 切换到 alltoall。
- vllm 最低版本（含 `deep_gemm_utils.ep_gather`）：实施时 grep `vllm` clone 确认 import 路径准确；若主仓 path 变化，调整 import。
