# F01：A2 上 dsa_cp 关 all2all

- 父 spec：`2026-05-25-vllm-ascend-a2-adapt-umbrella-design.md`
- 候选选定：A（强制关 CP）
- Commit：`feat(a2): disable all2all on dsa_cp for A2 devices`

## 1. 目标

A2 device（NpuArch 220 / 910B 系列）上让 DeepSeek Sparse Attention + Context Parallel（DSA-CP）不走任何 `all_to_all_single` 通信。`dist.all_to_all_single` 在 A2 HCCL 上有性能/稳定性问题，user 反馈 A2 上 dsa_cp 目前跑不起来。

## 2. 当前 main 的 all2all 实际落点

| 文件:行 | 代码 | 备注 |
|---|---|---|
| `vllm_ascend/attention/context_parallel/attention_cp.py:881` | `dist.all_to_all_single(dcp_context_attn_output, local_context_attn_output, group=self.dcp_group)` | DCP context attn 输出 all2all |
| `vllm_ascend/attention/sfa_v1.py:765` | `torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)` | SFA TP all2all（SFA = sparse flash attention，属 DSA 家族） |
| `vllm_ascend/attention/context_parallel/attention_cp.py:984` | 注释提及 chunked prefill 的 all2all 分支 | 该分支被 `dcp_size > 1` 守护 |

## 3. 候选方案

| 候选 | 实施 | 取舍 |
|---|---|---|
| A. 强制关 CP（**选定**） | A2 上 `dcp_size = pcp_size = 1`；881 / 984 的 `if dcp_size > 1:` 分支自然 dead；sfa_v1.py:765 加 `dcp_size > 1` guard | 实现集中、改动少；丢 A2 长 context CP 切分能力 |
| B. 用 all_gather + slice 等价替换 | 新 `attention_cp_a2.py:dcp_all_gather_then_slice`；替换 881 / 765 调用 | 保留 CP 语义；代码量大、需要严格测试正确性 |

候选 A 选定：user "关掉" 字眼明确；A2 上 DSA-CP 不是主用例；先稳态，后续如需 A2 长 context CP 再做 B。

## 4. 实施细节

### 4.1 PCP/DCP group 初始化处加 A2 guard

`vllm_ascend/distributed/parallel_state.py`（PCP / DCP group 初始化函数处；实施时 grep `init_decode_context_model_parallel_group` / `init_pcp_group` 定位准确入口）。引入 `_a2_disable_cp_if_needed(dcp_size, pcp_size) -> tuple[int, int]`：

```python
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

def _a2_disable_cp_if_needed(dcp_size: int, pcp_size: int) -> tuple[int, int]:
    if get_ascend_device_type() != AscendDeviceType.A2:
        return dcp_size, pcp_size
    a2_cfg = get_ascend_config().a2_adapt_config
    if not a2_cfg.dsa_cp_disable_all2all:
        return dcp_size, pcp_size
    if dcp_size > 1 or pcp_size > 1:
        logger.warning(
            "A2 device + dsa_cp_disable_all2all: forcing dcp_size/pcp_size=1 "
            "(was dcp=%d pcp=%d). DSA-CP all2all is disabled on A2.",
            dcp_size, pcp_size,
        )
    return 1, 1
```

在 group size 写入 group 创建参数前调一次此函数。

### 4.2 sfa_v1.py 加 dcp_size guard

`vllm_ascend/attention/sfa_v1.py:765` 上方加：

```python
dcp_size = get_decode_context_model_parallel_world_size()
if dcp_size > 1:
    attn_output = torch.empty_like(send)
    torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)
else:
    attn_output = send
```

A2 上 `dcp_size = 1`（来自 4.1 的强制），跳过 all2all。A3 上 dcp_size 不变，行为零变化。

### 4.3 envs.py 新增

```python
"VLLM_ASCEND_A2_DSA_CP_DISABLE_ALL2ALL": lambda: int(os.getenv("VLLM_ASCEND_A2_DSA_CP_DISABLE_ALL2ALL", 1)),
```

### 4.4 ascend_config.py 新增 A2AdaptConfig dataclass（在 infra commit 落）

```python
@dataclass
class A2AdaptConfig:
    dsa_cp_disable_all2all: bool = True
    moe_comm: str = "auto"   # "auto"|"alltoall"|"dispatch_combine"|"pp_ep_gather"|"pp_fused"|"none"
    dispatch_combine_bs_min: int = 1
    dispatch_combine_bs_max: int | None = None   # None = 运行期取 mc2_tokens_capacity

    def __post_init__(self):
        valid = {"auto", "alltoall", "dispatch_combine", "pp_ep_gather", "pp_fused", "none"}
        if self.moe_comm not in valid:
            raise ValueError(f"a2_adapt_config.moe_comm must be one of {valid}, got {self.moe_comm!r}")
```

### 4.5 UT 占位

`tests/ut/attention/test_dsa_cp_a2_disable.py`：

- monkey-patch `get_ascend_device_type` 返回 `AscendDeviceType.A2`
- monkey-patch `get_ascend_config().a2_adapt_config.dsa_cp_disable_all2all = True`
- 调 `_a2_disable_cp_if_needed(dcp_size=4, pcp_size=2)` → 断言 `(1, 1)`
- 同 fn device=A3 → 断言 `(4, 2)`

## 5. Error handling

- user 在 A2 上手动 set `additional_config.parallel.dcp_size=4` 且 `dsa_cp_disable_all2all=True`：log warning + 强制 1, 1，不抛异常。
- A3/A5/_310P：guard 函数 early return 原值，行为零变化。

## 6. 改动文件清单

- `vllm_ascend/envs.py`（+1 env）
- `vllm_ascend/ascend_config.py`（A2AdaptConfig dataclass，在 infra commit 已落，本 commit 只 reference）
- `vllm_ascend/distributed/parallel_state.py`（+ `_a2_disable_cp_if_needed` + 调用）
- `vllm_ascend/attention/sfa_v1.py`（765 行加 dcp_size guard）
- `tests/ut/attention/test_dsa_cp_a2_disable.py`（新文件，dummy UT）

## 7. 验收

- `python -c "import ast; ast.parse(open('vllm_ascend/distributed/parallel_state.py').read()); ast.parse(open('vllm_ascend/attention/sfa_v1.py').read())"` 不报错
- `ruff check vllm_ascend/distributed/parallel_state.py vllm_ascend/attention/sfa_v1.py vllm_ascend/envs.py` 通过
- commit message 末附 dry-run import 结果
