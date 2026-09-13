# vllm-ascend #16271 — 草稿注意力的阻塞 D2H：问题、原理、解法、收益

## 0. 环境（复现所需的全部坐标）

| 项 | 值 |
|---|---|
| 机器 | A2 `112.29.145.3`（bms-75847414-002），8×Ascend 910B4-1，用卡 4-7 |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:nightly-main`，容器 `nt-dspark-mrv2` |
| **vLLM commit** | **`2cf0a6915ce544dc493a0990f2ea38d81601128a`**（`0.28.0`） |
| **vllm-ascend commit（镜像内运行的树）** | **`e5118d151314ae18e56c0b63aa8dd00d294adc22`**（2026-09-09） |
| 本分支基线 | `feat/16271-approx-draft-kv-opt-in` @ `367a4673` ← `fix/16271-…` @ `adfa304e` ← upstream main `84d6dc83` |
| CANN / torch | 9.1.0 / torch 2.10.0 + torch_npu 2.10.0.post4 |
| 目标模型 | `/home/models/Qwen3.6-35B-A3B`，arch `Qwen3_5MoeForConditionalGeneration` |
| 草稿模型 | `/home/models/Qwen3.6-35B-A3B-speculator.dspark`，arch `Qwen3DSparkModel`，`num_speculative_tokens=8` |
| 并行 | TP4 + EP，`--distributed-executor-backend mp` |
| Runner | **MRV2**（`VLLM_USE_V2_MODEL_RUNNER=1`） |
| 确定性 | `HCCL_DETERMINISTIC=true`（否则贪婪解码不可复现，逐字节对比就没意义） |

⚠️ 镜像内的 vllm-ascend 树（`e5118d15`）与本分支的基线（upstream main `84d6dc83`）
相差 54 个不相关 commit。实测是在镜像树上做的，本分支是同一个改动在 main 上的形态；
两边被改动的位置逐字节相同（`attention_v1.py` 镜像内 md5=`3ffdd3667b206531740ed46c69285f41`，与 upstream main 一致）。

### 怎么跑

```bash
# 1. 起服（卡 4-7，MRV2，DSpark）
MRV2=1 bash harness/serve.sh <tag>        # 完整 vllm serve 命令行在这个脚本里
bash harness/wait_ready.sh <tag> 1800

# 2. 接受率 + 逐字节文本对比（贪婪，并发 1）
python3 harness/accept_probe.py <tag>_p1 256
python3 harness/pooled_acc.py /nt/logs/serve_<tag>.log
python3 harness/text_diff.py /nt/logs/accept_*_p1.json

# 3. 延迟 / 吞吐（第一跑是预热，不计数）
bash harness/bench.sh <tag> warm    48 8
bash harness/bench.sh <tag> measure 96 8

# 4. 开关（两个都默认关，本方案要两个同时开）
export VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1
export VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK=1

# 一键六臂 A/B（就是 §5.2 那张表的来源）
bash harness/leg_mask.sh
```

完整过程与所有中间否定结论见同级目录的 `REPRO.md`（§8.5–§8.7）；
六臂 A/B 的原始日志在 `logs/mask.out`。

---

## 1. 问题点

`vllm_ascend/attention/attention_v1.py` 在构建注意力元数据时有一行：

```python
seq_lens_list = seq_lens.tolist()
```

在 **parallel drafting**（DFlash / DSpark）下，`seq_lens` 是**设备张量**。
NPU 的流是有序的，所以 `.tolist()` 不是在等那次几微秒的拷贝，而是**等整条计算流排空**。

| 量 | 实测 |
|---|---|
| 触发比例（基线） | 20000 / 20000 次 build 全部走 D2H |
| 单次阻塞（队列满时） | **3286 µs**（同条件下纯异步下发是 23 µs，**143 倍**） |
| 累计阻塞 | 118.5 s / 787.6 s 墙钟 |

后果不是算得慢，是**主机下发被串行化**：host 本来领先设备几毫秒在排下一批活，这一行把它拽回来
和设备对齐，设备随即空转。

**为什么这在 MRV2 上特别难办**：草稿 build 需要的 KV 长度是
`num_computed + (K+1) − 这一步的拒绝数`，而**拒绝数要等目标模型前向跑完才在设备上产生**。
MRV1 里 vLLM 为了吐输出本来就把接受的 token id 列表放在 host 上，`rejected` 白拿；
MRV2 专门把那次拷贝设计成与 speculator 重叠（`sampled_token_ids=None`），
**host 上根本不存在这个值**。

---

## 2. 走过的死路（为什么解法长这样）

这三条都实测过，结论是否定的，记在这里是为了说明第 3 节不是"第一个想到的办法"。

| 方向 | 结论 |
|---|---|
| **把 KV 长度以设备张量喂给算子**（issue 原文的方向） | 逐请求回读，比传 list 贵 2.0×/4.7×/8.0×（n=8/32/64）。根因：`npu_fused_infer_attention_score` 的 ATen 签名是 `at::OptionalSymIntArrayRef`，PyTorch 被迫 `Tensor → IntArrayRef` **逐元素 `.item()`** |
| **换个接设备张量的入口** | 不存在。算子 IR 把 `actual_seq_lengths{,_kv}` 标成 **`ValueDepend`**，tiling 必须在宿主读值；GE fallback 也是 `GetData<int64_t>()` 拷回宿主；`aclnnInner…TensorGetWorkspaceSize` 不是设备张量入口，是 `GetMaxWorkspaceSize` 用**数据指针为 null 的假张量**调的 |
| **换 ATB 的 `_npu_paged_attention_splitfuse`** | `context_lens` 放设备直接 `tensor.hostData is null`；四种放置组合只有 cpu/cpu 能跑 |
| **在宿主侧估计拒绝数**（用上一步的值修正） | **剂量-反应单调反向**：残余误差 6.6483 → 3.9149 → **0.0670（砍掉 99%）**，而 accept_len 2.2874 → 2.2770 → **2.2077（−5.53σ，比什么都不做还差）**。代价按误差方向不对称：偏短 −53.3%、偏长 −20.0%，**偏短贵 2.6 倍** |

**所以：宿主侧拿不到精确值，猜也不行，长度本身又必须在宿主。** 看起来是死局。

---

## 3. 原理

### 3.1 关键观察

`actual_seq_lengths_kv` 必须在宿主，**但 `atten_mask` 不是 `ValueDepend` 参数**——
它就是一个普通的设备张量，**可以合法地依赖设备上的 `seq_lens`**。

于是把问题拆成两半：

| 参数 | 喂什么 | 从哪来 | 阻塞吗 |
|---|---|---|---|
| `actual_seq_lengths_kv` | **上界** `U = num_computed + K+1` | 宿主 numpy（调度器算的） | 否 |
| `atten_mask` | 按**真实** `L` 逐请求构造的屏蔽 | 设备张量，比较式生成 | 否 |

真实长度 `L` 全程只存在于设备上。

### 3.2 为什么喂上界会掉接受率，掩码又能补回来

上界 `U = L + d`（`d` = 这一步的拒绝数，实测 `delta == rejected`，`d ∈ [0,8]`，
两台独立仪器闭合到 0.59 SE）。直接喂 `U` 而不加掩码，kernel 会真的去 attend
`[L, U)` 这段——**那正是这一步刚被回滚的草稿 KV**，softmax 被稀释。
这就是现有开关 `VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1` 掉 ~5% 接受率的全部机制。

把这段盖掉，结果就和喂精确长度**逐位同级**：

```
query token j of request i  可见  kv ∈ [0, L_i − Q + j]
mask[i, j, c] = 1  iff  c > L_i − Q + j
```

### 3.3 唯一的结构性障碍：模板选择

直接在 TND 下换 `sparse_mode=0` + 三维掩码会被算子拒绝：

```
CheckFAIMask: "When attnMask is provided, sparseMode must be 3 or 4"
CheckFAIMaskShape: 最后两维必须 2048，前面的维必须 1
```

因为 `IsUsingFAI()` 的**第一个条件就是 `inputLayoutStr == "TND"`**：TND 一律走
split-fuse (FAI) 模板，而那个模板里逐请求掩码是**类别上不可能**的。

**绕法**：parallel drafting 下每个请求的 query 数**恒等于 `K+1`**，
所以 TND `[T, N, D]` 和 BSND `[B, S, N, D]` 是**同一块内存的两个视图**（`view`，零拷贝）。
换标签就离开了 FAI 模板，落到通用模板，而通用模板的 `MaskChecker` 接受

```
sparse_mode = 0,  atten_mask [B 或 1, ≥Q_S, ≥KV_S]     （PA 下 KV_S = maxBlockNumPerSeq × blockSize）
```

### 3.4 一个必须照抄的语义

滑窗（drafter 是 `sliding_window=2048`）也得编进同一个掩码。
实测扫偏移，**band 模式 `pre_tokens=W` 保留的是 W+1 个位置**，`[last−W, last]` 两端都闭：

```
不带窗（对照）               rel = 0.5903
window 2046 / 2047 / 2048 / 2050   0.186 / 0.157 / 0.157 / 0.149
window 2049  (= W+1)               0.005848   ← 唯一对上的
```

写成 `W` 的误差是 **0.157 —— bf16 噪声底的 27 倍，但远不到会崩的程度**，
正是那种能一路混到生产的偏差。所以代码里写的是 `cols < (last - sliding_window)` 而不是 `<=`。

---

## 4. 解决方案

补丁：本分支的代码改动本身（`vllm_ascend/` 下 2 个文件 + 单测）。
基于两条已有分支：`fix/16271-…`（`adfa304e`，目标那一路的精确修复）
和 `feat/16271-approx-draft-kv-opt-in`（`367a4673`，提供 `seq_lens_cpu_is_approximate` 与上界）。

开关：`VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK=1`（默认关）。

### 改动清单

| 文件 | 改动 |
|---|---|
| `vllm_ascend/envs.py` | 新增开关 |
| `attention_v1.py` · `AscendMetadata` | 3 个字段：`draft_kv_upper_bound` / `draft_query_lens` / `draft_tail_mask_cache` |
| `attention_v1.py` · 模块级 | `build_draft_tail_mask()` —— 掩码构造 |
| `attention_v1.py` · `build()` | 判定"这次 build 的 `seq_lens_list` 是上界"，把逐请求 query 长度和一个每步共用的掩码缓存挂上去 |
| `attention_v1.py` · `AscendAttentionBackendImpl` | 新方法 `_forward_draft_tail_masked()`；`forward_fused_infer_attention()` 里一处早返回 |

### 核心代码

```python
def build_draft_tail_mask(seq_lens, num_reqs, query_len, kv_span, sliding_window):
    device = seq_lens.device
    lens = seq_lens[:num_reqs].to(torch.int32)
    offsets = torch.arange(query_len, device=device, dtype=torch.int32)
    cols = torch.arange(kv_span, device=device, dtype=torch.int32).view(1, 1, -1)
    # [num_reqs, query_len, 1]: the last KV position each query token may see.
    last = ((lens - query_len).view(-1, 1) + offsets.view(1, -1)).unsqueeze(-1)
    mask = cols > last
    if sliding_window:
        mask |= cols < (last - sliding_window)     # band keeps W+1 positions
    return mask.to(torch.int8)
```

```python
attn_output, _ = torch_npu.npu_fused_infer_attention_score(
    query=query.view(num_reqs, query_len, self.num_heads, self.head_size),
    key=key, value=value,
    atten_mask=mask,                       # 设备张量，依赖设备上的 seq_lens
    block_table=block_table,
    input_layout="BSND",                   # 离开 FAI split-fuse 模板
    block_size=block_size,
    actual_seq_lengths=[query_len] * num_reqs,   # BSND 要逐请求长度，不是累积
    actual_seq_lengths_kv=actual_seq_lengths_kv, # 上界，宿主白拿
    num_key_value_heads=self.num_kv_heads,
    num_heads=self.num_heads,
    scale=self.scale,
    sparse_mode=0,
)
```

### 安全回退

任何一条不满足就 `return None`，调用方落回原路径（即 `c0` 行为），不会静默算错：
非草稿 build / `learnable_sink` 非空 / 非因果 / `block_table` 为空 /
**query 长度不是矩形**（BSND 的前提）/ token 数与 `B×q` 对不上。

### 实现细节

- 掩码 **每步构造一次**，由该注意力组的全部 5 层共用（缓存键含 `sliding_window`）；
- 掩码显存 `B × (K+1) × KV_S` 字节，`max_model_len=8192 / B=64 / K=8` ⇒ **4.7 MB**；
- 构造用**比较式**而不是 `index_select`：gather 在 910B4 上**按源张量大小计价**，
  从 2048×2048 里选几行要 444 µs，比较式只要 ~153–170 µs 且**与 batch 无关**。

---

## 5. 性能提升

### 5.1 算子层（单卡独立探针 `harness/maskfix2.py`，drafter 形状，对纯 torch 参考的相对误差）

滑窗 = 生产真实配置（`sliding_window=2048`）：

| n | 精确长度（今天，需同步） | **上界+设备掩码** | `c0` 现有开关 |
|---|---|---|---|
| 8 | 0.003493 / 150.5 µs | **0.003493 / 155.1 µs** | 0.1914 |
| 64 | 0.002919 / 201.1 µs | **0.002919 / 303.3 µs** | 0.2749 |

无滑窗 n=64：参考 `0.003275 / 201.6 µs`，本方案 **`0.003275 / 199.3 µs`**。

- **精度与"喂精确长度"逐位同级**；`c0` 差 **60~100 倍**。
- 成本：无滑窗持平（−2.7% / +6.1% / −1.1%）；滑窗下 n=8 +3.1%，**n=64 +50.8%**
  （通用模板没有 FAI 的 band 优化）。

### 5.2 端到端（六臂回文序 `base1 c01 mf1 mf2 c02 base2`，每臂两个独立实例；random 集，c=8，96 prompts）

| 臂 | 吞吐 tok/s | 中位 ITL ms | pooled accept_len |
|---|---|---|---|
| base1 / base2 | 132.14 / 131.33 | 75.60 / 75.11 | 1.6340 / 1.5893 |
| c01 / c02 | 176.90 / 184.04 | 50.21 / 50.03 | 1.5149 / 1.5472 |
| **mf1 / mf2** | **185.17 / 193.02** | **49.86 / 50.19** | **1.6277 / 1.6547** |

| | 吞吐 | 中位 ITL | accept_len |
|---|---|---|---|
| `c0`（已有开关） vs base | +37.0% | −33.5% | **−5.00%** |
| **本方案 vs base** | **+43.5%** | **−33.6%** | **+1.83%** |
| **本方案 vs `c0`** | **+4.8%** | −0.2% | **+7.19%** |

**收益归属要分清**：那 37% 早就在 `c0` 手里了，**本方案新增的是 +4.8% 吞吐，外加把 `c0` 付掉的 5% 接受率补回 0**。

**判据分离**：
- 本方案两实例的接受率（1.6277 / 1.6547）**全部高于** `c0` 两实例（1.5149 / 1.5472），最小值比最大值还高 0.08；
- 与 base（1.5893 / 1.6340）**区间重叠 ⇒ 统计上无法区分 ⇒ 接受率是全额拿回，不是拿回一部分**；
- 吞吐上两实例也都高于 `c0` 两实例。

**六臂输出逐字节完全相同**（sha `df34ea140872348c`）。
**踩在天花板上**：早先那个故意写错的天花板臂是 `c=8 ITL 50.0 ms`，本方案 **50.02 ms**，且没付接受率。

### 5.3 正向对照（先于数字）

这次调查两次被"静默空转"骗过（钩子没触发 / 判据上界差 1，读出来的数字都完全说得通），
所以判据是先立的：

```
[NT_MASKFIX] calls=22000 hit=22000 capturing=0 noncausal=0
             nonuniform_q=0 shape=0 sinks=0 verify_max_rel=0.0 verify_n=8
```

- `hit == calls`，四个回退分支一次没触发 ⇒ **这个臂不可能是 `c0` 的伪装**；
- `NT_MASKFIX_VERIFY=8` 让前 8 次草稿 build 额外用**精确长度**跑一遍原 TND 参考，
  四个 TP rank 全部 **`rel=0` 逐位相同**（`b=1 q=8 s2=8192 win=2048`）——
  窗口语义在真模型上确认（差一位这里会读到 ~0.15）。

### 5.4 整理后的正式形态：接受率与精确路径**逐计数器相同**（`logs/probe_acc.out`）

§5.2 的端到端数是 monkeypatch 形态（`harness/maskfix_patch.py`）跑的，正式形态（本分支的代码）
的复跑排了队。卡在 19:35 空出来，`harness/leg_clean.sh` 自动起了第一臂 `cl1`：
补丁应用、服务 342 s 起来、接受率 probe 12 条全部跑完、`accept_cl1_p1.json` 落盘——
**然后 19:42 容器被 SIGKILL（`Exited (137)`，非 OOM），别人起了一个占满 8 卡的 TP8 任务。**
bench 没跑成。

但 probe 跑完了，而 probe 恰好是**比 §5.2 那张表更硬的判据**：并发 1、贪婪、
`HCCL_DETERMINISTIC=true`、同样 12 条 prompt，所以计数器本身是确定的。
把每个臂的日志都截到 probe 阶段（第 13 个 completion 响应之前，`harness/probe_acc.py`）：

| 臂 | Accepted | Drafted | ratio | probe accept_len |
|---|---|---|---|---|
| `base1` / `base2`（精确长度） | 1883 | 9496 | 0.198294 | 2.5864 |
| `c01` / `c02`（只喂上界） | 1721 | 10832 | 0.158881 | **2.2710** |
| `mf1` / `mf2`（monkeypatch 形态） | 1883 | 9496 | 0.198294 | 2.5864 |
| **`cl1`（本分支的正式形态）** | **1883** | **9496** | **0.198294** | **2.5864** |

这不是"落在 `c0` 之上"，是**和精确路径逐计数器完全相同**——接受、起草的 token 数一个不差。
判据有区分力：同一把尺子上 `c0` 读出 1721/10832（少接受 162 个、多起草 1336 个，accept_len −12.2%），
所以 `cl1 == base` 不是"这个指标分不开"，是真的重合了。

它同时排除了这次调查最怕的那种失败：**如果正式形态的掩码路径整条回退掉了，
`cl1` 会精确等于 `c0` 的 1721/10832**（`cl` 臂的 `VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV` 一直是 1，
只有 `..._DEVICE_MASK` 在切换）。读到的是 base 那一组，所以掩码确实在工作。

还缺的：**正式形态的吞吐 / ITL 仍未实测**（bench 没跑到）。
已确立的是正确性与接受率等价，性能那一半只有 monkeypatch 形态的数（§5.2）。

---

## 6. 还没做的

1. **每臂只有 2 个实例**。方向一致、两两不重叠，但 2v2 只够说方向，给不了显著性。
2. **接受率贴地板**：pooled 数是 probe + bench 混算，random 集 `ratio≈0.07`、`accept_len≈1.6`。
   健康数据集上差距应该更大，也更有说服力。
3. **大并发未测**：§5.1 那个"n=64 + 滑窗贵 50.8%"在 c=8 下没有变成端到端回归（批量小），
   但 c=32/64 必须单独复测——那正是通用模板最吃亏的区间。
4. **`nonuniform_q` 回退没有单测**。
5. **上面的端到端数是用 monkeypatch 形态跑的**（`harness/maskfix_patch.py`），
   本分支是整理后的正式形态。两者的算法完全一致，但**整理过程本身抓出了一个真 bug**：
   `attention_v1.py` 没有导入 `vllm_ascend.envs`，monkeypatch 那版直接读 `os.environ` 所以没暴露，
   **只有跑完整单测才会报 `NameError`**。已修。
   正式形态的**接受率已在真机上验过，与精确路径逐计数器相同（→ §5.4）**；
   没验到的是它的**吞吐 / ITL**——`leg_clean.sh` 的 bench 没跑成，容器被别人的 8 卡任务 SIGKILL 了。
   性能那一半目前只有 monkeypatch 形态的数（§5.2）。

### 单测状态（在镜像内跑的，不占卡）

```
NEGATIVE CONTROL（不打补丁）        10 failed     ← 证明这些用例真在测新代码
新增用例（打补丁）                 10 passed
tests/ut/attention/test_attention_v1.py  43 passed, 0 failed
```

## 7. 文件

| 路径 | 内容 |
|---|---|
| `0001-16271-draft-kv-device-tail-mask.patch` | 正式形态补丁（`diff -u`，2 文件 / 153 行新增） |
| `harness/maskfix_clean.py` | 生成上面这份补丁的锚点式应用器（带 `--revert`） |
| `harness/maskfix_patch.py` | 实验形态（monkeypatch + `NT_MASKFIX_VERIFY` 黄金参考对拍），端到端数据出自它 |
| `harness/maskfix2.py` | 单卡算子级探针（§5.1 的数据） |
| `harness/leg_mask.sh` | 六臂端到端 A/B |
| `harness/leg_clean.sh` | 正式形态的复跑验证 |
| `harness/probe_acc.py` | 把 serve 日志截到 probe 阶段再汇总接受率（§5.4 的尺子） |
| `logs/mask.out` | 六臂端到端 A/B 原始日志 |
| `logs/clean.out` | 正式形态复跑的那半截（到 `cl1` probe 为止，之后容器被杀） |
| `logs/probe_acc.out` | §5.4 的计数器对比，含逐窗口原始行 |
| `REPRO.md` | 完整过程，含所有被否定的方向与踩过的坑 |
