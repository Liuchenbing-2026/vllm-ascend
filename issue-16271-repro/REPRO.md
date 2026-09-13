# vllm-ascend #16271 — DSpark 的 host 下发被 `seq_lens.tolist()` 阻塞
## 定位 · 修复 · 验收 · 完整复现流程

- Issue: <https://github.com/vllm-project/vllm-ascend/issues/16271>
- 分支（**建议合入**）: <https://github.com/Liuchenbing-2026/vllm-ascend/tree/fix/16271-dspark-attn-seq-lens-host-sync>
  commit `adfa304e` —— 目标模型那一路，正确且净收益
- 分支（**证据，不要合**）: <https://github.com/Liuchenbing-2026/vllm-ascend/tree/exp/16271-draft-seq-lens-device-tensor>
  commit `fd6ab8ad` —— 草稿那一路走 device 张量：正确，但算子绑定层逐请求回读导致负优化（见 §8.5）
- 分支（**可选开关，实测有效**）: <https://github.com/Liuchenbing-2026/vllm-ascend/tree/feat/16271-approx-draft-kv-opt-in>
  commit `367a4673`（基于 `adfa304e`）—— `VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1`，默认关：
  草稿那一路改读乐观上界，**输出逐字节不变、接受率 −6.4%、c=8 吞吐 +31%**（见 §8.5.5）
- base = upstream main `84d6dc83`（2026-09-11 00:03）
- 补丁副本: `0001-fix-16271.patch`（仅第一条）、`0001-0002-16271.patch`（两条）；全部脚本在 `scripts/`

---

## 0. 结论速览

| | |
|---|---|
| **根因** | `attention_v1.py:349` 的 `seq_lens.tolist()` 在 `parallel_drafting`（DFlash/DSpark）下作用在 **device 张量**上，每次 build 都要等计算流排空 |
| **基线实测** | 20000/20000 次 build 走 D2H，累计阻塞 **118.5 s / 787.6 s 墙钟** |
| **已交付修复** | 目标模型那一路（**75.8%** 的 build）改从精确 host 镜像取；**30264 次逐元素比对零偏差** |
| **修复收益** | c=8 中位 ITL **75.395±0.615 → 74.543±0.399 ms（−1.1%）**，每臂 6 次基准 / 3 次独立重启、只取不带探针的干净 leg，t=2.84 df=10 **p≈0.018**；吞吐在噪声内（详见 §7.2） |
| **天花板探针** | 把**草稿那一路**也去掉同步（故意写错，只量上限）：c=8 ITL **50.0 ms（−34%）**、吞吐 **179.8 tok/s（+31%）** |
| **草稿那一路** | 同步也能全部去掉（实测 d2h=0、168000 次比对零偏差、接受率不变），但 **FIA 传 device 张量比传 list 贵 2.0×/4.7×/8.0×（n=8/32/64），开销 ~+35 µs/请求** ⇒ **issue 提的方向在这个算子版本上是负优化**，只在低并发有收益（c=1 −6.7%，c=8 −1.4%） |
| **⚠️ 未解决** | 天花板的 −34% ITL / +31% 吞吐仍在桌上。三条路：算子侧去掉逐请求回读 / 框架侧提前给出精确 host 长度（§8.5.4），或者 —— **也许根本不需要精确值**（§8.5.5）|
| **🏁 2026-09-11 更新（换 18 机时发现）** | 别人已经把"框架侧给出精确 host 长度"做出来了（`/data2/dflash2-adapt/vllm-ascend`，2026-08-21），**但只在 MRV1**：同步调度下 vLLM 为了吐输出本来就把接受的 token id 列表放在 host 上，`rejected = num_draft+1-len(ids)` 零额外代价。**MRV2（我 A2 测的）拿不到** —— `sampled_token_ids=None`，那次 D2H 是故意与 `propose` 重叠的。⇒ **MRV1 有正确解法；MRV2 那 +31% 仍然只有算子侧或有损路。** 另：他们的 `parallel_drafting_seq_lens_cpu_valid` 与我的 `seq_lens_cpu_is_exact` 一一对应，独立收敛到同一设计。详见 §8.5.4 |
| **🏆 §8.5.5 已实测：天花板是可用的有损选项，不是上限** | 三臂实跑（2026-09-11）：**72 份贪心输出逐字节完全相同**（含语义上写错的天花板臂）⇒ 推测采样的保证在真机成立；接受率 pool 3.7~4.0 万次草稿后 **−6.40%**，而构造上正确的对照臂只偏 −1.07%（信噪比 6 倍）；总产出 token 两臂只差 0.2%，ceil 是多花 6.6% decode 步。⇒ **用 ~6% 接受率换 −34% ITL / +31% 吞吐，且输出不变** —— 建议做成默认关闭的显式开关 |
| **🔬 §8.5.5.5 代数闭合（2026-09-12）** | 近似误差 `delta = host − device` **就是这一步被拒绝的 token 数**，取值 `[0, K]`，`K=8`。两台独立仪器对上：接受率计数器反推 `K(1−r)=6.7808`，1417 次 build 的 trace 实测 `6.7509±0.0510`（差 **0.59 SE**）。之前推不通是因为我假设 host 装的是目标前向**之前**的长度，实际乐观上界里已经把 `+num_query_per_req` 折进去了。目标 build 对照 **10745/10745 全 0**。**50.18% 的步 `delta` 恰好等于 8**（整步只留下 bonus token） |
| **❌ §8.5.5.7 选项 1 已实装实测（2026-09-12）** | 上一步的拒绝数**零 device 交互**就能算（`ub_{t−1} − computed_t`，两个 host numpy 数组）。修正覆盖率 0→49.5→96.9% 的**剂量-反应是单调反向的**：残余误差 6.6483→3.9149→**0.0670（砍掉 99%）**，accept_len **2.2874±0.0129(n=4)→2.2770→2.2077（−5.53σ，比什么都不做还差）**。⇒ **估得越准越亏**，机制=无偏但双向，拿便宜的单边偏长换了贵 2.6 倍的双向偏短。另有两个**静默** bug，都是靠对照而非数字抓出来的：①钩子挂在 `build_draft_attn_metadatas` 而 drafter 是 `enforce_eager` ⇒ 一次没触发；②有效性上界写成 `num_query_per_req−1`=7（该属性是 8 不是 9）⇒ 恰好丢掉占 45% 的 `d=8`、也就是误差最大的那一半步 |
| **❌ §8.5.5.6 修正拒绝数这条路走不通（2026-09-12）** | 四臂 A/B：`base` 2.6064 / `c0`(现有开关) 2.2916(−12.08%) / `ub−7` 2.3194(−11.01%，**0.52σ = 噪声**) / `ub−8` 2.1829(**−16.25%，更差**)。8 份输出 sha 全同。**`ub−8` 的恰好命中率 50.6%（`c0` 只有 2.1%）、mean\|e\| 好 5.7 倍，接受率反而掉** ⇒ "估得更准就能买回接受率"被直接证伪。按方向拆每步代价：**偏短 −53.3%，偏长 −20.0%，偏短贵 2.6 倍**（偏长只是多读被回滚的草稿 KV，偏短是读不到刚接受的 token）。**恒不偏短的常数只有 `est=0`，就是已发布的 `c0`。** 由此外推"用上一步拒绝数"≈2.26~2.36，**最好也只是和 `c0` 打平** ⇒ **不实现** |
| **🛑 §8.6 MRV2 上“别搬到宿主”这条路也关了（2026-09-13）** | 之前据以为“底层算子本来就吃设备张量、卡的只是 torch_npu 绑定”—— **错的，此处更正**。FIA 的 op IR 把 `actual_seq_lengths{,_kv}` 标成 `ValueDepend`，tiling 必须在宿主读值；GE fallback 也是 `GetData<int64_t>()` 拷回宿主；`aclnnInner…TensorGetWorkspaceSize` 不是设备张量入口，是 `GetMaxWorkspaceSize` 用**数据指针为 null 的假张量**调的。换 ATB 的 `_npu_paged_attention_splitfuse` 也不行：**`context_lens` 放设备直接 `tensor.hostData is null`**（四种放置组合实测，只有 cpu/cpu 跑得通）。⇒ **这台栈上每个可用的注意力算子，tiling 都从宿主的序列长度算起；MRV2 上无损去同步不是 vllm-ascend 能改的事**。剩下的是一个具体的 CANN 请求（把 `isTilingSink` 做成真执行模式 + aclnn 开个收 `aclTensor*` 的入口）|

| **✅ §8.7 拿回来的办法找到了（2026-09-13 实测）** | `atten_mask` **不是 `ValueDepend`**，它可以合法地依赖设备上的 `seq_lens`。于是：**长度喂上界（宿主白拿）+ 尾巴用设备侧逐请求掩码盖掉**。TND 一律走 FAI split-fuse 模板（给掩码就只能 2048²+sm3/4），但草稿这一路每请求 q 恒为 K+1，**TND 和 BSND 是同一块内存的两个视图**，换 BSND 就落到通用模板，`sparse_mode=0` + `[B,≥Q_S,≥KV_S]` 逐请求掩码合法。**实测：候选与“喂精确长度”逐位同级**（滑窗 n=8/64 两档 rel 0.003493 / 0.002919 与参考**完全相同**），而 `c0` 是 0.19~0.27（**差 60~100 倍**）。成本：无滑窗持平（−5～+6%），滑窗下 n=8 +3.1%、n=64 **+50.8%**；掩码构造 ~153–170 µs 且与 n 无关、一步一次五层共用。➕ 坐实了一个必须照抄的语义：band 模式 `pre_tokens=W` 保留的是 **W+1** 个位置（两端都闭），写错误差 0.157 |

| **🏆 §8.7.5 端到端 A/B：拿回来了（2026-09-13）** | 六臂回文序、每臂两个独立实例。**`mf` vs base：吞吐 +43.5%（131.74→189.09 tok/s）、中位 ITL −33.6%（50.02 ms）、accept_len +1.83%（与 base 区间重叠 = 统计上无法区分）**；`c0` vs base 是 +37.0% / −33.5% / **−5.00%**。`mf` 的两个实例在接受率和吞吐上**全部高于** `c0` 的两个，两两不重叠。**六臂输出逐字节相同**（sha `df34ea140872348c`）。踩在 §8.5.5 天花板上（天花板 50.0 ms vs 实测 50.02 ms）**且没付接受率**。正向对照：`calls=22000 hit=22000`、四个回退分支全 0、`verify_max_rel=0.0`（前 8 次 build 与精确长度参考四个 rank 全部 `rel=0`）。❗每臂只有 2 个实例，接受率贴地板（random 集 ratio≈0.07），大并发未测 |


**所以：issue 对问题的判断完全正确（代价比它说的还大），但它提的解法方向经实测是负优化。
我交付了能证明为精确且净收益的那一半，并把另一半连同否定它的数据一起推上去。**

---

## 1. 机器 / 镜像 / 模型

| 项 | 值 |
|---|---|
| 机器 | A2 `root@112.29.145.3`（bms-75847414-002），8×910B4-1 |
| 接入 | `scripts/a2.sh`（plink + hostkey + 密码，带退避重试；本机 OpenSSH 连不通） |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:nightly-main` |
| 容器 | `nt-dspark-mrv2`（`scripts/start_ct.sh`） |
| vllm | 0.28.0 `/vllm-workspace/vllm` |
| vllm-ascend | `e5118d151314ae18e56c0b63aa8dd00d294adc22`（2026-09-09）`/vllm-workspace/vllm-ascend` |
| CANN / torch | 9.1.0 / torch 2.10.0 + torch_npu 2.10.0.post4 |
| 卡 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`，TP4 + EP |
| 目标模型 | `/data01/models/Qwen3.6-35B-A3B`，arch **`Qwen3_5MoeForConditionalGeneration`**（就是 issue 说的 Qwen3.5 系） |
| 草稿 | `/data01/models/Qwen3.6-35B-A3B-speculator.dspark`，arch `Qwen3DSparkModel`，8 spec tokens |
| Runner | **MRV2**（`VLLM_USE_V2_MODEL_RUNNER=1`） |

镜像里的 `attention_v1.py` 与 upstream main **逐字节相同**（`md5=3ffdd3667b206531740ed46c69285f41`），
所以这台机上测到的行为可以直接对应 main。`attention/utils.py`、`worker/v2/attn_utils.py` 与 main 有
54 个 commit 的差异，但**被改动的 4 个 hunk 逐字节相同**（§6 有校验）。

该镜像的 triton 已可用（vector-add smoke `max_abs_err=0.0`），**不再需要**早前那套
`backends/ascend/bishengir/` 覆盖法。

---

## 2. 根因

`vllm_ascend/attention/attention_v1.py`（main == 镜像）：

```python
322  # Prefer _seq_lens_cpu (always available, updated during draft
323  # iterations) over seq_lens_cpu (None in async spec decode mode).
324  if common_attn_metadata._seq_lens_cpu is not None:
325      seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]   # host 镜像
326  elif common_attn_metadata.seq_lens_cpu is not None:
327      seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
328  else:
329      seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")
...
334  if isinstance(self.kv_cache_spec, CrossAttentionSpec):
335      seq_lens = common_attn_metadata.seq_lens
337  elif self.speculative_config and self.speculative_config.parallel_drafting:
338      seq_lens = common_attn_metadata.seq_lens         # ← 换成 device 张量
...
349  seq_lens_list = seq_lens.tolist()                    # ← 阻塞式 D2H
```

`vllm/config/speculative.py:1429-1430` 对 `method in ("dflash","dspark")` 强制
`parallel_drafting = True` ⇒ **DSpark 下每一次 build 都命中 337，然后在 349 同步。**

**MRV2 并不绕开它**：`vllm_ascend/worker/v2/attn_utils.py:315` 调的就是同一个
`attn_metadata_builder.build()`。MRV2 反而**另外**还付一次 host 同步 ——
`worker/v2/model_runner.py:687` 的 `self.num_computed_tokens_event.synchronize()`
（只要开了投机就付），用来维护精确的 `seq_lens_np`。

---

## 3. 基线量化

诊断探针（`nt_patch.py --probe`，只统计"这次 build 的 Python list 是不是从 device 张量取的"）：

```
[NT_PROBE] calls=20000 d2h=20000 d2h_total_ms=118477.5 checked=0 mismatch=0 wall=787.6s
```

- **20000 / 20000 次 build 全部走 D2H**
- 累计阻塞 **118.5 s / 787.6 s 墙钟**，平均每次 5.9 ms（就是"等计算流排空"的时间）

---

## 4. 一条被自己的检查否掉的错误路线（保留备查）

第一版补丁把草稿路径的 host 镜像换成 `seq_lens_cpu_upper_bound`。
打开 `NT_SEQ_LENS_CHECK=1` 做逐元素比对：

```
by_src={'np': [8316, 8316, 0], 'ub': [2684, 2684, 2682]}
[NT_PROBE][MISMATCH] src=ub n=8 idx=0 host=530 dev=522 delta=8
```

- `np`（目标模型，源自 `seq_lens_np`）：8316 次比对 **0 偏差**
- `ub`（草稿，源自上界）：2684 次里 **2682 次偏 +8**，正好 = `num_speculative_tokens`

上界假设"草稿全被接受"，而拒绝只在 device 上结算。upstream 自己也写明了
（`vllm/v1/attention/backends/utils.py:726-729`："Upper bound is exact for prefill rows"，
只用于分 prefill/decode）。**草稿路径 host 侧根本没有精确值。**

> 已有的两个社区 PR 都掉进同一个坑：**#15229**（同方向，但取的镜像在 MRV2 草稿路径上是
> `np.full(max_seq_len)` 占位符）、**#9095**（直接删 337 那个 elif）。按现状合入会把草稿的
> KV 长度喂错。

---

## 5. 最终补丁（3 个文件 + 3 个单测）

1. **`vllm_ascend/attention/utils.py`** — `AscendCommonAttentionMetadata` 新增
   `seq_lens_cpu_is_exact: bool = False`，并在 `unpadded()` 透传。
   默认 False：未审计的 producer 行为不变（MRV1 的 `model_runner_v1.py:3373` 注释自己写了
   "Always pass optimistic_seq_lens_cpu"，放进 `_seq_lens_cpu` 的是**乐观**值）。
2. **`vllm_ascend/worker/v2/attn_utils.py`** — `seq_lens_cpu_is_exact = seq_lens_np is not None`，
   即只有目标模型 build 宣称自己有精确镜像。
3. **`vllm_ascend/attention/attention_v1.py`** — 把 `seq_lens_list` 的来源与 `seq_lens` 张量
   解耦：镜像仍然描述 `seq_lens` 时从镜像取，否则（cross-attention / 草稿 build）保留原来的
   device `.tolist()`。张量语义、padding、metadata 字段全部不变。

单测（`tests/ut/attention/test_attention_v1.py`，3 个新用例）：
- 全文件 **36 passed**（打了补丁的树）
- 同样 3 个用例在**未打补丁**的树上 **3 failed** ⇒ 判据有区分度

---

## 6. 制品溯源（分支里的字节 == 机器上跑过的字节）

| 文件 | 机器上（打 fix、不带探针） | 分支 commit blob | GitHub raw |
|---|---|---|---|
| `attention/attention_v1.py` | `ab14d57b25b03bf395e9edffaf3626dc` | 同 | 同 |
| `attention/utils.py` | 4 个 hunk 逐字节相同 | — | — |
| `worker/v2/attn_utils.py` | 4 个 hunk 逐字节相同 | — | — |

`attention_v1.py` 整文件 md5 三处一致。另两个文件整文件 md5 不同，是因为机器在
`e5118d15`、分支在 `84d6dc83`，两者之间有 54 个**与本改动无关**的 commit；
我按 hunk 做了逐字节比对（脚本见 §9）。

---

## 7. 验收

### 7.1 同步次数与正确性（决定性判据）

打了补丁 + 探针 + `NT_SEQ_LENS_CHECK=1`，每个 rank：

```
[NT_PROBE] calls=10000 d2h=2434 d2h_total_ms=57572.4 checked=7566 mismatch=0 pad_only=0
           by_src={'np': [7566, 7566, 0], 'ub': [2434, 0, 0]}
```

- **D2H 从 10000/10000 降到 2434/10000（−75.7%）**
- 走 host 镜像的 7566 次，**逐元素与 device 张量比对，0 偏差**；4 个 rank 合计 **30264 次比对全对**
- 剩下的 2434 次全部是草稿 build（`ub`），按设计继续读 device

### 7.2 延迟 / 吞吐（基准）

`vllm bench serve --dataset-name random --random-input-len 512 --random-output-len 256 --ignore-eos`，
每个点跑两遍。**S_base / S_fix / S_ceil 是同一轮里依次起停的三个实例**：

| 并发 | 指标 | S_base（未修） | S_fix（本补丁） | S_ceil（天花板探针） |
|---|---|---|---|---|
| 1 | 中位 ITL (ms) | 57.43 / 57.37 | 55.94 / 55.86 | **49.03 / 48.88** |
| 4 | 中位 ITL (ms) | 66.35 / 66.43 | 64.02 / 64.48 | **49.59 / 49.38** |
| 8 | 中位 ITL (ms) | 75.93 / 75.97 | 74.12 / 74.00 | **50.03 / 49.96** |
| 1 | 吞吐 (tok/s) | 28.77 / 33.80 | 32.86 / 35.12 | 33.54 / 33.03 |
| 4 | 吞吐 (tok/s) | 85.05 / 81.48 | 86.36 / 83.99 | **106.85 / 100.26** |
| 8 | 吞吐 (tok/s) | 138.45 / 136.00 | 141.03 / 133.75 | **179.80 / 179.88** |

**跨重启漂移必须扣掉。** c=8 这个点、**只取不带探针的干净 leg**（带探针/带 CHECK 的 leg
时间被污染，一律剔除，且它们不进统计量 —— 见 [[ab-session-is-the-variance-unit]]）：

| 臂 | c=8 中位 ITL (ms)（每臂 3 个独立实例 × 2 次） | 均值 | sd |
|---|---|---|---|
| baseline（A2_base / A3_base / S_base） | 74.63 / 74.69 / 75.32 / 75.83 / 75.93 / 75.97 | **75.395** | 0.615 |
| fix（B4_fix / B5_fix / S_fix） | 74.00 / 74.12 / 74.54 / 74.82 / 74.85 / 74.93 | **74.543** | 0.399 |
| ceiling（S_ceil） | 49.96 / 50.03 | **50.00** | — |

⇒ **本补丁 −0.852 ms（−1.13%），t=2.84 / df=10 / p≈0.018 —— 小但真实，两臂区间只在
74.63–74.93 有一点重叠；天花板 −25.4 ms（−33.7%），远在任何漂移之外。**
吞吐同理：本补丁在噪声内（baseline 134.6±3.7 vs fix 136.6±6.1 tok/s），天花板 +31%。

### 7.3 Profiling：空泡的形状变了

`--profiler-config.profiler=torch`，`/start_profile` → 6 请求 × 128 token / 并发 1 → `/stop_profile`，
离线 `torch_npu.profiler.profiler.analyse()` 后统计 rank0 的 `task_time.csv`
（去掉首尾各 10%，把所有流的任务区间求并集）：

| 指标 | S_base | S_fix | S_ceil |
|---|---|---|---|
| 真实器件工作量 `work_duration_sum_ms` | 7569.7 | 7936.3 | 7764.2 |
| **相邻真实工作间隙 p90 (µs)** | **199.75** | **110.25** | **27.5** |
| 50–100 µs 间隙个数 | 22437 | 18807 (−16%) | 4857 (**−78%**) |
| 100–200 µs 间隙个数 | 18013 | 8759 (**−51%**) | 1112 (**−94%**) |
| 200–500 µs 间隙个数 | 33801 | 20631 (**−39%**) | 2009 (**−94%**) |
| 50–500 µs 间隙合计 | 74251 | 48197 (**−35%**) | 7978 (**−89%**) |
| `device_bubble_pct`（把 EVENT_WAIT 也算占用） | 58.87 | 35.07 | **3.54** |

读法：**每次 build 一个同步 → 在时间线上留下一串 50–500 µs 量级的空泡。**
补丁把这类空泡砍掉 35%，天花板砍掉 89%，p90 间隙 199.75 → 110.25 → 27.5 µs，三档单调。

> ⚠️ 两个我自己踩过的读数陷阱，写在这里免得后人重蹈：
> 1. **别把 EVENT_WAIT 算成"设备在干活"**。我第一版的 `device_bubble_pct` 就是这么算的，
>    读出"55.5% → 39.6%"的漂亮改善；换成只算真实工作后这个改善**消失**。
>    EVENT_WAIT 变多恰恰说明"host 提前把活排进去了、流在等彼此"，不等于设备更忙。
> 2. **profiler 本身会把三臂的差异压扁**（window 只差 5%，而不带 profiler 时 ITL 差 15%）。
>    所以延迟结论要用不带 profiler 的基准，profiling 只用来看**空泡的形状**。

---

## 8. 完整复现流程

```bash
# ---------- 0) 上机，先确认卡是空的 ----------
scripts/a2.sh 'npu-smi info'

# ---------- 1) 起容器 ----------
scripts/a2put.sh scripts/start_ct.sh /root/nt/start_ct.sh
scripts/a2.sh 'bash /root/nt/start_ct.sh'

# ---------- 2) 把脚本放进容器（宿主 /data01/nt-work == 容器 /nt） ----------
for f in nt_patch.py serve.sh wait_ready.sh bench.sh prof.sh stop.sh \
         runexp.sh runexp3.sh runall.sh runall3.sh \
         analyze_tasktime.py analyse_offline.py collect.sh; do
  scripts/a2put.sh scripts/$f /data01/nt-work/$f
done

# ---------- 3) 打/卸补丁（每个文件都留 .nt-orig，--revert 逐字节还原） ----------
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --status
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --fix            # 生产补丁
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --fix --probe    # + 诊断探针
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --fix --ceiling  # 天花板探针（故意写错）
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --revert

# ⚠️ 每次改完代码必须清 torch.compile 缓存，否则会撞
#    ValueError: too many values to unpack (expected 47)
docker exec nt-dspark-mrv2 rm -rf /root/.cache/vllm/torch_compile_cache

# ---------- 4) 一条命令跑完 baseline / fix / ceiling 三臂 ----------
docker exec -d nt-dspark-mrv2 bash -lc 'bash /nt/runall3.sh > /nt/logs/runall3.txt 2>&1'
#   每臂 = 起服 → warmup → c1/c4/c8 各两遍 → c=1 profiling 窗口 → 停服
#   全程约 75 分钟

# ---------- 5) 收基准数字 ----------
docker exec nt-dspark-mrv2 bash -lc 'bash /nt/collect.sh'

# ---------- 6) 解析 profiling 并统计空泡 ----------
docker exec nt-dspark-mrv2 bash -lc \
  'python3 /nt/analyse_offline.py /nt/prof/S_base/*rank0*_ascend_pt \
                                  /nt/prof/S_fix/*rank0*_ascend_pt \
                                  /nt/prof/S_ceil/*rank0*_ascend_pt'
for t in S_base S_fix S_ceil; do
  docker exec nt-dspark-mrv2 bash -lc \
    "python3 /nt/analyze_tasktime.py /nt/prof/$t/*rank0*_ascend_pt"
done

# ---------- 7) 正确性扫（探针逐元素比对 host 列表 vs device 张量） ----------
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --revert
docker exec nt-dspark-mrv2 python3 /nt/nt_patch.py --fix --probe
docker exec nt-dspark-mrv2 rm -rf /root/.cache/vllm/torch_compile_cache
MRV2=1 NT_SEQ_LENS_PROBE=1 NT_SEQ_LENS_CHECK=1 NT_MODE=check \
  scripts/a2.sh 'bash /root/nt/launch_exp.sh B3_fix_check'
# 期望: by_src={'np':[N,N,0],'ub':[M,0,0]}，mismatch=0

# ---------- 8) 单测 ----------
docker exec nt-dspark-mrv2 bash -lc \
  'cd /vllm-workspace/vllm-ascend && python3 -m pytest tests/ut/attention/test_attention_v1.py -q'

# ---------- 9) 收工：停服 + 还卡 ----------
docker exec nt-dspark-mrv2 bash -lc 'bash /nt/stop.sh'
docker rm -f nt-dspark-mrv2
scripts/a2.sh 'npu-smi info | tail -20'
```

### 环境里必须知道的坑

| 坑 | 表现 | 处理 |
|---|---|---|
| vLLM 0.28 换了 profiling 开关 | `VLLM_TORCH_PROFILER_DIR` 报 *Unknown vLLM environment variable*，`/start_profile` 404 | 用 `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=...` |
| `stop_profile` 后不自动解析 | 没有 `ASCEND_PROFILER_OUTPUT/` | 离线跑 `torch_npu.profiler.profiler.analyse(profiler_path=<*_ascend_pt>)`；每 rank 原始 ~870 MB，解析 6–8 分钟 |
| torch.compile 缓存 key 不含 vllm-ascend 源码哈希 | 改完代码起服崩 `too many values to unpack (expected 47)` | 每次改码清 `/root/.cache/vllm/torch_compile_cache` |
| plink 命令行长度有限 | 传大文件时 base64 一把梭会静默失败 | `a2put.sh` 已改成分块（24000 字符/次） |

---

## 8.5 第二轮：把草稿那一路也做了 —— 结论是 **issue 提的方向在这个算子版本上是负优化**

第一轮留下的判断是"剩下 24% 的同步扛着 ~97% 的代价，要拿到它必须让 FIA 收 device 张量"。
第二轮把这件事做完了，并且**测出这个方向行不通**。

### 8.5.1 实现（分支 `exp/16271-draft-seq-lens-device-tensor`，commit `fd6ab8ad`）

- `AscendMetadata.seq_lens_list` 允许为 `None`（= 没有 host 副本）；
  `get_seq_lens_list()` 按需materialize并缓存，`get_seq_lens_kv()` 在没有 host 副本时
  直接返回 device 张量。树内所有读取点都改走这两个访问器。
- `build()` 对草稿 build 不再materialize列表。
- `forward_fused_infer_attention` 按类型分派：device 张量 → v2 算子，list → 原来的 v1 调用不动。
- `full_graph_fia` 和 C8 路径只有 v1 算子，显式materialize，行为不变。

### 8.5.2 正确性（全绿）

| 判据 | 结果 |
|---|---|
| 算子级：v1+list / v2+list / v2+device张量，9 种形状（1 bonus + 8 masked query、1~64 请求、对齐与非对齐 KV 长度） | **逐位完全相同** |
| 运行时探针：每 rank 42000 次 build | **d2h = 0**（同步全部消失） |
| 运行时探针：逐元素比对 host 列表 vs device 张量 | 42000 × 4 rank = **168000 次，零偏差** |
| DSpark 接受率（自然语言 prompt、贪心） | 2.356 vs 未修 2.376 —— **没有退化** |
| 单测 | 打补丁树 38 passed；未打补丁树 5 个新用例全 fail |

> ⚠️ 算子级测试第一版全是 NaN，原因是我用 bf16 造 mask；运行时真正用的是
> `get_splitfuse_attn_mask()` 的 **int8 上三角** mask。换对之后 9 个形状全部 nan=0。

### 8.5.3 但是：性能不达预期，原因是算子

同会话三臂（T_base / T_fix / T_draft），中位 ITL：

| 并发 | T_base | T_fix | **T_draft** | draft vs base |
|---|---|---|---|---|
| 1 | 59.42 / 57.23 | 57.61 / 56.62 | **54.54 / 54.35** | **−6.7%** |
| 4 | 67.69 / 65.61 | 65.49 / 65.08 | **63.08 / 62.88** | **−5.5%** |
| 8 | 74.68 / 74.82 | 74.36 / 75.12 | **73.44 / 73.93** | **−1.4%** |

同步全没了，为什么只有这么点？直接量算子（每次迭代换一组不同的 KV 长度，
保证两边都不能复用 tiling）：

| num_reqs | v1 + list | v2 + list | **v2 + device 张量** | 倍数 |
|---|---|---|---|---|
| 8 | 272.6 µs | 273.0 µs | **553.8 µs** | 2.0× |
| 32 | 290.6 µs | 296.2 µs | **1362.4 µs** | 4.7× |
| 64 | 321.8 µs | 345.4 µs | **2569.6 µs** | 8.0× |

- **v2 和 v1 在传 list 时一样快** ⇒ 不是算子版本的问题；
- 传张量就贵，而且**开销随请求数线性增长，约 +35 µs/请求**
  ⇒ 绑定层在**逐个请求把长度从 device 读回来**；
- 预分配 workspace 不解决（500 vs 535 µs）；把 qlen 也换成张量更糟（850 µs）；
  连 **CPU 张量**都比 list 贵（210 vs 137 µs）⇒ 是"张量→内部数组"的逐元素转换。

⇒ **省下一次批量 D2H，换来 num_reqs 次标量回读。** 并发越大越亏，
c=8 已经基本打平，再大就是净负。所以这条分支**不建议合入**，只作为证据推上去。

### 8.5.4 那 −34% 到底怎么拿

天花板探针走的是"**list 路径 + host 侧就能算出来的值**"（乐观上界），
所以它既没有 D2H、也没有张量绑定开销 —— 这才是 50.0 ms / +31% 的来源。
要在**正确**的前提下复制它，只有两条路：

1. **算子侧**：让 FIA 接受 device 侧长度而不做逐请求回读（这是算子/绑定层的事）；
2. **框架侧**：让 speculator 在**不等目标模型前向**的前提下给出草稿块的精确 host 长度。

   > ### ⚠️ 这条判断要分 runner 说 —— 我先写错了一次，又纠枉过正了一次
   >
   > 原话"拒绝数只在 device 结算、host 拿不到"，**对 MRV1 是错的，对 MRV2 基本是对的**。
   >
   > **MRV1 + 同步调度：能，而且免费。** vLLM 为了吐出输出 token，本来就每步把接受的
   > token id 列表 materialize 到 host，所以
   > `rejected = num_draft + 1 - len(valid_sampled_token_ids_cpu[i])`，零额外代价。
   > 18 机 `/data2/dflash2-adapt/vllm-ascend`（**别人的树，2026-08-21**）已经这么实现了：
   >
   > ```python
   > # worker/model_runner_v1.py:1690  _get_current_num_rejected_tokens_cpu
   > # "Synchronous bookkeeping has already materialized the accepted token
   > #  IDs on the host. Prefer those lists so DFlash does not wait for a
   > #  second D2H copy of the same per-request counts."
   > if valid_sampled_token_ids_cpu is not None and len(...) == num_reqs:
   >     valid_counts = [len(ids) for ids in valid_sampled_token_ids_cpu]   # 免费
   > else:
   >     valid_counts = self._get_valid_sampled_token_count()               # event 兜底
   > rejected_counts.append(num_draft + 1 - num_valid)
   > ```
   >
   > 再由 `dflash_proposer._update_markov_seq_lens_cpu` 把 host 镜像修成精确值并置位
   > `parallel_drafting_seq_lens_cpu_valid = True`（拿不到就 `return`，保留 device 路径）。
   >
   > **MRV2（0.28）：拿不到，而且是设计使然。**
   > `vllm/v1/worker/gpu/model_runner.py:1773-1835`：
   >
   > ```python
   > model_runner_output = ModelRunnerOutput(..., sampled_token_ids=None)
   > # Start async output copy here so that it can overlap with speculator proposal.
   > async_output = AsyncOutput(...)
   > ...
   > draft_tokens = self.speculator.propose(..., num_sampled, num_rejected, ...)  # 都是 device 张量
   > ```
   >
   > host 侧**根本没有** `sampled_token_ids`（显式 `None`），而那次 D2H 是**故意**排在
   > `propose` 之前、让它与草稿提议重叠。要在草稿 build 里等它，等于把 MRV2 专门设计出来的
   > 那层重叠又抹掉。
   >
   > ⇒ **正确的结论：**
   >
   > | runner | 草稿块的精确 host 长度 |
   > |---|---|
   > | MRV1 + 同步调度 | **免费拿得到**，已有实现（18 机那棵树，别人做的） |
   > | MRV1 + async 调度 | 没有免费列表，退回 event 兜底 |
   > | **MRV2（我在 A2 测的就是这个）** | **拿不到**，除非取消 MRV2 的重叠设计 |
   >
   > 所以 A2 上那 +31% 在 MRV2 下仍然只有两条路：算子侧（第 1 条）或 §8.5.5 的有损路。
   > **但 MRV1 那条是真的通了。**
   >
   > 顺带：他们的 `parallel_drafting_seq_lens_cpu_valid: bool = False`
   > （`attention/utils.py:245`，默认 False、producer opt-in、拿不到就保留 device 路径）
   > 与我的 `seq_lens_cpu_is_exact` **一一对应** —— 两边独立收敛到同一形状，
   > 算是这个设计正确性的旁证。**那份实现只在 adapt 树里，真正跑着的 runtime 树是未修的 stock。**

   **但这个模式在本仓库里已经有完整实现了，不用从零设计** ——
   MRV1 的 `_copy_valid_sampled_token_count`（`model_runner_v1.py:1777`）+
   `_correct_optimistic_seq_lens_cpu`（:1749）就是"上一步末尾在副流上异步 D2H →
   下一步开头 event.synchronize() → 在 host 上把乐观值修正成精确值"，
   而且 docstring 明确论证了**稳态下那次 synchronize 是空操作**。
   MRV2 侧的对应件是 `worker/v2/model_runner.py:672-687`。

   所以难点不是机制，而是**时序**：现有实现修正的是**上一步**的拒绝数，
   草稿 build 要的是**这一步**的。要么把草稿 build 推迟一步（改的是投机解码的流水，
   不只是 metadata 管线），要么就只能走第 1 条或 §8.5.5 那条。

### 8.5.5 第三条路：草稿那一路根本不需要精确值 —— **2026-09-11 已实测验证**

把代码读完之后，我把自己第一轮的一个用词打了个问号：我一直把天花板探针叫“**故意写错**”。
它确实喂给草稿一个偏长的 KV 长度（乐观上界 = 假设草稿全接受，比真值长 `num_rejected`）。
但这个错误落在**哪里**，决定了它是不是真的不能用：

- 该长度只进 `actual_seq_lengths_kv`，**只影响草稿模型读 KV 的范围**；
  写 KV 的 `slot_mapping` 是 Triton 核单独算的（`dspark_proposer.py:271-295`），不经过它。
- 草稿的输出只是**候选 token + draft logits**，全部进拒绝采样：
  `worker/v2/model_runner.py:597-603` 走 `rejection_sampler(logits, input_batch, speculator.draft_logits)`。
- 标准推测采样的保证是：**只要接受判据和回退分布用的是同一个 q，输出分布就精确等于
  目标模型的 p，与 q 好坏无关**。这里正是同一个（退化的）q 同时进了两边。

⇒ **喂错草稿的 KV 长度不会改变输出分布，只会降低接受率。**
那么 −34% ITL / +31% 吞吐就不一定是“拿不到的上限”，而可能是一个**可以开关控制的有损选项**：
用一点接受率换掉整条同步。而且它不需要算子改、也不需要框架改，今天就能做。

**但我手上的数据支撑不了这个结论 —— 而且原因很具体，不是“数据少”，是量错了地方：**

第一轮确实测过三臂的接受率，但那是在 `--dataset-name random` 上：

| 臂（第一轮，random 数据集） | 平均接受长度 |
|---|---|
| S_base（未修） | 1.706 |
| S_fix（**正确**） | 1.654 |
| S_ceil（喂上界） | 1.663 |

**正确的那一臂反而比喂错的还低** ⇒ 三个数差异全在噪声里。原因很确定：8 个投机 token 只接受了
~0.7 个，已经贴地板，再坏也掉不下去 —— random 本来就把接受率压到底
（见 [[dflash2-random-dataset-reverses-verdict]]）。按 [[null-results-need-a-power-check]]：
**在地板上读到“没变化”，只说明尺子坐到底了，不是证据。**

**关键是：第二轮已经把有功效的尺子造出来了，只是没量这一臂。**
`scripts/accept_probe.py`（自然语言 prompt、贪心、并发 1）在同样的模型上读出接受长度 **2.376**，
离地板很远，而且它已经成功判过草稿走 device 张量那一臂（2.356 vs 2.376，判定无退化）。
**它从来没跑过天花板臂。** 所以缺的不是一轮大实验，是**一次 15 分钟的补测**。

`scripts/runall5.sh` 就是这次补测（base / fix / ceiling 三臂，每臂两次探针，~45 分钟）：

```bash
bash /nt/runall5.sh                       # 三臂
python3 /nt/accept_compare.py C_base C_fix C_ceil
```

它给两个互相独立的判据：

1. **文本判据（更强，先看这个）**：`temperature=0` 时拒绝采样退化成“与目标模型贪心 token 精确比对”，
   所以**不管草稿多烂，三臂的输出必须逐字节相同**。这正是上面那个论断的可证伪形式。
   `runexp5.sh` 为此设了 `HCCL_DETERMINISTIC=true`，否则 HCCL all-reduce 跨调用不确定、
   贪心 decode 本身就不可复现，同一个二进制都过不了（[[ascend-tp-moe-nondeterminism]]）。
   **文本一旦不同，下面的接受率不用看了 —— 先解释文本为什么变。**
2. **接受率判据**：每臂两次探针，用臂内跨度当噪声尺。

决策规则：

- 文本相同 + 接受率掉 < ~5% ⇒ **+31% 吞吐今天就能拿**，做成
  `VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1` 之类的开关（代码已经在 `nt_patch.py --ceiling` 里）；
- 接受率明显掉 ⇒ 这条路死，回到 §8.5.4 的算子侧 / 框架侧；
- 文本不同 ⇒ 我上面的推理有洞，优先查这个。

### 8.5.5.1 实测结果（A2，三臂，2026-09-11）

停掉占卡的服务后跑完了 `runall5.sh`（C_base / C_fix / C_ceil，每臂两次贪心探针）。

#### 判据一：输出文本 —— **决定性，通过**

```
=== TEXT (temperature 0; every arm must match the reference) ===
  C_base   rep1/rep2  identical to C_base (12 prompts)
  C_fix    rep1/rep2  identical to C_base (12 prompts)
  C_ceil   rep1/rep2  identical to C_base (12 prompts)
```

**3 臂 × 2 次 × 12 个 prompt × 160 token，全部逐字节相同**（`HCCL_DETERMINISTIC=true`）。
C_ceil 喂的是**语义上错的**（偏长的）KV 长度，输出却和正确基线一字不差
⇒ **推测采样那条数学保证在真机上成立：草稿再烂也不改变输出。**

#### 判据二：接受率 —— 要用对估计量

`runexp5.sh` 里按窗口平均 `Mean acceptance length` 的那个读数**功效不够**：

| 臂 | rep1 / rep2 | 均值 | 臂内跨度 |
|---|---|---|---|
| C_base | 2.448 / 2.494 | 2.471 | 0.046 |
| C_fix（**正确**，内部对照） | 2.772 / 2.454 | 2.613 | **0.318** |
| C_ceil | 2.058 / 2.288 | 2.173 | 0.230 |

C_fix 是**构造上就精确**的臂，真值差应当为 0，却读出 **+5.7%** ⇒ 这把尺子在 n=2 下
噪声就有 ±0.3，**分辨不出 −12%**。原因也清楚：那些行是按**日志窗口**平均的，
而窗口数本身在变（4 个 vs 5 个），部分窗口是空载的。

换成**把原始计数 pool 起来**（每行都带 `Accepted: N tokens, Drafted: M tokens`，
`accept_len = 1 + K·ΣAccepted/ΣDrafted`，K=8），每臂约 37000~40000 次草稿：

| 臂 | Accepted | Drafted | 草稿次数 | **accept_len** | vs base |
|---|---|---|---|---|---|
| C_base | 21516 | 300272 | 37534 | **1.5732** | — |
| C_fix（正确，对照） | 21143 | 303960 | 37995 | **1.5565** | **−1.07%** ← 噪声/系统偏差地板 |
| C_ceil | 18909 | 320160 | 40020 | **1.4725** | **−6.40%** |

**内部一致性自检**：总产出 token = Accepted + 草稿次数（每次一个 bonus）
= 59050（base）vs 58929（ceil），**只差 0.2%** —— 两臂干的活一样多；
ceil 是**多花了 6.6% 的 decode 步**（37534 → 40020）才产出同样多的 token，
这正是接受率下降该有的样子。两个独立的量互相对上了。

⇒ **接受率代价 ≈ −6.4%**，而对照臂只偏 −1.07%，**信噪比约 6 倍**。

### 8.5.5.2 结论：这条路是划算的

| | 代价 | 收益 |
|---|---|---|
| 输出正确性 | **零**（72 份输出逐字节相同） | — |
| 接受率 | **−6.4%**（≈ 多 6.6% 的 decode 步） | — |
| c=8 中位 ITL | — | **50.0 ms vs 75.4 ms（−34%）** |
| c=8 吞吐 | — | **179.8 vs 136.0 tok/s（+31%）** |

⚠️ **+31% 是净值，没有重复计算** —— 天花板臂的基准本来就是带着这 6.6% 额外步数跑出来的。

⇒ **天花板不是"拿不到的上限"，是一个可以开关控制的有损选项。**
建议做成 `VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1` 之类的显式开关，默认关，
文档写明"用 ~6% 接受率换 ~30% 吞吐，输出分布不变"。代码就是 `nt_patch.py --ceiling` 那几行。

### 8.5.5.3 已实现为分支 `feat/16271-approx-draft-kv-opt-in`（`367a4673`）

不再是 `nt_patch.py --ceiling` 那种"永不上线"的探针，而是一个正经的可选开关：

- `VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV`，**默认 0**（`vllm_ascend/envs.py`）
- `AscendCommonAttentionMetadata.seq_lens_cpu_is_approximate`，默认 False。
  **故意与 `seq_lens_cpu_is_exact` 分开** —— 后者必须继续表示"等于 device 张量"，
  以免任何真正需要精确值的地方被这个近似骗到
- `worker/v2/attn_utils.build_attn_metadata` 只在**草稿 build**（`seq_lens_np is None`）
  且拿得到上界时置位；目标 build 两种情况下都不受影响
- `attention_v1.py` 的 build 接受两个 flag 中任意一个作为"可以从 host 取列表"的许可

单测 5 个（2 个在 `test_attention_v1.py`，3 个在 `test_attn_utils_v2.py`），
在镜像里验过：**打补丁全 PASS，只打 `--fix` 不打这个则 5 个全 FAIL** ⇒ 判据有区分度。

> ⚠️ **溯源上的一点差异，要说明**：镜像里的 `envs.py` 比 main 落后约 54 个 commit，
> **没有 `_strict_binary_env` 这个 helper**。所以在机器上验证时，
> `approx_patch.py` 注册这个环境变量用的是等价的
> `lambda: os.getenv(...) == "1"`，而**分支里用的是 `_strict_binary_env`**（main 上有）。
> 逻辑等价，但**这一行的字节不是机器上跑过的那一行** —— 另外 3 个文件的改动逐字节相同。

### 8.5.5.4 近似误差量在源头（2026-09-11 实测）

开着开关跑一条带探针的腿（`VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1` + `NT_SEQ_LENS_CHECK=1`），
每 rank 2000 次 build：

```
[NT_PROBE] calls=2000 d2h=0 checked=2000 mismatch=404
           by_src={'np': [1566, 1566, 0], 'ub': [434, 434, 404]}
```

- **`d2h=0`** —— 开关确实把同步全去掉了
- 目标 build 1566 次，**0 偏差**（精确，与 §7.1 一致）
- 草稿 build 434 次（21.7%），**404 次偏（93%）**

偏差分布（`delta = host − device`，**全部为正**，即一律偏长）：

| delta | +1 | +3 | +4 | +5 | +6 | +7 | **+8** |
|---|---|---|---|---|---|---|---|
| 次数 | 8 | 4 | 8 | 8 | 12 | 12 | **28** |

**均值 ≈ 5.9，众数 +8。**

> ⚠️ 当时算式对不上（我按 `delta = rejected − num_query_per_req` 推，应当为负，实测全正），
> 所以没写修正公式。**下一节已经把它测清楚了，上面这个 5.9 也是有偏的**（见 §8.5.5.5）。

---

### 8.5.5.5 代数闭合：`delta` 就是这一步的拒绝数（2026-09-12 实测）

#### 先修两个测量缺陷

1. **上一节那个 5.9 是有偏的。** 探针里有 `if s["mismatch"] <= 20:` 的硬上限
   （`nt_patch.py` 探针段），所以打印出来的是**跑起来最早的 20 条**，不是这一批的抽样。
   改用 `seqtrace_patch.py` 把**每一次 build** 落 CSV（`build,approx,num_reqs,idx,host,dev`）。
2. **`card_watch2.sh` 的等卡判据是坏的。** `cards_ready()` 里的 HBM 检查遍历
   `npu-smi info` **整份输出**里所有 `<used>/ 65536` 字段，而不是只看 4-7。
   卡 0-3 长期被别的容器占着 ~59.9 GB ⇒ 那个循环恒为 false。
   watcher 09-12 02:54→08:54 白等 6 小时**不是因为卡忙**，是因为它判不出来。
   已改成按卡号 awk 解析（`launch_leg.sh`）。

#### 分解探针的结果（`leg_decomp.sh`）

```
[NT_DECOMP] n=1 mirrored=True src_mirror=[62] dev_seq_lens=[55] seq_lens_cpu=[62]
            _seq_lens_cpu=None upper_bound=[62] is_exact=False is_approx=True
```

- `seq_lens_cpu == upper_bound == src_mirror`，且 **`_seq_lens_cpu is None`**
  ⇒ MRV2 下 `dspark_proposer.py:310-315` 那段改 host 镜像的代码**确实没执行**（与 §8.5 的代码审计一致）
- 草稿 build 读的就是 `seq_lens_cpu_upper_bound`

#### 无偏直方图（`leg_trace.sh`，1417 次草稿 build / 10745 次目标 build）

| | delta 分布 |
|---|---|
| **目标 build（精确镜像）** | **10745 / 10745 全 0** —— 对照通过，精确镜像确实精确 |
| **草稿 build（近似镜像）** | 0:2.96% 1:1.55% 2:1.41% 3:1.41% 4:3.53% 5:6.21% 6:9.32% 7:23.43% **8:50.18%** |

`mean = 6.7509 ± 0.0510`，`min=0`，`max=8`。

#### 判据：两台独立仪器对上了

| 量 | 值 |
|---|---|
| 同一次运行的接受率计数器反推 `K·(1−ΣAccepted/ΣDrafted)` | **6.7808**（Accepted=1641 / Drafted=10768） |
| trace 实测 `mean(delta)` | **6.7509 ± 0.0510** |
| 差 | 0.030 = **0.59 SE** |

⇒ **`delta` 就是这一步被拒绝的 token 数**，取值 `[0, K]`，`K = num_speculative_tokens = 8`。

之前推不通的原因很简单：我假设 host 端装的是 `n`（目标前向**之前**的长度），
实际**乐观上界里已经把 `+ num_query_per_req` 折进去了**：

```
host   = n + q            ← 乐观：假设这 q 个 token 全被留下
device = n + q − rejected ← 真值
delta  = rejected ∈ [0, K]
```

`rejected` 的上界是 `K` 而不是 `q`，因为 bonus token 恒被接受 ⇒ `accepted ≥ 1`。
**50.18% 的步恰好 `delta = 8`**，即整步只留下 bonus token、草稿颗粒无收。

#### 由此得到的估计器排名（n=1405 对，lag-1 ρ=0.3217）

| 估计器 | mean err | mean\|e\| | RMS | 恰好命中 | 方向 |
|---|---|---|---|---|---|
| 假设 0 拒绝（现在的 `ceil` 臂） | +6.809 | 6.809 | 7.049 | 2.1% | **恒 overshoot** |
| **`ub − K`（假设全拒）** | −1.191 | **1.191** | 2.179 | **50.6%** | **恒不 overshoot** |
| **上一步的拒绝数（= 选项 1）** | +0.059 | 1.317 | 2.185 | 41.6% | 双向 |
| `ub − (K−1)` | −0.191 | 1.204 | **1.834** | 23.6% | 50% 超 1 |

> **这推翻了"用上一步拒绝数"是首选的预期。** delta 的 lag-1 自相关只有 0.32，
> 而分布极度双峰（一半的步恰好等于 K），所以**一个零状态的常数比看上一步更准**，
> 且不需要任何跨步管线（MRV2 下拿上一步拒绝数要新开 `AscendCommonAttentionMetadata`
> 字段 + producer 侧改动，见 §8.5 的 MRV1/MRV2 分野）。

**关键性质：`c0` / `ub−(K−1)` / `ub−K` 三臂的下发行为完全相同**（草稿 build 都是零 D2H），
只有传进 `actual_seq_lengths_kv` 的那个数不同 ⇒ **吞吐必然一样，唯一变量是接受率**。
所以 A/B 只需要接受率 + 文本判据，不需要再跑一遍 benchmark。

---

### 8.5.5.6 四臂 A/B：**修正拒绝数这条路走不通**（2026-09-12 实测）

`leg_ab.sh`，同一台机同一个镜像连跑四臂，每臂 1 个实例 × 2 遍探针
（12 prompt × 256 token，贪心 + `HCCL_DETERMINISTIC=true`）：

| 臂 | 传给草稿的 `actual_seq_lengths_kv` | Accepted/Drafted | `accept_len` | vs base |
|---|---|---|---|---|
| `base` | 精确（device 路径，`.tolist()`） | 3714/18496 | **2.6064** | — |
| `c0` | `ub`（= 假设 0 拒绝，即已发布的开关） | 3375/20904 | 2.2916 | **−12.08%** |
| `cK1` | `ub − (K−1)` = `ub − 7` | 3321/20136 | 2.3194 | −11.01% |
| `cK` | `ub − K` = `ub − 8` | 3298/22304 | 2.1829 | **−16.25%** |

#### 文本判据：**通过**

8 个 json（4 臂 × 2 遍）**sha256 全部相同**，`reference sha=df34ea140872348c`，
12 条补全 11544 字符逐字节一致 —— 包括最差的 `cK` 臂。
⇒ 推测采样的保证在真机上再次成立，这仍然是纯接受率的交易。
（同时每臂两遍自身也完全一致 ⇒ 实例内可复现，判据本身站得住。）

#### 显著性（按**步**算，不是按 token）

一步里的 8 个草稿 token 是前缀式接受、强相关，用 token 数当 n 会虚报显著。
用 `sd(每步接受量) = sd(delta) = 1.92`（§8.5.5.5 实测）、`n_步 ≈ Drafted/8`：

| 对比 | Δaccept_len | SE | σ |
|---|---|---|---|
| base − c0 | 0.3148 | 0.0548 | **5.74** |
| base − cK1 | 0.2870 | 0.0553 | **5.19** |
| cK1 − c0 | 0.0278 | 0.0536 | 0.52 ← **噪声** |
| c0 − cK | 0.1087 | 0.0523 | 2.08 ← 勉强 |

⚠️ 每臂只有 **1 个实例**，而 [[ab-session-is-the-variance-unit]] 说段间差远大于段内 σ，
所以 0.52σ 和 2.08σ 这两个**都不该单独下结论**；能站住的是
"两个误差统计好得多的臂都没把接受率拿回来"这个**联合**事实。

#### 这否定了"修正拒绝数"的前提

`cK` 相对 `c0`：**恰好命中率 50.6% vs 2.1%（24×）、mean|e| 1.19 vs 6.81（5.7×）**，
接受率**反而更低**。`cK1` 同样大幅改善误差统计，接受率只动了 0.52σ。

按方向拆每步代价（用 `c0`/`cK` 两个方程反解，base 每步接受 1.6064 个草稿 token）：

| 该步 KV 长度 | 该步接受量 | 相对 base |
|---|---|---|
| 恰好正确 | 1.6064 | — |
| **偏长** overshoot | 1.2852 | −20.0% |
| **偏短** undershoot | 0.7496 | **−53.3%** |

**偏短比偏长贵 2.6 倍。** 机制自洽：偏长是在真实上下文之外多读几个**被回滚的草稿 KV**
（真实数值、只是错的，稀释注意力）；偏短是**读不到刚刚被接受的那几个 token**，
而那正是预测下一个 token 最需要的。这也解释了为什么"看起来更安全"的
`cK`（恒不 overshoot）反而最差。

> **恒不 undershoot 的常数只有 `est = 0` 一个。** 因为 `rejected` 有 2.96% 的步等于 0，
> 任何正常数在那时都会偏短。也就是说**已经发布的 `c0` 恰好就是唯一那个安全方向的选择**。

#### 对"用上一步的拒绝数"（选项 1）的推论

trace 里那个估计器的残差分布是 41.6% 命中 / 29.46% 偏短 / 28.94% 偏长，代入上表：

```
0.416×1.6064 + 0.2946×0.7496 + 0.2894×1.2852 = 1.2610  →  accept_len ≈ 2.261
```

**低于 `c0` 的 2.2916。** 这个模型只看方向不看幅度，对 `cK1` 低估了 0.10（1.9σ，
因为 `cK1` 的 overshoot 只有 +1、比 `c0` 平均 +6.8 便宜），所以真值大概落在
**2.26 ~ 2.36，最好情况是和 `c0` 打平**，代价是 MRV2 下要新开
`AscendCommonAttentionMetadata` 字段 + producer 侧跨步管线。

**但最强的证据不是这个外推，是那个直接反证：`cK` 有一半的步拿到了完全正确的
KV 长度，接受率一点没回来。** "把拒绝数估得更准 ⇒ 买回接受率"这个前提被证伪了，
它对**任何**估计器成立，不只是上一步那个。

⇒ **结论：不实现选项 1。** 有损开关就保持 `c0`（`feat/16271-approx-draft-kv-opt-in` @ `367a4673`）
不动 —— 它既是实测最好的有损臂，又是唯一不会偏短的那个。
那 +31% 吞吐的真正出路仍然只有 §8.5.3 的算子/绑定层（去掉逐请求回读）。

#### 复现

```bash
bash /data01/nt-work/launch_leg.sh leg_ab.sh ab.out     # 按卡号查 4-7，20s 复检后起
bash /data01/nt-work/wait_ab.sh 560                     # 远端阻塞等（本地长轮询会被回收杀）
# 单独看某一臂： docker exec nt-dspark-mrv2 python3 /nt/pooled_acc.py /nt/logs/serve_<arm>.log
# 文本判据：     docker exec nt-dspark-mrv2 python3 /nt/text_diff.py /nt/logs/accept_*.json
```
全程 ~35 分钟（4 臂 × (起服 ~340s + 2×~72s 探针)）。跑完 `trap cleanup EXIT`
无条件 `stop.sh` + `nt_patch.py --revert`，实测收工后树回到 pristine
`md5=3ffdd3667b206531740ed46c69285f41`、卡 4-7 回到空载 ~3432 MB。

---

### 8.5.5.7 选项 1 已实装并实测：**误差砍掉 99%，接受率反而更差**（2026-09-12）

§8.5.5.6 是靠 `ub−7`/`ub−8` 外推出"选项 1 不会赢"。用户要求直接测，于是实装了。

#### 实现：零 device 交互就能拿到上一步的拒绝数

不需要等任何拷贝，也不需要碰 MRV2 那条被刻意重叠的异步输出路径 —— 两个 host numpy 数组就够：

```
ub_{t−1}   = computed_{t−1} + scheduled_{t−1}
computed_t = computed_{t−1} + accepted_{t−1}     ← 调度器已修正
⇒ ub_{t−1} − computed_t = scheduled_{t−1} − accepted_{t−1} = rejected_{t−1}
```

`computed_t` 确实是修正过的**不是假设**：否则误差会跨步累积，而 §8.5.5.5 实测 delta
恒在 `[0, K]` 且上界恰好取到 K。`0 ≤ d ≤ K` 同时充当有效性校验
（槽位被新请求复用、或该步不是纯 decode 时落在区间外 ⇒ 退回 0 = 原上界）。

挂钩点 `AscendDSparkSpeculator.propose`（`scripts/prevreject_patch.py`），
`finally` 里把上界换回去，草稿 build 之后没人看得到这个估计值。

> ⚠️ **第一版挂错了地方，而且是静默的。** 先挂在 `build_draft_attn_metadatas` 上，
> 那个函数只有 `run_fullgraph` 会走；本配置 `--speculative-config` 里带
> `"enforce_eager": true`（当初为规避 drafter 图捕获的 FIA 561002 加的，见
> [[a2-qwen36-dspark-test-site]]），drafter 根本不进图模式 ⇒ **钩子一次都没触发**。
> 那一轮读出 `cprev=2.3175` vs `c0=2.3025`，看起来像个漂亮的小改善，
> **实际上只是 c0 又跑了一遍**。是正向对照（日志里 `[NT_PREV]` 行数=0）抓住的，
> 不是数字 —— 数字完全说得通。

#### 第二个静默 bug：有效性判据的上界差 1

第一次跑出来命中率只有 49.5%，看着像"workload 的性质"。诊断腿（`leg_diag.sh`，
把 `d` 的原始直方图打出来）证明**不是**：

```
[NT_PREV] calls=1200 reqs=1263 hit=647 (51.2%) nohist=64 dummy=1
          d_hist(top)=[(8,541),(7,280),(6,135),(5,72),(4,53),(0,45),(3,22),(1,20)]
```

直方图里的 `d` **全都落在 `[0,8]`**，加起来 ~1199，而 `hit` 只有 647 ——
按累积和对，647 恰好是 `d ≤ 7` 的累计（658）。所以判据用的上界是 **7 不是 8**：
我写的 `k = self.num_query_per_req - 1`，而 **speculator 上 `num_query_per_req` 是 8 不是 9**
（日志已确认：`cap k=8 from speculative_config (num_query_per_req=8)`）。

`d = scheduled − accepted = 9 − accepted ∈ [0, 8]`，**`d=8` 占 45%（一半的步只留下
bonus token）**，被整个当成"槽位失效"丢掉了 ⇒ 那一版**恰好从不修正误差最大的那一半步**。
改成从 `speculative_config.num_speculative_tokens` 解析，并把解析结果打进日志
（否则同一个错误还会再藏一次）。

#### 剂量-反应：**估得越准，接受率越低**

修好后覆盖率 96.9%（剩下的是 `nohist`，每个请求第一步没有历史），
`mean_est=6.748` 与真实拒绝均值 6.65~6.78 吻合。今天四条腿给了
`base`/`c0` 各 **4 个独立实例**，所以方差单位用实例而不是保守估计
（[[ab-session-is-the-variance-unit]]）：

| 修正覆盖率 | 实测残余误差均值 | 残余误差范围 | accept_len | vs `c0` |
|---|---|---|---|---|
| 0%（`c0`，n=4） | 6.6483 | [0, 8] 单边 | **2.2874 ± 0.0129** | — |
| 49.5%（k 写错那版） | 3.9149 | [−7, 8] | 2.2770 | −0.72σ |
| **96.9%（修好）** | **0.0670** | [−8, 8] | **2.2077** | **−5.53σ** |

`base` 四实例 **2.6063 ± 0.0073**。文本判据：6 份 json **sha 全同**。

**估计器现在几乎完美 —— 平均误差 6.6483 → 0.0670，砍掉 99% —— 接受率却掉得更多。**
而且是**单调的剂量-反应，方向是反的**：修正得越多越差。
这不是"没效果"，是一个符号相反的正结果，比任何单点比较都强。

机制上完全自洽（§8.5.5.6 的方向不对称）：满覆盖的上一步估计器**无偏但双向**，
约 30% 的步偏短。它拿"便宜的单边偏长"换成了"贵 2.6 倍的双向偏短" ——
残余误差的 **sd 反而从 1.96 涨到 2.89**，而涨的那部分正好落在贵的那一侧。

**结论（实测，不是外推）：修正这个拒绝数买不回接受率，做得越好越亏。**
四个互相独立的估计器 —— 两个常数 + 半覆盖和满覆盖的自适应 —— 全部失败。
有损开关维持 `c0`（`feat/16271-approx-draft-kv-opt-in` @ `367a4673`）不动：
它既是实测最好的有损臂，又是**唯一恒不偏短**的那个（因为真值有 ~3% 的步等于 0，
任何正的估计都会在那时偏短）。

#### 复现

```bash
bash /data01/nt-work/launch_leg.sh leg_prev2.sh prev2.out   # ~26 分钟，3 臂
bash /data01/nt-work/wait_prev2.sh 560
```
`prevreject_patch.py` 改的 `dspark/speculator.py` **不在 `nt_patch.py` 的 `FILES` 里**，
所以它自带 `.nt-orig` 备份和 `--revert`，腿的 `cleanup` 两个都要调
—— 否则树会留脏给下一次跑。实测收工后两个文件都回到 pristine
（`3ffdd366…` / `34a8406b…`），卡回到 ~3432 MB。

**适用边界（别外推）**：贪心（temperature=0）下输出是逐字节相同；
temperature>0 时保证是**分布相同**而非逐字节相同。接受率代价只在
Qwen3.6-35B-A3B × DSpark × 8 spec tokens × 这组自然语言 prompt 上测过。

---

❗ 另外两个必须一并查的点（天花板探针跑完整 benchmark 没出事，但那不是证明）：
block_table 行数是否也按上界分配（否则 FIA 会报 561002，见 [[a2-qwen36-dspark-test-site]]）；
读到的多余 KV 位是否可能是未初始化显存（NaN 进 draft logits）。

---

---

## 8.6 MRV2 上还剩什么路：**"别把长度搬到宿主"在这套栈上是关的**（2026-09-13 实测）

§8.5.4 的结论是：MRV2 框架侧拿不到精确的 KV 长度（第 t 步的拒绝数在目标前向跑完之前
宿主上不存在），§8.5.5.6/8.5.5.7 又把"猜"这条路判死。剩下唯一无损的方向是
**根本不把长度搬到宿主**。这一节把它测到底。

### 8.6.1 为什么以前以为这条路是通的（这个判断是错的，此处更正）

之前的证据链是：

| 观察 | 当时的解读 |
|---|---|
| torchair GE converter 的 `actual_seq_kvlen: Optional[Union[List[int], Tensor]]` | 底层算子吃设备张量 |
| `nm -D` 在 `libopapi.so` 里找到 `aclnnInnerFusedInferAttentionScore**Tensor**GetWorkspaceSize` | 库里已经有接张量的入口 |
| eager 的 ATen 签名是 `at::OptionalSymIntArrayRef` | 卡住的只是 torch_npu 的绑定 |

**三条全部被源码推翻。** 从 `hicann/ops-transformer`（分支名 = CANN 版本，见
[[cann-op-source-ops-transformer-repo]]）读 `attention/fused_infer_attention_score/`：

1. **IR 把这两个输入标成 `ValueDepend`** —— `op_host/fused_infer_attention_score_def.cpp`：

   ```cpp
   this->Input("actual_seq_lengths")     .ParamType(OPTIONAL).ValueDepend(OPTIONAL)
   this->Input("actual_seq_lengths_kv")  .ParamType(OPTIONAL).ValueDepend(OPTIONAL)
   ```

   `ValueDepend` 的含义就是 **tiling 需要这个输入的"值"在宿主**。9.0.0 / master 两个分支
   都是这样，没有改过。

2. **`aclnnInner...TensorGetWorkspaceSize` 不是"接设备张量"的入口**，它是
   `aclnnFusedInferAttentionScoreV5Get**Max**WorkspaceSize` 用的。
   `op_api/aclnn_fused_infer_attention_score_v5.cpp` 里，那条路先把 `aclIntArray` 喂给
   `FakeArray()`：

   ```cpp
   outTensor = aclCreateTensor(shape.data(), shape.size(), ACL_INT64, nullptr,
                               0, ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
   //                                            ^^^^^^^ 数据指针是 null
   ```

   造出来的是**只有形状、没有数据**的假张量，用来算 acl graph 捕获期的最坏 workspace。
   真正执行的 `GetWorkspaceSize` 走的是非 Tensor 的那条，参数照旧是 `const aclIntArray *`。

3. **图模式也一样要宿主值** —— `op_graph/fallback_fused_infer_attention_score.cpp`：

   ```cpp
   const int64_t *actSeqData = fiaTensors.actualSeqLengthsGeKv->GetData<int64_t>();
   for (...) actualSeqInfo.actSeqArrayKv.push_back(actSeqData[i]);
   ```

   GE 在 tiling 之前已经按 `ValueDepend` 把它拷回宿主了。converter 签名里的 `Tensor`
   只是 Python 层的类型，不代表值留在设备上。

**唯一一处"值不在宿主也能活"的分支确实存在**，但它不是执行路径 ——
`op_host/fused_infer_attention_score_tiling.cpp:1584` 和 `flash_attention_infer_tiling.h`：

```cpp
if (actualSeqQ == nullptr || actualSeqKv == nullptr) { // tiling下沉场景
    ...
} else { faInfo.isTilingSink = true; }
...
ge::graphStatus FAInferTiling::DoTiling(FAInferTilingData &tilingdata) {
    FillBasicTilingData(tilingdata);
    if (!faInfo_.isTilingSink) {          // <- 下沉时整块分核逻辑被跳过
        FillSplitCoreTilingData(tilingdata);
        if (faInfo_.flashDecodeFlag) splitBN2S1GS2(tilingdata);
        else if (faInfo_.decodingFlag) SplitCoreDecodeBS1GN2(tilingdata);
    }
    FillWorkSpaceTilingData(tilingdata);
}
```

`isTilingSink=true` 时**一点分核都不做**，只填最坏 workspace —— 也就是只服务于上面
那个 `GetMaxWorkspaceSize`。所以它是一个"接口上容忍值缺席"的坑位，不是一条能跑的模式。

### 8.6.2 那换个算子呢：ATB 的 `_npu_paged_attention_splitfuse`

vllm-ascend 的 310P 后端调它的样子，形状上正好是我们要的：

```python
torch_npu._npu_paged_attention_splitfuse(
    query=query, key_cache=..., value_cache=..., mask=mask, block_table=block_table,
    seq_len=qlens,                       # query 长度：调度器给的，宿主上白拿
    context_lens=attn_metadata.seq_lens, # KV 长度
    ...)
```

而且 `_310p/attention/attention_v1.py:268` 还显式 `seq_lens.to(device=query.device)`，
看起来 KV 长度是走设备张量的。它也支持 q_len>1（splitfuse 本来就是给 chunked prefill 用的）。
掩码那一关也能绕：310P 那条 `get_splitfuse_mask()` 里有 `.tolist()`，但同样的行**可以完全在设备上**
按比较式构造，不需要任何宿主长度：

```python
limit = ((kvlen_dev - Q).view(-1, 1) + arange_q.view(1, -1)).reshape(-1, 1)
mask  = (cols.view(1, -1) > limit).to(dtype).mul_(-10000.0)
```

（顺带一条可复用的结论：先用 `tri.index_select(0, pos)` 做同样的事要 **444 µs**，
而比较式只要 **~180 µs** —— gather 在这颗芯片上是**按源张量大小计价**的，
源是 2048×2048=4.2M 元素，选几行不影响价钱。同 [[aclnn-scatter-cost-is-source-driven]]。）

### 8.6.3 实测：`context_lens` 必须在宿主

单卡独立探针（`scripts/sf_probe5.py`，不起 vLLM），drafter 形状
heads=4 / kv_heads=1 / D=128 或 256 / block=128 / q=9 / bf16：

| `context_lens` | `seq_len`(q) | 结果 |
|---|---|---|
| **device** | cpu | ❌ `param.cpp:29 tensor.hostData is null` → `build param from host tensor fail` |
| **device** | device | ❌ 同上 |
| cpu | cpu | ✅ `wall=53.7 µs`，对纯 torch 参考 `rel=0.00242`（与 FIA 同量级） |
| cpu | **device** | ❌ `param.cpp:101 qLensTensor inTensors(6) hostData is null` |

**ATB 的 `PagedAttentionOperation` 两个长度张量都要 `hostData`。**
也就是说这条路不是绕开 D2H，是把 D2H 换个地方（`.cpu()` 和 `.tolist()` 一样阻塞）。
310P 那句 `.to(query.device)` 之所以没炸，是因为在正常路径上
`AscendMetadata.seq_lens` 本来就是 **CPU 张量**（`attention_v1.py:398` 里
`seq_lens=seq_lens, seq_lens_cpu=seq_lens` 是同一个对象）；
只有 `parallel_drafting` 那一支（`:337-338`）才把它换成设备张量，然后在 `:349` 被
`tolist()` 掉 —— 这正是 #16271。

### 8.6.4 测量纪律：这一轮栽了两次，两次都是"静默的假结果"

1. **第一次**（round 3/4）splitfuse "跑通了"，`wall` 在 n=8/32/64、D=128/256 六种配置上
   **恒定 58 µs**。那是假的：同进程里前面的 FIA 臂已经失败，**ATB/aclnn 的报错是异步的**，
   之后的算子会直接返回、根本不碰输出缓冲区。表现是：`.abs().max()` 读出**负数**、
   两个确定性张量的 `torch.equal` 在同一份代码下时真时假、denormal 当成"误差为零"。
   ⇒ **一个进程只放一个臂**（[[perf-measure-one-shape-per-process]] 的另一个理由），
   而且判据里必须有一个恒定值的对照（这里是"耗时不随 n 变"）。

2. **第二次**：round 3 报"FIA 的 TND 不支持 headDim=256"
   （`ValidateNoRopeLayoutDim ... only 64/128/192 are supported`）。这个结论是错的 ——
   真因是我把 paged KV 缓存喂成了 4 维。FIA 在 TND 下要 **3 维**
   `[num_blocks, block_size, kv_heads*head_size]`；改成 3 维之后 D=256 一次通过，
   对纯 torch 参考 `rel=0.00254`。**算子的报错信息会指着一个无辜的参数。**

### 8.6.5 结论

| 层 | KV 长度能不能留在设备 | 判据 |
|---|---|---|
| vllm-ascend 框架 | ✅ 有（`seq_lens` 本来就在设备） | — |
| torch_npu eager ATen | ❌ `at::OptionalSymIntArrayRef` | 头文件 |
| aclnn 公开 API V2–V5 | ❌ `const aclIntArray *` | 头文件 + 仓库源码 |
| FIA 的 op IR / tiling | ❌ `ValueDepend`，tiling 读宿主值 | `*_def.cpp` + `fallback_*.cpp` |
| FIA 的 `isTilingSink` | ⚠️ 只给 MaxWorkspace，跳过全部分核 | `DoTiling()` |
| ATB PagedAttention / splitfuse | ❌ `hostData is null` | 实测四组合 |

**这台栈上每一个能用的注意力算子，tiling 都从宿主上的序列长度算出来。
所以 MRV2 上"无损去掉这次同步"不是 vllm-ascend 能改的事。**

框架侧还剩下的，就是已经交付的两件：
目标那一路的精确修复（`adfa304e`，已合入建议），
和草稿那一路默认关闭的有损开关（`367a4673`，输出逐字节不变、换 +31% 吞吐）。

**要拿回那 −34% ITL / +31% 吞吐，需要 CANN 侧一个具体的改动**（这是现在能提出的、
有证据支撑的最小请求）：

> FIA 的 FAI tiling 已经有 `isTilingSink` 这条"宿主上没有长度也能继续"的分支。
> 把它做成真正的执行模式（核在运行期从 GM 读 `actual_seq_lengths_kv`，
> tiling 用形状上界分核），并在 aclnn 上开一个
> `actualSeqLengthsKvOptional` 收 `aclTensor*` 的入口。
> vllm-ascend 这边的分支 `exp/16271-draft-seq-lens-device-tensor`（`fd6ab8ad`）
> 已经写好且验证过正确（9 形状逐位相同、d2h=0、168000 次比对零偏差、接受率不变），
> 入口一旦存在，它就是精确、零同步的解。

---

## 8.7 **拿回来的办法：长度给上界，精度用设备侧掩码补**（2026-09-13 实测，路通了）

§8.6 判死的是"把 KV 长度以设备张量喂进算子"。但**还有一个参数是设备张量**：
`atten_mask`。它不是 `ValueDepend`，所以它**可以合法地依赖设备上的 `seq_lens`**。

`c0`（已发布的有损开关）掉接受率的机制很具体：喂上界 `U` 之后，kernel 真的去 attend
了尾巴 `[L, U)` —— 也就是这一步刚被回滚的草稿 KV，softmax 被稀释。
那就把那段**用掩码盖掉**：

| 参数 | 喂什么 | 从哪来 | 要不要同步 |
|---|---|---|---|
| `actual_seq_kvlen` | 上界 `U = num_computed + K+1` | 宿主 numpy | **否** |
| `atten_mask[i, j, c]` | `1  iff  c > L_i − Q + j` | 设备张量比较式构造 | **否** |

`L` 全程只存在于设备上。宿主永远不知道它。

### 8.7.1 第一次尝试失败，原因在模板选择

TND 下直接换 `sparse_mode=0` + 三维掩码会被拒：

```
CheckFAIMask: "When attnMask is provided, sparseMode must be 3 or 4"
CheckFAIMaskShape: 最后两维必须 2048，前面的维必须 1
TilingProcess4SplitFuse -> tiling process for split fuse failed
```

因为 `IsUsingFAI()` 的第一个条件就是 `inputLayoutStr == "TND"`：TND 一律走 split-fuse
(FAI) 模板，而那个模板里逐请求掩码是**类别上不可能**的。

**但草稿这一路每个请求的 query 数恒等于 `K+1`=9**，所以
TND `[T, N, D]` 和 BSND `[B, S, N, D]` 是**同一块内存的两个视图**（`view`，零拷贝）。
换成 BSND 就离开了 FAI 模板，落到通用模板，而通用模板的
`MaskChecker::ValidateMaskDimAndShape` 接受

```
sparse_mode 0,  atten_mask [B 或 1, >=Q_S, >=KV_S]      (PA 下 KV_S = maxBlockNumPerSeq*blockSize)
```

正好是我们要的逐请求掩码。

### 8.7.2 实测（单卡独立探针 `scripts/maskfix2.py`，drafter 形状 heads=4/kv=1/D=256/q=9/bf16）

判据是对**纯 torch 参考**（用真实长度 `L` 算的 fp32 注意力）的相对误差。
`d`（每请求的拒绝数）随机取 `[0,8]`、`L` 随机，避免对称性让错掩码蒙混过关。

**无滑窗（`sparse_mode=3`，kv≈1024）**

| n | `tnd_ref` 精确长度（今天，但需要同步） | `bsnd_mask_U` 上界+设备掩码 | `tnd_c0` 已发布的有损开关 |
|---|---|---|---|
| 8 | rel 0.002929 / 153.7 µs | **rel 0.002761 / 149.5 µs** | rel 0.2662 |
| 32 | rel 0.002656 / 171.6 µs | **rel 0.002634 / 182.0 µs** | rel 0.2622 |
| 64 | rel 0.003275 / 201.6 µs | **rel 0.003275 / 199.3 µs** | rel 0.2662 |

**滑窗（生产真实配置：`sliding_window=2048` ⇒ `sparse_mode=4`，kv≈3000）**

| n | `tnd_swa_ref` 精确长度 | `bsnd_swa_U` 候选 | `tnd_swa_c0` |
|---|---|---|---|
| 8 | rel 0.003493 / 150.5 µs | **rel 0.003493 / 155.1 µs** | rel 0.1914 |
| 64 | rel 0.002919 / 201.1 µs | **rel 0.002919 / 303.3 µs** | rel 0.2749 |

- **精度：候选与"喂精确长度"逐位同级**（滑窗两档 rel 打印到小数点后 6 位完全相同）。
  而 `c0` 是 **0.19~0.27，差 60~100 倍** —— 这是第一次把它的损失量化到算子层。
- **成本：无滑窗基本持平（−2.7% / +6.1% / −1.1%）；滑窗下 n=8 +3.1%，n=64 +50.8%。**
  代价集中在"大 batch + 滑窗"，因为通用模板没有 FAI 那套 band 优化。
- 掩码构造 **~153–170 µs 且与 n 无关**，每步一次、5 层共用。

**账**：n=64 滑窗最坏情形，每步多 5×102 µs（5 层）+ ~170 µs 掩码 ≈ **680 µs 设备时间**，
换掉 **3286 µs 的宿主阻塞**。而且宿主阻塞是**下发气泡**（设备空转），
多出来的是**设备上的有效工作**。净向好，但必须端到端验证，不能只看算子。

### 8.7.3 一个必须照抄的语义：`pre_tokens=W` 保留的是 **W+1** 个位置

第一版滑窗掩码对不上（rel 0.157）。不是候选错，是我对 band 边界的理解错。
拿 FIA 自己当参考扫偏移：

```
mask 不带窗（对照）   rel_vs_tnd_sm4 = 0.5903
window=2046 (-2)      0.1856
window=2047 (-1)      0.1565
window=2048 (+0)      0.1568
window=2049 (+1)      0.005848   <-- 唯一对上的
window=2050 (+2)      0.1488
```

即 band 模式保留 `[limit−W, limit]`，**两端都闭**。实现里必须写 `window + 1`，
否则误差 0.157 —— **是 bf16 噪声底的 27 倍，但远没到会崩的程度，正是那种能一路混到生产的偏差**。

### 8.7.4 落地清单（改 `vllm_ascend/attention/attention_v1.py` 的 parallel_drafting 分支）

1. `seq_lens_cpu` 取**乐观上界**（`c0` 已有的路径），**删掉 `seq_lens.tolist()`**；
2. query 由 TND `[T,N,D]` `view` 成 BSND `[B,K+1,N,D]`（零拷贝），
   `actual_seq_lengths` 从**累积**改成**每请求的 q 长度**（BSND 语义不同）；
3. `sparse_mode` 3/4 → **0**，`atten_mask` 换成 builder 每步构造一次的
   `[B, K+1, S2]` int8 设备张量，`S2 = maxBlockNumPerSeq * block_size`：

   ```python
   limit = (seq_lens_dev - q).view(-1, 1) + arange_q.view(1, -1)      # [B, q]
   m = cols.view(1, 1, -1) > limit.unsqueeze(-1)                       # 因果 + 尾部
   if sliding_window:                                                  # band，注意 +1
       m |= cols.view(1, 1, -1) <= (limit.unsqueeze(-1) - (sliding_window + 1))
   atten_mask = m.to(torch.int8)
   ```

4. 掩码显存：`B*(K+1)*S2` 字节，`max_model_len=8192 / B=64 / K=8` ⇒ **4.7 MB**；
5. padding 请求（图捕获用）要一起补进 B 维，别让 `actual_seq_lengths` 和 B 对不上。

**还没验的（必须在上生产前做）**：
- 端到端 A/B（接受率 + ITL + 吞吐），判据用 §7.2 那套（每臂多实例，
  [[ab-session-is-the-variance-unit]]）；
- 目标模型那一路不动（`adfa304e` 已经是精确的），只改草稿这一路；
- n=64 + 滑窗那 +50.8% 在整网里占多少 —— 算子只占一步的一部分。

---

### 8.7.5 端到端 A/B：**接受率全额拿回，吞吐还比 `c0` 再高一截**（2026-09-13）

`scripts/maskfix_patch.py`（builder 一处锚点插入 + `forward_fused_infer_attention`
的 monkeypatch），`scripts/leg_mask.sh` 六臂回文序
`base1 c01 mf1 mf2 c02 base2`——每臂两个独立实例、在时间上对称摆放，
因为**重启才是方差单位**（[[ab-session-is-the-variance-unit]]），
单调漂移不能全落在一个臂上。每臂：`accept_probe`（接受率 + 逐字节文本）
→ bench 预热 → bench 正式（random，c=8，96 prompts）。

| 臂 | 吞吐 tok/s | 中位 ITL ms | pooled accept_len |
|---|---|---|---|
| base1 / base2 | 132.14 / 131.33 | 75.60 / 75.11 | 1.6340 / 1.5893 |
| c01 / c02 | 176.90 / 184.04 | 50.21 / 50.03 | 1.5149 / 1.5472 |
| **mf1 / mf2** | **185.17 / 193.02** | **49.86 / 50.19** | **1.6277 / 1.6547** |

| | 吞吐 | 中位 ITL | accept_len |
|---|---|---|---|
| `c0` vs base | **+37.0%** | **−33.5%** | **−5.00%** |
| **`mf` vs base** | **+43.5%** | **−33.6%** | **+1.83%** |
| `mf` vs `c0` | +4.8% | −0.2% | **+7.19%** |

**判据分离**：
- `mf` 的两个实例（1.6277 / 1.6547）**完全高于** `c0` 的两个（1.5149 / 1.5472）——
  最小值比最大值还高 0.08，两两不重叠；
- `mf` 与 base 的区间**重叠**（1.6277–1.6547 vs 1.5893–1.6340）⇒
  **接受率与精确长度那一路统计上无法区分，也就是全额拿回来了**；
- 吞吐上 `mf` 的两个实例也都高于 `c0` 的两个（185.17/193.02 vs 176.90/184.04）。

**六臂输出逐字节完全相同**（sha `df34ea140872348c`，和之前四臂 leg 同一个值）。

**对上了 §8.5.5 的天花板**：那次故意写错的天花板臂是 `c=8 ITL 50.0 ms`，
这里 `mf` 是 **50.02 ms** —— 不是接近，是踩在上面，而且**没有付接受率**。

#### 正向对照（这次调查已经栽过两次静默空转，所以判据先于数字）

真机日志（rank 0，`NT_MASKFIX_EVERY=2000`）：

```
[NT_MASKFIX] calls=22000 hit=22000 capturing=0 noncausal=0 nonuniform_q=0
             shape=0 sinks=0 not_marked=2300 verify_n=8 verify_max_rel=0.0
```

- `hit == calls`：**所有被标记的草稿 build 全部走了新路径**，四个回退分支一次没触发
  ⇒ 这个臂不可能是 `c0` 的伪装（那正是 §8.5.5.7 里读出"看起来合理的小改善"的方式）；
- `verify_max_rel=0.0`：`NT_MASKFIX_VERIFY=8` 让前 8 次草稿 build 额外用**精确长度**
  跑一遍原 TND 参考，四个 TP rank 全部 **`rel=0` 逐位相同**
  （`b=1 q=8 s2=8192 win=2048`）。窗口语义在真模型上确认——差一位这里会读到 ~0.15。

⚠️ 那 8 次 verify 都是启动期的单请求 build，`U == L`，所以它验的是**布局 + 窗口语义**，
**没有覆盖尾部裁剪**。尾部裁剪的直接证据是**接受率**：掩码要是没盖住尾巴，
`mf` 就会掉到 `c0` 那一档——它没有。

#### 要留意的

1. **每臂只有 2 个实例**。`mf > c0` 在接受率上两两不重叠（可信），
   吞吐上 +4.8% 也是两两不重叠，但 2v2 只够说"方向一致"，不够给显著性。
2. **接受率贴地板**：pooled 数是 probe + bench 混算的，random 数据集 `ratio≈0.07`、
   `accept_len≈1.6`（§9 第 0 条早就标了这个问题）。在接受率健康的数据集上差距应该更大。
3. §8.7.2 测到的"n=64 + 滑窗，通用模板贵 50.8%"**没有变成端到端的回归** ——
   c=8 时批量小。大并发要单独复测。
4. 合入形态还没做：现在是 monkeypatch + 锚点插入，要变成正式补丁得改
   `forward_fused_infer_attention` 本体，并给 `nonuniform_q` 那条回退写单测。

---

## 9. 遗留 / 后续（按价值排序）

0. **最便宜的一步：在一个接受率不贴地板的数据集上重跑三臂。** 见 §8.5.5：
   喂草稿一个偏长的 KV 长度**不会改变输出分布**（拒绝采样的数学保证），只掉接受率。
   如果接受率掉得少，那 +31% 吞吐今天就能拿，不用改算子也不用改框架。
   现有数据（random，接受长度 ~1.7）**对这个问题没有功效**，不能当证据。
1. **~~草稿 build 走 device 张量~~ —— 已做，已否**。见 §8.5：正确但负优化，
   算子绑定层逐请求回读长度（+35 µs/请求）。代码在
   `exp/16271-draft-seq-lens-device-tensor`，**不要合**。
   真正要做的是 §8.5.4 的两条路之一（算子侧去掉逐请求回读，或框架侧提前给出精确 host 长度）。
2. **MRV1 也有同样的问题，暂时没受益 —— 但审计完发现它在一种配置下本来就能开。**
   `model_runner_v1.py:3388` 把 `optimistic_seq_lens_cpu` 放进 `_seq_lens_cpu`，
   注释自己写了 "Always pass optimistic_seq_lens_cpu"，所以 `seq_lens_cpu_is_exact`
   保持 False、行为不变 —— 这是安全的默认。

   **但这个"乐观"值有一条会被修正成精确的路径**（本次纯代码审计，**未上机验证**）：

   ```
   model_runner_v1.py:1520   if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
                                 self._correct_optimistic_seq_lens_cpu(num_reqs)
   ```

   - `_needs_seq_lens_cpu_sync`（:499）对 **`AscendAttentionBackend`** 就是 True ——
     正是本 issue 这个后端；
   - `_correct_optimistic_seq_lens_cpu`（:1749）用上一步末尾异步 D2H 的
     `valid_sampled_token_count` 把 `num_computed_tokens` 的乐观假设修正回真值，
     它自己的 docstring 明说：拷贝是**整整一步之前**发起的，稳态下 event 早已 signaled，
     **这个 synchronize 实际是空操作**，不会把被去掉的那次同步加回来。

   ⇒ 在 **MRV1 + AscendAttentionBackend + async spec decode** 下，
   `optimistic_seq_lens_cpu` 对**目标 build** 就是精确的，可以按同样的条件置
   `seq_lens_cpu_is_exact=True`，MRV1 也能吃到那 −75.8%。
   条件不成立时（非 async）必须继续保持 False。

   **验证方法现成**：`serve.sh` 里 `MRV2=0` + `NT_SEQ_LENS_CHECK=1` 跑一条探针腿，
   看 `by_src` 的 mismatch 是不是 0。一条腿约 15 分钟。
3. `query_start_loc` 那次 `pin_memory().to(device, non_blocking=True)`（348 行的 TODO）
   **没有动**——pinned + 非阻塞，不产生同步，是另一件事。
4. 已有 PR **#15229 / #9095** 都需要按 §4 的结论修正后才能合。

---

## 10. 原始数据落点（机器上）

```
/data01/nt-work/logs/exp3_S_{base,fix,ceil}.txt    三臂的完整 leg 日志
/data01/nt-work/logs/bench_*.log                   每次 vllm bench serve 的原始输出
/data01/nt-work/logs/serve_*.log                   服务日志（含 [NT_PROBE] 计数行）
/data01/nt-work/prof/S_{base,fix,ceil}/            profiling 原始数据 + ASCEND_PROFILER_OUTPUT
/data01/nt-work/logs/analyse3.txt                  离线解析日志
```

---

## 10.5 第三轮换到 18 机跑（A2 的卡被别人占满）

A2 8/8 被别人的 `tokenizer-cache-test` 占着，等不到。排查了三台机之后换到
**18 机 `183.236.60.18`**（脚本在 `m18/`）。

### 为什么 18 机能替代

18 机的镜像是 0.26，本以为太老，但 `attention_v1.py` 与 main **逐行相同**：

```
321:  elif self.speculative_config and self.speculative_config.parallel_drafting:
322:      seq_lens = common_attn_metadata.seq_lens
333:  seq_lens_list = seq_lens.tolist()
```

而且 `dflash_proposer.py:186-187` 给草稿 build 喂的就是 `optimistic_seq_lens_cpu`
（`seq_lens_cpu` 和 `seq_lens_cpu_upper_bound` 都是它）。
⇒ **把 321 那个 elif 关掉，草稿就去读乐观上界 —— 这既是天花板探针，又逐字是 PR #9095 干的事。**
一个实验同时回答 §8.5.5 和 #9095 两个问题。

### 换掉的变量（证据强度按这个读）

| | A2 原计划 | 18 机实跑 |
|---|---|---|
| 投机方法 | DSpark | **DFlash**（同样强制 `parallel_drafting=True`，同一分支） |
| 模型 | Qwen3.6-35B-A3B | **Qwen3.5-35B-A3B** + 配套 DFlash 草稿 |
| 并行 | TP4+EP | **TP2+EP**（物理卡 1、3） |
| runner | MRV2 | **MRV2**（这个 DFlash 草稿强制要求，见下） |
| vllm | 0.28.0 | 0.26.0 |

**卡故意选 1 和 3** —— 这两张是 18 机已知 HBM 带宽只有 1/3 的坏卡
（[[m18-cards-1-3-hbm-third-bandwidth]]）。判据是接受率和输出文本，都与带宽无关，
这样把可能健康的 4 号卡留给别人。**这台机上的任何延迟数字都不作数。**

### 结构对照（防"两臂其实没区别"）

两臂都开探针。基线臂必须读到 `d2h>0`、天花板臂必须读到 `mismatch>0`；
读不到就说明两臂根本没差异，任何"无影响"的结论都是假的
（[[silent-noop-traps-and-ceiling-probe]]）。

### 踩到的坑

**坑 2：这个 DFlash 草稿强制要 MRV2。** MRV1 下直接拒绝构建：
`NotImplementedError: DFlash drafters with mixed sliding/full attention require the
V2 model runner; relaunch with VLLM_USE_V2_MODEL_RUNNER=1`
（`vllm/model_executor/models/qwen3_dflash.py:109`）。加上 `VLLM_USE_V2_MODEL_RUNNER=1` 即可，
反而把这台机拉回到跟 A2 同一个 runner，替换掉的变量少了一个。

> ⚠️ 但 MRV2 下**天花板臂读到的到底是什么值需要看探针**：走 `dflash_proposer` 那条是
> `optimistic_seq_lens_cpu`（偏长约 `num_rejected`，正是要测的）；走 MRV2 的
> `build_attn_metadata` 则是 `seq_lens_np is None` 时填的 `np.full(num_reqs, max_seq_len)`
> **占位符**（离谱错值）。探针打出的 `delta` 能直接区分：≈`num_speculative_tokens` 是上界，
> 巨大就是占位符。后者恰好是我断言 #15229/#9095 在 MRV2 下会喂错的那个点。

**坑 1：**容器用 `--device=/dev/davinci{1,3}` 白名单挑卡后，**运行时把它们重编号成 0 和 1**，
`ASCEND_RT_VISIBLE_DEVICES` 写物理号 `1,3` 会静默只剩一张卡，起服死在
`worker.py:438 AssertionError: DP adjusted local rank 1 is out of bounds for 1 devices`
（报错完全不提这个变量）。正确写法 `0,1`，钉卡靠白名单。
见 [[ascend-container-device-renumbering]]。

---

## 11. 待发出去的东西（已写好，卡在权限）

`github/` 目录下四份，字节即最终文案：

| 文件 | 去处 | 要点 |
|---|---|---|
| `issue-16271-comment.md` | issue #16271 | 根因量化 + 两条分支 + **算子基准表**（劝阻 @modelpath-dev 照原方向做）+ 天花板 + 内嵌可跑的 `fia_timing` 脚本 |
| `pr-15229-comment.md` | PR #15229 | 草稿那一路取 host 镜像是错的；点出其单测 `seq_lens_cpu = seq_lens.clone()` 是自证式、抓不到；给 10 行探针 |
| `pr-9095-comment.md` | PR #9095 | 同上，外加"那个 elif 不是冗余的" |
| `pr-body.md` | 新 PR body（`fix/…` → `vllm-project:main`） | 按仓库模板三段式 |

**两个阻塞项：**

1. **权限**：Windows 凭据里的 token 是 fine-grained PAT，只能写自己的 fork。
   `POST /repos/vllm-project/vllm-ascend/issues/16271/comments` → **403
   `Resource not accessible by personal access token`**；`GET /repos/vllm-project/vllm-ascend`
   的 `permissions.push = False`。要发需要 **classic PAT + `public_repo` scope**
   （或装 `gh` 走浏览器 OAuth —— 这台机器上没装 `gh`）。
2. **DCO**：两个 commit 的 author 是 `wzy85 <wzy85@local>` 且没有 `Signed-off-by:`。
   vllm-ascend 要求 DCO（`docs/source/developer_guide/contribution/index.md:77`）。
   开 PR 前必须 `git commit --amend --author=... -s` 重写并 force-push（blob 不变，只换 commit SHA）。
   仓库里常见写法是 noreply：如 `zhao-stack <80399320+zhao-stack@users.noreply.github.com>`。
