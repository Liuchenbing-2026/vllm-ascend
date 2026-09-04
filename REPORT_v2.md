# vllm-ascend LoRA Triton 算子性能回归：根因分析与优化报告

- 分支：https://github.com/CCH-gif/vllm-ascend/tree/lora-triton-vllm-serve（commit 0d56b76）
- 机器：m18（Ascend 910B4，40 AIV 核 / 192KB UB，单卡）；Qwen3.6-27B + r16 LoRA（openscad）
- 日期：2026-09-04；全部数字为本机实测

## 结论（TL;DR）

| gsm8k e2e（chat、K=1 顺序、64 题、512 token、temp 0） | acc | 吞吐 | 总耗时 | 输出 |
|---|---|---|---|---|
| AscendC（stock 基线） | 79.69% | 12.83 tok/s | 2536 s | — |
| 分支原样（triton_v1） | 79.69% | 10.17 tok/s（**−20.7%**） | 3201 s | 与 AscendC **64/64 逐字节一致** |
| 本补丁（triton_v2） | 79.69% | **14.39 tok/s（+12.2%）** | **2262 s（−10.8%）** | 与 AscendC **64/64 逐字节一致** |

- v2 相对分支原样提速 **41.5%**，相对 AscendC 基线反超 **12.2%**；
- 精度无损的判据取到最强：三臂 64 道题的输出**逐字节相同**（acc 自然相同），
  背后是算子级 **~200 组配置逐位（bit-exact）等于 AscendC**（torch.equal，非容差）。

## 一、性能下降的根因（按影响排序，全部实测确认）

### 1. C++ launcher 把完整逻辑网格当 blockDim 传给 rtKernelLaunch（TTFT 主凶）
`lora_ops_triton.py::_cpp_launch` → `lora_cpp_launcher.cpp`：blockDim=逻辑 grid。
triton-ascend 自带 launcher（backends/ascend/driver.py，enable_auto_map_parallel_blocks）
会钳位到物理核数 40；编译出的核用 arg buffer 里的 gridX 做 grid-stride 循环，
所以钳位后逐位不变。同一二进制、同一形状，只改这一个数：

| 算子/形状 | 原样 | 钳位 40 | 加速 |
|---|---|---|---|
| sgmv_shrink H=17408 B=1024 | 20147 µs | 1567 µs | 12.9× |
| sgmv_expand Ho=17408 B=1024 | 33829 µs | 2585 µs | 13.1× |
| sgmv_expand Ho=17408 B=256 | 2486 µs | 700 µs | 3.6× |
| B≤8（decode） | 不变 | — | 1.0× |

serve 级证据：v1 在 893-token prefill 的 TTFT = **10.11 s**（AscendC 1.42 s，v2 2.82 s）。

### 2. no-lora 行不早退：base 请求也整步付 LoRA 代价（decode 主凶之一）
AscendC 核对 idx<0 的行是 `continue`（近零代价）；v1 的 triton 核**全量计算完再用
keep 掩码丢弃**。decode 图（aclgraph）回放时 indices 只是运行时张量内容，于是
**不带 adapter 的 base 请求每步也执行全部 LoRA 计算**：

| decode TPOT（K=1） | base | lora |
|---|---|---|
| AscendC | 62.19 ms | 75.12 ms（LoRA +12.9 ms = 17.2%/步） |
| triton_v1 | **96.19 ms** | 96.23 ms（base≡lora，+34 ms ≈ v1 算子 B=1 全量成本 ✓） |
| triton_v2 | 67.58 ms | 67.52 ms（同机制，v2 算子便宜所以只 +5.4 ms） |

即分支原样连 **base 流量都劣化 55%**。（v2 对此为已知限界，见附录；纯 adapter
流量不受影响。）

### 3. 核结构低效
- shrink 用 64 元素内块（为对齐 AscendC 求和序），把连续的 LoRA-A 行打碎成
  128B 跨步突发（8× 事务）；
- expand 网格 (B,)：decode 只用 4~8/40 个核；权重按 [BLOCK_HO,R] 步长 R 寻址
  → 每行 32B 突发；
- `_to_int32` 每调用多发 2 个 cast 核，被 aclgraph 烤进每个 decode step；
  eager prefill 的 host 税 ~125 µs/调用。

### 4. 死代码与潜藏故障
- vLLM V1 非 CUDA 平台硬写 `LoRAMapping(is_prefill=True)`
  （vllm/v1/worker/lora_model_runner_mixin.py:57）⇒ serve 只走 sgmv_*，bgmv_* 全死代码；
- `_cpp_case_key` 对 float 取 `.dtype` ⇒ bgmv_shrink 的 C++ 路径必抛 AttributeError（被死代码掩盖）；
- `TRITON_LORA_CPP=0` 回退路径 `m3` 未绑定（NameError）；一处重复计时；
  测量配置里 TRITON_LORA_TIME=1 未关。

### 5.（开发中发现的陷阱，已在 v2 根除）运行时标量的 Triton 特化
expand 核原带运行时 int 参数 B；Triton 对编译时恰为 1 的 int 会特化成常量并从
签名删除，而 C++ launcher 仍按原布局打包 ⇒ gridX 槽位错 4 字节 ⇒ 程序 0 被重复
执行 ~20 次、其余 chunk 不执行（结果错 + B=64 时 36× 设备时间）。v2 里 B 改由
token_nums 求和推导，核不再有任何运行时标量。

## 二、逐位无损是怎么做到的

判据 `torch.equal`；全扫 = B∈{1..1024} 十档 × NR∈{1,2,3} × 10 个 serve 形状 ×
两种数值幅度，**0 失配**；另有 idx=-1 段 + 种入 −0.0 的专项，同样逐位。

- **shrink**：AscendC 顺序 = 64 元素组内树形规约、组间按序累加、以 11776
  （TILE_LENGTH）为窗口。EXACT=1 用 tl.split 剥出组部分和 + 显式加法链复刻
  （掩码逐列的朴素写法代价 2.5×，tl.split 版只有 ~1.1×）。
- **expand**：AscendC 用 BlockReduceSum(8)+PairReduceSum。采集 300 个 bf16 舍入
  平局样本在 host 枚举 14 种结合序：**相邻配对二叉树 300/300 全中**（其余 ≤171/300）。
  实现关键：对**连续内轴**的 `tl.sum` 恰好下发同一硬件规约 ⇒ 天然逐位；
  转置外轴规约快 ~30% 但 ~1/50k 平局元素差 1 ulp，弃用。
- 后端陷阱实录（都会破坏逐位或直接崩）：
  1) 多级 size-2 轴 `tl.sum` 会被 bishengir 重新结合，≠ 相邻树；
  2) 剥列布局传播进 load 的 UB alloc → "cannot align 1 axis" 编译失败；
  3) 组合条件 `(row_ok & lid>=0)` 作 DMA mask → 设备 aivec 崩溃（标量 UB 越界），
     须改用 `tl.where` 值选择 + 位保持写回（bf16→fp32→bf16 往返恒等，−0 保号）。

## 三、v2 的结构性改动与算子级性能

改动：launcher blockDim 钳位 40；shrink BLK 64→512（8× 事务合并）+ B≤2 走
(B×R,) 网格（同求和序、16 核并行）；expand 网格 (⌈B/TB⌉×NCHUNK,)（Ho 摊到核上）
+ TB 按 B 分桶 1/2/4/8（TB 行共享一次权重装载）+ 权重一次连续装载后 reshape；
case 键修复、空 `_to_int32`、`m3`/计时修复；EXACT 默认开启（TRITON_LORA_EXACT=0 可关）。

每层（7 shrink + 7 expand）设备时间，NPUGraph replay，经 C++ launcher（serve 真实路径）：

| B | AscendC | v2（逐位配置） | 比值 |
|---|---|---|---|
| 1 | 168.9 µs | 106.3 µs | **0.63×** |
| 4 | 174.2 µs | 194.7 µs | 1.12× |
| 8 | 176.6 µs | 237.7 µs | 1.35× |
| 64 | 359.6 µs | 823.1 µs | 2.29× |
| 256 | 1064.2 µs | 2968.8 µs | 2.79× |
| 1024 | 3705.1 µs | 11445.9 µs | 3.09× |

B=1（K=1 评测的工作点）shrink 0.55×、expand 0.77×。prefill 大 B 慢于 AscendC
是逐位规约的已知代价（对比：v1 原样在 B=1024 单个 expand 就要 33.8 ms）。

## 四、serve 级三臂 A/B 全景

decode（探针，base 与 adapter 同 serve 对照）：

| lora TPOT | K=1 | K=2 | K=4 |
|---|---|---|---|
| AscendC | 75.12 | 76.54 | 79.07 |
| triton_v1 | 96.23（+28%） | 97.52 | 100.19 |
| triton_v2 | **67.52（−10.1%）** | **71.99（−5.9%）** | 81.92（+3.6%） |

TTFT（median，max_tokens=1）：

| prompt tokens | AscendC | triton_v1 | triton_v2 |
|---|---|---|---|
| ~125 | 1370 ms | 1131 ms | **1049 ms** |
| ~437 | 1476 ms | 2682 ms | 1414 ms |
| ~893 | 1416 ms | 10108 ms | 2817 ms |

- 本 stack 的 eager prefill 强 host 主导（AscendC 三档几乎平：~1.4s 平台底）。
  AscendC binding 在 SetCustomHandler 闭包里每次调用查 aclGetDeviceCapability，
  ~900 次/步的 host 税是其平台底偏高的合理解释（代码可见，未单独剖析计时）；
- v2 在 gsm8k 的 prompt 规模（~125 token）TTFT 反而最低（gsm8k 内实测
  ttft_med：v2 1014 ms vs AscendC 1448 ms）；~900 token 才显出逐位规约的
  prefill 代价（+1.4 s）；v1 则是 10 秒级灾难。
- gsm8k 三臂 completion tokens 完全相同（32541），finish 分布相同——逐字节
  一致的自然推论。

## 五、交付物与复现

- `apply_v2_patch.py`：对分支两个 py 文件做**可审计的精确字符串替换**，
  用法 `python3 apply_v2_patch.py <分支目录> <输出目录>`；
- m18: `/data2/lora-triton-work/`：`v2/`（最终树）、`serve_only.sh <arm>`（一键三臂切换）、
  `run_suite.sh`（三臂全套：share 探针 + TTFT 探针 + gsm8k + 跨臂逐字节对比）、
  `eval_gsm8k.py` / `cmp_evals.py` / `bench_exact2.py`（逐位全扫 + 计时）；
- 评测产物：`/data2/lora-triton-work/evals/*.json`（本报告同目录 evals/ 有副本）。
- 环境开关：`TRITON_LORA_V2=0` 回退 v1 行为；`TRITON_LORA_EXACT=0` 关逐位
  （shrink 提速 ~35%@prefill，偏差 ~3e-7 相对值）；`TRITON_LORA_CPP=0` 走
  eager triton launcher。

## 附录：已知限界（均有守卫或已文档化）

1. prefill 大 B 的 expand 慢于 AscendC 2.3~3.1×（换取逐位）；gsm8k 场景占比 <1%，
   长 prompt 高并发场景建议后续做 AscendC 式 token-per-core 的逐位 prefill 核。
2. base 请求与 adapter 同 serve 时，decode 图内 v2 算子对 idx=-1 行仍计算后丢弃
   （+5.4 ms/步 vs AscendC 的行内早退）；纯 adapter 流量无影响；后续可加标量早退。
3. R≠16 的 expand 走快速树（~1 ulp，非逐位）。对抗审计发现的三处正确性缺陷
   已参考上游 AscendC 的任意形状支持，在 v2 内核内用掩码原生修复，不再回退慢路径：
   (a) shrink 尾块加 oh<H 掩码（H%64≠0，如 Llama down_proj H=1376）；
   (b) expand 加 ho<Ho 掩码 + ceil 分块（slice 宽 %bh≠0，如 head_dim=80）；
   (c) idx<0（no-lora）行按 AscendC `continue` 语义跳过写回（−0.0 保号）。
   实测：标准全矩阵 0 失配；边缘 {1008,2000,1376,112,80} 均逐位并吃到 v2 加速；
   仅 <16 对齐形状（如 1000）不支持——AscendC 自身同样不支持（32B DMA 对齐）。
4. bgmv_* 在 vLLM V1 下是死代码，但 `_cpp_case_key` 崩溃已修，防止未来启用时踩雷。
