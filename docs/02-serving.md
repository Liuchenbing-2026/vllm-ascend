# 02 —— 服务接线：prefix cache + 图模式 + MTP

## 结论

三个特性可以**同时开**，实测全部通过，**不需要改 vllm-ascend 的代码**。

唯一的硬性前提：容器要用 `--ulimit memlock=-1` 启动。

## 验证方法：先用迷你模型，再上真权重

真权重 311 G、加载一次十几分钟、要独占 8 卡，用来试错太贵。
所以先造了一个**结构完整但只有 10.7 GiB** 的迷你模型（`test/make_tiny_model.py`）在 2 卡上跑通路：

```
8 层 = 6 层 KDA (0,1,2,4,5,6) + 2 层 DSA (3,7)   ← 保留 every-4th 的 cadence
+ 第 8 层 = MTP 层（eh_proj/enorm/hnorm/shared_head/MLA/indexer/MoE 齐全）
mlp: 0/1/2 dense，3-8 MoE(16 routed + 1 shared, top-8)
ViT 4 blocks；vocab 154880 不动（tokenizer 必须可用）
```

关键：它是**读真 checkpoint 的 safetensors header 反解出张量名/形状/dtype** 造的，
所以布局与真权重逐字节对齐（逐专家 MoE、拆开的 q/k/v_conv1d、真的 MTP 层）。
vllm-ascend 自带的 `tools/generate_glm5_next_safetensors.py` 走 HF 模块树，
产出的是打包专家 + 合并 conv1d 且**没有 MTP 层**，不能用于这个目的。

## 迷你模型阶梯（TP2，卡 0/1）

| # | 配置 | 结果 |
|---|---|---|
| S1 | eager + 无 APC | ✓ 出 token |
| S2 | + 图模式（默认 PIECEWISE） | ✗ → 见下「memlock」 |
| S2b | + 图模式 `FULL_DECODE_ONLY` | ✓ 捕获 18 s / 0.30 GiB |
| S3 | + prefix cache | ✗ → 同一个 memlock 根因 |
| S3e | 修 memlock 后 | ✓ **96.4× 加速**，hits=4352=**2×2176** |
| S4 | + MTP | ✓ `Glm5NextMTPModel` 加载，125 drafts；APC 命中降到 2176=**1 block** |
| S5 | 默认 PIECEWISE，修 memlock 后 | ✓ 捕获 9 s |

两处**精确命中代码分析预测**的地方：

- **S3e**：`hits = 4352 = 2 × 2176`。2176 正是 TP2 下由 KDA 状态大小反算的逻辑 block size
- **S4**：开 MTP 后命中从 2 个 block 掉到 1 个 block ——
  `kv_cache_coordinator.py:103` 把所有 KV group 标成 eagle group，每次命中砍掉一整块

## 真权重（TP8 + EP）

```
Available KV cache memory: 14.11 GiB
GPU KV cache size: 337,547 tokens
Graph capturing finished in 58 secs, took 0.42 GiB
每卡 59 GB / 65536 MB
```

| 项 | 数值 |
|---|---|
| 正确性 | `17 * 23` → **391**；中文问答正确 |
| Prefix cache | 5001-token prompt：冷 1.89 s → 热 0.83 s（2.3×）；**hits = 3840 = 6.00 × 640** |
| MTP | drafts 115 / accepted 84 → 接受率 **73.0%** |
| 解码 | eager ≈ 1.01 s/token → 图模式+MTP **40.5 ms/token** |

`640` = TP8 下的逻辑 block size，与 TP2 的 2176 一样由 KDA 状态大小反算
（`patch_mamba_config.py:131-139`）。命中落在整数倍上，一个 token 不差。

> GLM-5.3-Flash 是 thinking 模型，`/v1/chat/completions` 的 `content` 里
> `</think>` 之后才是答案。做正确性判据时别只看开头。

## 根因：容器的 memlock 是 64 KB

所有图捕获失败的**唯一**根因。

```
$ docker exec <容器> bash -lc 'ulimit -l'
64                     # KB
$ ulimit -l            # 宿主机
131806324              # KB ≈ 126 GB
```

`--privileged` **不**覆盖 memlock rlimit，必须显式 `--ulimit memlock=-1`。

失败长这样：

```
allocate_host_memory_slowpath:.../CachingHostAllocator.cpp:244
  NPU function error: aclrtMallocHostWithCfg, error code is 207001
[Error]: Failed to apply for memory.
  Not_Supported(EE1016): ... The current thread is in the capture state and the current
  operation cannot be performed ... This operation is supported only in the RELAXED mode.
  rtMemcpy execution failed, reason=operation not permitted when a stream is capturing...
  rtsMallocHost execution failed, reason=driver error:out of memory
```

**那条 EE1016「stream is capturing」是误导**。分配失败在 rlimit 上，
驱动把捕获态一并报出来而已。我一开始按字面读，去改了 vllm-ascend 里两处
`.clone().pin_memory()`（`worker/block_table.py:284` 和 `worker/utils.py:15`），
**做完单变量 A/B 证明补丁不需要** —— 还原补丁、只留 memlock，默认 PIECEWISE 照样 9 秒捕获通过。
补丁的记录留在 `patches/README.md`，**不要用**。

**为什么只有开了 prefix cache 才暴露**：APC 关闭时 mamba group 的 block table 每请求只有 1 条
（`model_runner_v1.py:4991`），开了之后变成 `cdiv(max_model_len, block_size)` 条，
缓冲区够大才会走到 caching host allocator 的 slowpath 去新分配 pinned 内存。

## 另外三个必须知道的

### `--quantization ascend` 必须显式传

`ASCEND_QUANTIZATION_METHOD = "ascend"`（`utils.py:49`）。
`AscendModelSlimConfig.override_quantization_method` 只在 `hf_quant_cfg is not None`
且其中没有 `quant_method` 时返回 `"ascend"`；而产物的 `config.json` **根本没有
`quantization_config` 字段**，所以 `hf_quant_cfg` 是 `None`，自动探测不触发。

`get_config_filenames()` 返回 `[]`（跳过 vLLM 的文件查找），
真正的配置在 `maybe_update_config()` 里读 `quant_model_description.json`。

### `xxhash` 镜像里没有

`--prefix-caching-hash-algo xxhash` 不装包就是 500：
`ModuleNotFoundError: xxhash is required for the 'xxhash' prefix caching hash algorithms`。

为什么推荐它：`hash_block_size = gcd(640, 4) = 4`（`patch_kv_cache_utils.py:373-395`），
hash 粒度比 block 细 160 倍，32K prompt 要在调度器关键路径上算 ~8192 次（默认 sha256）。
**不要**传 `--hash-block-size 640`，会因 `4 % 640 != 0` 直接 raise。

### 不要 `cd /vllm-workspace`

那里的 `vllm` 仓库目录会遮蔽已安装的包：

```
ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
```

### 停服务要多杀一轮 `VLLM::`

`pkill -f "vllm serve"` 之后 `VLLM::Worker_TP*` 会被 reparent 继续占着 HBM，
下一次启动就会报
`Free memory on device (21.22/60.96 GiB) is less than desired GPU memory utilization`。
判据以 `npu-smi info` 的进程表为准，不是 `ps`。

## 完整的约束清单（prefix cache）

| 约束 | 出处 |
|---|---|
| 必须 `--enable-prefix-caching`（hybrid 模型默认关） | `config/model.py:1852` |
| chunked prefill 必须开（默认开） | `models/config.py:371` 断言 |
| **禁** `--disable-hybrid-kv-cache-manager` | `patch_kv_cache_utils.py:528` raise |
| **禁** `--disable-chunked-mm-input` | `config/vllm.py:2120` 断言 |
| **禁** `VLLM_USE_V2_MODEL_RUNNER=1` | `config/vllm.py:2125` 断言 |
| **禁** `--mamba-cache-mode`（自动推成 align） | `patch_mamba_config.py:222` |
| `--max-num-batched-tokens >= 640`（建议 ≥2048） | `config/vllm.py:2112` |
| `--long-prefill-token-threshold`（若设）≥ 640 | `config/vllm.py:2118` |
| `--max-model-len` 要设小（别用默认 1M） | `model_runner_v1.py:4991` |
| `--block-size` 别传（只能调大，会被改写成 640） | `patch_mamba_config.py:153` |

## 图模式

- 这个镜像里**没有 torchair**，旋钮叫 `xlite_graph_config`（`ascend_config.py`）。
  但开 xlite 会强制 `block_size <= 128`（`utils.py:1439`），**与 APC 要的 640 冲突** ⇒ 走 aclgraph
- `@support_torch_compile` 只挂在 `AscendGlm5NextModel`（语言主干，`glm5_next.py:2244`）；
  `AscendGlm5NextMTP` 和视觉塔都没有 ⇒ **drafter 和 ViT 走 eager**
- 进程启动即 `VLLM_USE_BREAKABLE_CUDAGRAPH = False`（`platform.py:60-63`）
- `ASCEND_LAUNCH_BLOCKING=1` 与 ACL graph 互斥，会 raise（`platform.py:641`）
- 默认（PIECEWISE + FULL decode）与 `FULL_DECODE_ONLY` 都实测可用

## MTP

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

- `patch_speculative_config.py`：`model_type in ("glm5_next","glm5_next_text")` → `glm5_next_mtp`
  → `architectures: ["Glm5NextMTPModel"]`，并把 `"glm5_next_mtp"` 加进 `MTPModelTypes`
- `models/__init__.py`：`Glm5NextMTPModel` / `Glm5NextMTP` → `glm5_next_mtp:AscendGlm5NextMTP`
- 不给 `model` 时 draft 自动取 target 路径，**量化配置从 target 继承**（`speculative.py:556-568`）
- `num_nextn_predict_layers = 1`，**建议 `num_speculative_tokens` 就用 1**；
  >1 会在同一个 MTP 层上多次前向，接受率会掉（`speculative.py:719-724` 有 warning）
- MTP × APC 的代价：所有 KV group 被标成 eagle group（`kv_cache_coordinator.py:103`），
  每次命中砍掉一整个 block ⇒ **prompt < 2×block 时 APC+MTP 基本零命中**（实测复现）

## 一个被证伪的风险

代码分析阶段标了个 `[U]`：align 模式每步要跑 Triton 的 `ops/triton/batch_memcpy.py`，
**非 310P 上没有 torch 回退路径**，担心 triton-ascend 在 910B4 上编不出来。

**已证伪**：APC 在迷你模型和真权重上都真命中了，说明 align 模式跑起来了，这两个 kernel 是好的。
