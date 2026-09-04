# 03 —— 代码级结论

证据强度：**[F]** 读到代码/产物并引用；**[I]** 由读到的代码推断；**[U]** 未验证。

其中「prefix cache」一节的 16 条结论经过 3 视角对抗验证（12 条 verdict 全部
`refuted=false / confidence=high`）。其余各节是单遍代码阅读 + 实测，**未经对抗验证**。

---

## 量化产物加载

| # | 结论 | 强度 | 出处 |
|---|---|---|---|
| 1 | 量化方法名是 `"ascend"` | [F] | `vllm_ascend/utils.py:49` |
| 2 | 自动探测不会触发，必须显式 `--quantization ascend`：`override_quantization_method` 要求 `hf_quant_cfg is not None`，而产物 config.json 无 `quantization_config` | [F] | `quantization/modelslim_config.py:520-525` + 产物实测 |
| 3 | `get_config_filenames()` 返回 `[]`，配置在 `maybe_update_config()` 里读 `quant_model_description.json` | [F] | `modelslim_config.py:507-513` |
| 4 | `apply_vllm_mapper()` 把模型的 HF→vLLM WeightsMapper 应用到 quant_description 的 key，所以张量改名在量化侧也被处理 | [F] | `modelslim_config.py:526+` |
| 5 | `glm5_next` 有 `model.layers.` → `language_model.model.layers.` 的兜底（多模态包装层） | [F] | `modelslim_config.py:605-614` |
| 6 | `glm5_next` / `glm5_next_text` / `glm5_next_mtp` 都有 `".mtp_block." → "."` 的 substr 映射 | [F] | `modelslim_config.py:339-347` |
| 7 | HF 命名 → vLLM 命名的 10 条映射，使 msmodelslim 产物与官方 release 两种命名都能加载 | [F] | `models/glm5_next.py:109-121` |
| 8 | 专家是逐专家非打包布局 | [F] | `models/glm5_next.py:123-132` |

## Prefix caching（经对抗验证）

| # | 结论 | 强度 | 出处 |
|---|---|---|---|
| 9 | 默认关：`is_prefix_caching_supported` 对 `attn_type=="hybrid"` 返回 False（模型实现了 `IsHybrid`） | [F] | `config/model.py:1852-1857`, `model.py:1624-1632`, `glm5_next.py:2437` |
| 10 | 不开 APC 时 `mamba_cache_mode` 被强制 `"none"`，整套 align 机制不加载 | [F] | `models/config.py:388-389` |
| 11 | 逻辑 block size 由 KDA 状态大小反算：TP8=640 / TP4=1152 / TP2=2176；`--block-size` 只能调大 | [F] | `patch_mamba_config.py:131-139,153-159` + **实测命中落在 640 / 2176 的整数倍上** |
| 12 | KV init 之后 `cache_config.block_size` 被改写成 4 | [F] | `v1/engine/core.py:283-285` |
| 13 | `patch_glm5_next_mamba_scheduler.py` 就是为扛 #12 而存在；已核实它在 `patch/platform/__init__.py:45` 被导入，且在 `patch_mamba_config`(:28) 之后（顺序不能反，它 import 后者的 `_is_glm5_next_model`） | [F] | 两处 `__init__.py` 行号 |
| 14 | `hash_block_size = gcd(640,4) = 4` | [F] | `patch_kv_cache_utils.py:373-395` |
| 15 | indexer 的 kpool 状态是**位置函数**（只在绝对位置 ≡ -1 mod kpool 写条目、按位置寻址）⇒ 内容可寻址、可缓存；真正顺序的只有 KDA 递归状态 | [F] | `core/kv_cache_interface.py:43`, `ops/glm5_next_kpool_state_compress.py:159,167` |
| 16 | `models/layer/attention/layer.py:117-120` 那处 `enable_prefix_caching=False` **对本模型是死代码** | [F] | `layer.py:112-120`, `attention/dsa_v1.py:195`, `ops/indexer_kpool_mla.py:121` |
| 17 | `platform.py:321` 的 `update_block_size_for_backend` 在 APC 开启时是 no-op | [F] | `platform.py:312` |
| 18 | 开 APC 后 mamba block table 从每请求 1 条变成 `cdiv(max_model_len, block)` 条 | [F] | `model_runner_v1.py:4991-4998` |
| 19 | align 模式的状态搬运依赖 Triton `batch_memcpy_kernel`，非 310P 无 torch 回退 | [F] | `patch/worker/patch_mamba_utils.py:178-193` |
| 20 | ↑ 该 kernel 在 910B4 上**能用** —— APC 实测命中即证明 align 模式跑通 | [F] | 实测 |

## 图模式

| # | 结论 | 强度 | 出处 |
|---|---|---|---|
| 21 | 本镜像**无 torchair**，旋钮是 `xlite_graph_config {enabled, full_mode}` | [F] | `ascend_config.py:37,667-668`；全文无 `torchair_graph_config` |
| 22 | 开 xlite 会强制 `block_size <= 128`，**与 APC 要的 640 冲突** ⇒ xlite 与 prefix cache 互斥 | [I] | `utils.py:1439-1443`（未实测） |
| 23 | `@support_torch_compile` 只在 `AscendGlm5NextModel`；MTP 类和 ViT 都没有 ⇒ drafter/ViT 走 eager | [F] | `glm5_next.py:2244`, `glm5_next_mtp.py:186`, `glm5_next_multimodal.py` |
| 24 | 进程启动即 `VLLM_USE_BREAKABLE_CUDAGRAPH = False` | [F] | `platform.py:60-63` |
| 25 | PIECEWISE 时 `splitting_ops` 追加 `vllm::mla_forward` / `vllm::dsa_forward` | [F] | `platform.py:604-618` |
| 26 | `ASCEND_LAUNCH_BLOCKING=1` 与 ACL graph 互斥，raise | [F] | `platform.py:641-647` |
| 27 | 捕获尺寸含 `decode_query_len = 1 + num_speculative_tokens` | [F] | `platform.py:236-239` |
| 28 | 默认 PIECEWISE 与 `FULL_DECODE_ONLY` 都可用 | [F] | 实测 |

## MTP

| # | 结论 | 强度 | 出处 |
|---|---|---|---|
| 29 | `glm5_next`/`glm5_next_text` → `glm5_next_mtp` → `Glm5NextMTPModel`，并扩了 `MTPModelTypes` | [F] | `patch/platform/patch_speculative_config.py` |
| 30 | `Glm5NextMTPModel` / `Glm5NextMTP` → `glm5_next_mtp:AscendGlm5NextMTP` | [F] | `models/__init__.py:19-24` |
| 31 | `method="mtp"` 且不给 `model` 时 draft 取 target 路径，量化配置从 target 继承 | [F] | `src_vllm/config/speculative.py:556-568` |
| 32 | MTP 层索引 = `config.num_hidden_layers`（=45）；`rot.weight` 也被路由到该层 | [F] | `glm5_next.py:104,135-146` |
| 33 | `num_speculative_tokens > 1` 会在同一个 MTP 层上多次前向，上游有 warning | [F] | `speculative.py:719-724` |
| 34 | MTP 开启后所有 KV group 被标成 eagle group，每次 APC 命中砍掉一整块 | [F] | `kv_cache_coordinator.py:103` + **实测：命中从 2 block 掉到 1 block** |
| 35 | `get_mamba_state_shape_from_config` 与逐层 `get_state_shape` 在 MTP 下对 conv state 长度差 6144 B | [I] | `ops/kimi_kda_state.py:29` vs `glm5_next.py:1698` —— 对 APC 无害，MTP 正确性待查 |
| 36 | 真权重实测接受率 **73.0%**（`num_speculative_tokens=1`） | [F] | 实测 |

## 运行时/环境

| # | 结论 | 强度 | 出处 |
|---|---|---|---|
| 37 | 容器 `ulimit -l` 默认 64 KB，`--privileged` 不覆盖；这是所有图捕获失败的唯一根因 | [F] | 实测 + 单变量 A/B |
| 38 | 驱动在 pinned 分配失败时会同时打印 EE1016「stream is capturing」，**是误导** | [F] | 单变量 A/B：还原补丁只留 memlock，PIECEWISE 照样通过 |
| 39 | `xxhash` 镜像里没有，`--prefix-caching-hash-algo xxhash` 不装就 500 | [F] | 实测 |
| 40 | `cd /vllm-workspace` 会让仓库目录遮蔽已安装的 vllm 包 | [F] | 实测 |
| 41 | `pkill -f "vllm serve"` 之后 `VLLM::Worker_TP*` 被 reparent 继续占 HBM，要多杀一轮 | [F] | 实测 |
| 42 | msmodelslim 峰值每卡 ~20 GB HBM（linear_quant DTS 阶段），卡上有别的负载会 OOM | [F] | 实测（被并发的 vLLM 挤掉过一次） |
| 43 | KDA prefill 的 `recurrent_state[state_indices]` gather 开销随状态缓存规模走，`gpu-memory-utilization` 吃太满会自己撑死自己（迷你模型 TP2 上 0.85 时单次 6-token prompt 就要 20.13 GiB，0.35 通过） | [F] | `ops/kimi_kda.py:427` + 实测 |
| 44 | 镜像里 pip index-url 指向不可达的 k8s 内网 cache | [F] | 实测 |
