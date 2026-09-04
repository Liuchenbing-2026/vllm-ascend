# GLM-5.3-Flash w8a8 —— 量化产物与 vLLM-Ascend 服务接线

在 Atlas A2（8×910B4-1，64 GB HBM/卡）上把 **ZhipuAI/GLM-5.3-Flash** 量化成 w8a8，
并在 vllm-ascend 上把 **prefix caching + ACL graph + MTP 投机解码**三个特性同时拉起来。

**结论：三个特性可以同时开，全部实测通过。** 不需要改 vllm-ascend 的代码，
唯一的硬性前提是容器要用 `--ulimit memlock=-1` 启动。

## 实测数据

| 项 | 数值 |
|---|---|
| 服务 | TP8 + EP，KV cache 14.11 GiB / 337,547 tokens，每卡 59 GB |
| 图捕获 | 58 s，0.42 GiB（PIECEWISE + FULL decode 都捕获） |
| Prefix cache | 5001-token prompt：冷 1.89 s → 热 0.83 s；**hits = 3840 = 6.00 × 640** |
| MTP | 接受率 **73.0%**（84 / 115 draft tokens，`num_speculative_tokens=1`） |
| 输出正确性 | `17 * 23` → **391**；中文问答正确（thinking 模型，答案在 `</think>` 之后） |
| 解码速度 | eager ≈ 1.01 s/token → 图模式+MTP **40.5 ms/token** |

`hits = 3840` 正好是 **6 × 640**，一个 token 不差 —— 640 是 TP8 下由 KDA 状态大小反算出的
逻辑 block size，见 `docs/02-serving.md`。

## 环境

```
镜像   quay.io/atlas-ci/vllm-atlas-temp:glm-5.3-flash-0902-1-910b-openeuler-33646519920-1-arm64-temp
       Python 3.12.13 / torch 2.10.0 / torch_npu 2.10.0.post2 / CANN 9.1.0
       transformers 5.16.0.dev0 (editable) / vllm 0.23.0 / vllm_ascend db701c1f
量化   msmodelslim, gitcode.com/qq_46439621/msmodelslim @ glm5_next_quant_0829 (b57416c)
```

## 快速开始

```bash
# 1. 起容器（--ulimit memlock=-1 是硬性的，见脚本里的注释）
bash serve/docker-run.sh

# 2. 量化（可选，若已有产物则跳过；~72 分钟 / 8 卡）
docker exec glm53q bash -lc "bash quantize/run_quant.sh"

# 3. 核对产物
docker exec glm53s bash -lc "python3 quantize/inspect_artifact.py /data02/GLM-5.3-Flash-w8a8-b0829"

# 4. 起服务（三个特性全开）
docker exec glm53s bash -lc "bash serve/serve.sh"

# 5. 冒烟 + 特性验证
docker exec glm53s bash -lc "cd /root && python3 test/smoke_test.py"
```

## 三个必须知道的坑

1. **容器 `ulimit -l` 默认 64 KB，`--privileged` 不覆盖它。**
   缺了 `--ulimit memlock=-1`，图捕获会挂在 `aclrtMallocHostWithCfg` 上，
   而且**只有开了 prefix cache 之后才会暴露**（block table 变大才走到 slowpath）。
   驱动同时会打印一条 `EE1016 ... operation not permitted when a stream is capturing`，
   **那是误导** —— 分配失败在 rlimit 上，不在捕获模式上。

2. **`--quantization ascend` 必须显式传。**
   产物的 `config.json` 里没有 `quantization_config` 字段，
   `AscendModelSlimConfig.override_quantization_method` 只在 `hf_quant_cfg is not None` 时返回
   `"ascend"`，所以自动探测不会触发。

3. **`--prefix-caching-hash-algo xxhash` 需要 `pip install xxhash`，镜像里没有。**
   不装就是 500：`ModuleNotFoundError: xxhash is required for the 'xxhash' prefix caching hash algorithms`。
   （之所以推荐 xxhash：hash 粒度被 `gcd(640, 4)` 钉死在 4 token，32K prompt 要算 ~8192 次。）

## 目录

```
docs/01-quantization.md   量化过程、三份产物、0829 vs 0830 分支分叉、b0830 为什么是废品
docs/02-serving.md        三特性接线、逐级验证记录、每个失败的根因
docs/03-findings.md       代码级结论（每条标注证据强度 F/I/U）
docs/04-open.md           未解问题与已知限制（含多模态）

quantize/run_quant.sh                          三份产物的量化命令
quantize/glm_5_next_w8a8.shared_experts_fp.yaml  0829 + 共享专家不量化（= b0829se 的配方）
quantize/inspect_artifact.py                   产物核对：张量数 / ViT / 量化覆盖面

serve/docker-run.sh                容器启动（含 --ulimit memlock=-1 与 xxhash）
serve/serve.sh                     验证过的启动命令
serve/chat_template_mm.jinja       多模态对话模板（可选，默认不用，见 docs/04-open.md）
serve/make_mm_chat_template.py     从官方模板派生上面那份

test/make_tiny_model.py            造一个结构完整的迷你 GLM-5.3-Flash（不占 8 卡就能验通路）
test/smoke_test.py                 正确性 + prefix cache 命中 + MTP 接受率
test/prefix_cache_test.py          单独测 prefix cache
test/mm_test.py                    图像请求（当前会失败，见 docs/04-open.md）

patches/README.md                  两个被证否的 vllm-ascend 补丁（不要用，留作记录）
```

## 产物

| | 分支 | shared_experts | ViT | tensors | 大小 | 状态 |
|---|---|---|---|---|---|---|
| **b0829** | `glm5_next_quant_0829` | W8A8 | **347** ✓ | 113,353 | 311 G | **推荐** |
| b0830 | `glm5_next_quant_0830` | FLOAT | **0** ✗ | 112,748 | 311 G | **废，别用** |
| b0829se | 0829 + 一行 yaml | FLOAT | **347** ✓ | 113,095 | 312 G | b0829 的单变量对照 |

b0829 vs b0829se 的精度/性能 A/B 还没做，见 `docs/04-open.md`。

## 多模态

模型**是**原生多模态的（官方模型卡原话），347 个 ViT 张量在 b0829 里齐全，
vLLM 启动时的 profile_run 已经在设备上跑过一次视觉塔。

**但官方随包的 `chat_template.jinja` 是纯文本模板**，会把 image content part
换成一句 "You are unable to process this image"，占位符 `<|image|>` 从不出现，
于是每个图像请求都是 HTTP 500（已复现）。

最干净的修法是**一个启动参数、不碰厂商模板**：

```
--chat-template-content-format string   # 可选再加 --interleave-mm-strings
```

**当前没有启用** —— 这是在覆盖厂商明确关掉的开关，先跟上游确认。
图像通路一次都没产出过正确答案。详见 `docs/04-open.md`。
