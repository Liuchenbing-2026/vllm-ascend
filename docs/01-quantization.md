# 01 —— 量化

## 模型是什么

**GLM-5.3-Flash 是混合注意力架构，不是纯 MLA。** 这条决定了后面 prefix cache 的全部难点。
以下全部实测自 `config.json` 的 `layer_types` / `linear_attn_config`：

```
45 层 = 34 层 linear_attention (KDA，带 conv1d k=4 + forget_gate)
      + 11 层 deepseek_sparse_attention (MLA + lightning indexer)，在 idx 3,7,11,…,43
+ 第 45 层 = MTP 层（也是 DSA）
mlp: 层 0/1/2 dense，其余 42 层 MoE (288 routed + 1 shared, top-8)
每层都有 hyper-connections: hc_attn_{base,fn,scale} / hc_ffn_{base,fn,scale}
                            (mhc=True, hc_mult=4, hc_sinkhorn_iters=20)
MLA:     q_lora 1536 / kv_lora 512 / qk_nope 256 / qk_rope 0 (mla_use_nope) / v_head 256
indexer: index_topk 2048, n_heads 32, head_dim 128, kpool 4, kpool_compress, always_select_tail
KDA:     num_heads 64, head_dim 128, short_conv_kernel_size 4, gate_lower_bound -5.0
vision:  24 blocks, hidden 1024, patch 14, spatial_merge 2, out_hidden 4096  (347 个张量)
vocab 154880, hidden 4096, max_position_embeddings 1,048,576
```

官方模型卡：320B 总参 / 18B 激活，"the first natively multimodal model in the GLM-5 series"，
30T-token 多模态预训练语料。

## 分支：0829 和 0830 是分叉，不是新旧

```
b11d409  (共同祖先: GLM5.3-FLASH 更名 && 更新最终 W8A8 方案)
├── 4b73b71  ViT权重挂载模型树随量化导出自动落盘        (09-02)
│   └── b57416c  quarot: visual merger 补离线旋转      (09-03)  ← glm5_next_quant_0829
└── de0d4b1  「新增共享专家量化」                        (08-30)  ← glm5_next_quant_0830
```

- **0829 反而更新**（09-03），比 0830 多两个 commit
- 0830 唯一多出来的是 yaml 里一行：`exclude: + '*shared_experts*'`。
  注意 commit 标题写「新增共享专家量化」，**代码干的是把共享专家排除出量化**（保持 BF16）

## b0830 是废品

commit `4b73b71`（只在 0829 上）改掉了旧的 ViT 处理方式。旧方式是在 `_save_vlm_assets` 里
拷贝「只含 `model.visual.*` 的源分片」。实测这个 checkpoint：

```
347 个 visual 张量，全部落在 model-00120-of-00120.safetensors
该分片同时含 184 个语言模型张量
→ 纯 visual 分片数 = 0 → 一个文件都拷不到
```

所以 **b0830 的导出目录里 `model.visual.*` 是 0 个**，作为多模态模型不可加载。实测：

```
b0829    tensors=113353  visual=347  rot=True  W8A8=111871  shared_experts={'W8A8_DYNAMIC': 387}
b0830    tensors=112748  visual=0    rot=True  W8A8=111484  shared_experts={'FLOAT': 129}
b0829se  tensors=113095  visual=347  rot=True  W8A8=111484  shared_experts={'FLOAT': 129}
```

**要做「共享专家不量化」，用 0829 + 那一行 yaml（= `quantize/glm_5_next_w8a8.shared_experts_fp.yaml`），
不要用 0830 分支。**

## 量化覆盖面

拆 `quant_model_description.json`（113,358 条）：

```
W8A8_DYNAMIC  111,871   MoE experts (288×43层) + shared_experts + 层0-2 的 dense mlp
FLOAT           1,482   注意力(MLA+KDA)、indexer、mlp.gate + e_score_correction_bias、
                        hc_attn_*/hc_ffn_*、lm_head、embed_tokens、347 visual、
                        MTP 的 eh_proj/enorm/hnorm/shared_head
```

也就是说这个「w8a8」实际上是 **MoE-only 量化**：注意力全 bf16。
体积 599 G → 311 G（52%），与 MoE 占大头一致。

### 那条 `'*gate'` 未匹配告警是良性的

msmodelslim 会打：

```
These exclude patterns are not matched any module, please ensure this is as expected: ['*gate']
```

**不用管。** `mlp.gate` 本来就不是可量化的 Linear 模块，从没进过候选集，
实测在 description 里就是 `FLOAT`（43 层的 `mlp.gate.weight` 和 `e_score_correction_bias` 全是）。
**不需要因为这条告警重新量化。**

## 三处正好对上 vllm-ascend 的地方

### ① 张量改名，被 WeightsMapper 接住了

msmodelslim 通过 HF 模块树加载再 `named_parameters()` 导出，所以写的是 **HF 模块名**，
与官方 release 的扁平命名差 10 个张量：

| release | 量化产物 |
|---|---|
| `layers.N.hc_attn_{base,fn,scale}` | `layers.N.attn_hc.{base,fn,scale}` |
| `layers.N.hc_ffn_{base,fn,scale}` | `layers.N.ffn_hc.{base,fn,scale}` |
| `layers.N.self_attn.A_log` | `layers.N.self_attn.forget_gate.A_log` |
| `layers.N.self_attn.dt_bias` | `layers.N.self_attn.forget_gate.dt_bias` |
| `layers.N.self_attn.f_a_proj` | `layers.N.self_attn.forget_gate.f_a_proj` |
| `layers.N.self_attn.f_b_proj` | `layers.N.self_attn.forget_gate.f_b_proj` |

`vllm_ascend/models/glm5_next.py:109-121` 的 `GLM5_TRANSFORMERS_INTERNAL_WEIGHTS_MAPPER`
正好做 HF→vLLM 的这个映射，**两种命名都能加载**。

### ② MTP 比 release 更完整，还多一个 QuaRot rot.weight

量化产物第 45 层比官方 release 多两个张量：`embed_tokens.weight`、`shared_head.head.weight`
（release 靠 tie 到主 embed/lm_head），另外多一个顶层 **`rot.weight`**（独立文件 `rot.safetensors`）。

`glm5_next.py:104` 定义 `MTP_ROT_WEIGHT_NAME = "rot.weight"`，
`get_spec_layer_idx_from_weight_name` 专门把它路由到 MTP 层。
**vllm-ascend 是照着「QuaRot 量化产物」这个形态设计的，不是照 release。**

### ③ 专家是逐专家非打包布局

`GLM5_PACKED_MODULES_MAPPING["experts"] = ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]`
与产物一致。

> **陷阱**：HF transformers 5.16 内部用的是**打包**的 `experts.gate_up_proj` + 合并的
> `self_attn.conv1d`，与磁盘布局不同（转换发生在 `conversion_mapping.py`）。
> 别拿 HF 的模块名当判据。vllm-ascend 自带的
> `tools/generate_glm5_next_safetensors.py` 生成的也是 HF 布局，**而且不含 MTP 层**，
> 所以它造出来的模型跟真 checkpoint 布局不符 —— 我们另写了 `test/make_tiny_model.py`。

## 量化命令

```bash
git clone https://gitcode.com/qq_46439621/msmodelslim.git
cd msmodelslim && git checkout glm5_next_quant_0829 && bash install.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msmodelslim quant \
  --model_path /data02/GLM-5.3-Flash-BF16 \
  --save_path  /data02/GLM-5.3-Flash-w8a8-b0829 \
  --device npu:0,1,2,3,4,5,6,7 \
  --model_type GLM-5.3-Flash \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 8 卡约 **72 分钟**，6 卡约 **74 分钟**（`--device npu:2,3,4,5,6,7`）
- 峰值每卡 ~20 GB HBM（linear_quant 的 DTS 阶段）—— **卡上有别的负载会 OOM**
- 命中配方 `glm_5_next_w8a8`（QuaRot → flex_smooth_quant(norm-linear) → linear_quant，
  act per_token int8 对称 minmax / weight per_channel int8 对称 minmax，校准集 `mix_calib.jsonl` 48 条）

### 装 msmodelslim 时的坑

镜像里 pip 的 index-url 指向一个 k8s 集群内网 cache（`cache-service.nginx-pypi-cache.svc.cluster.local`），
在这台机器上不可达。要覆盖：

```bash
docker exec -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
            -e PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn <容器> \
  bash -lc "cd <msmodelslim> && bash install.sh"
```
