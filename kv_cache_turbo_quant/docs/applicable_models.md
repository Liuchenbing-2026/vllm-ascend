# kv_cache_turbo_quant 适用模型调研

> 算子约束：`kv_vectors [N, num_kv_heads, head_dim]` BF16,**head_dim 固定 128**；标准 GQA/MQA 注意力、逐 head 独立 KV cache 的模型。
> 判定标准：(1) head_dim == 128;(2) 标准 MHA/GQA/MQA(非 MLA/DSA 等 latent cache 结构);(3) BF16 KV cache 的 dense/MoE 模型均可(MoE 只看 attention 部分)。

## 一、已实测验证(本机 Atlas 800T A2 + vllm-ascend 0.28.0rc,影子模式接入)

| 模型 | model_type | head_dim | kv_heads | 层数 | 适用 | 验证 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | qwen3 | 128 | 8 | 28 | 是 | 已拉起,eager/aclgraph 正常,影子量化 500+ 次调用 |
| Qwen3-8B | qwen3 | 128 | 8 | 36 | 是 | 已拉起,eager 正常,2000+ 次调用 |
| Qwen3-4B | qwen3 | 128 | 8 | 36 | 是 | config 核实 |
| Qwen3-30B-A3B (MoE) | qwen3_moe | 128 | 4 | 48 | 是 | 已拉起(TP=2,双 worker 均量化,key=[N,2,128]),eager 正常 |
| GLM-4-9B-0414 | glm4 | 128 | 2 | 40 | 是 | 已拉起,eager 正常,2500+ 次调用 |
| DeepSeek-V4-Flash (w8a8) | deepseek_v4 | 512(latent) | 1 | 43 | 否 | MLA 结构,KV 已是压缩 latent,无 128 维 per-head KV |
| GLM-5.2-w4a8c8 | glm_moe_dsa | 192 | 64 | 78 | 否 | DSA/MLA 类结构,head_dim=192 |
| Qwen3.5 全系 (2B/4B/9B/27B/35B-A3B) | qwen3_5(_moe) | 256 | 2-4 | - | 否 | head_dim=256,且为混合线性注意力架构 |
| Qwen3.6 (27B/35B-A3B) | qwen3_5(_moe) | 256 | 2-4 | - | 否 | 同上 |
| MinerU2.5-Pro-1.2B | qwen2_vl | 64 | 2 | 24 | 否 | head_dim=64 |

## 二、按架构族推断(HuggingFace config 公开参数)

### 适用(head_dim=128,标准 GQA/MQA)

| 模型族 | 说明 |
|---|---|
| **Qwen3 dense 全系**(0.6B/1.7B/4B/8B/14B/32B) | head_dim=128,kv_heads=8;Qwen3-MoE(30B-A3B/235B-A22B)同为 128 |
| **Qwen2.5** 1.5B~72B | 1.5B/3B/7B/14B/32B/72B head_dim 均 128;0.5B 为 64(不适用) |
| **Llama-3/3.1** 8B/70B、**Llama-3.2-3B**、**Llama-3.3-70B** | head_dim=128;Llama-3.2-1B 为 64(不适用) |
| **Mistral-7B / Mixtral-8x7B / 8x22B** | head_dim=128,SWA 层同样适用(滑窗内逐 head KV) |
| **GLM-4 9B/32B (0414)** | head_dim=128(GLM-4-9B 已实测核实) |
| **InternLM2.5/InternLM3-8B** | head_dim=128 |
| **Yi-1.5 9B/34B、Baichuan2 7B/13B** | head_dim=128 |
| **Qwen2** 1.5B/7B/72B | head_dim=128(0.5B 为 64) |

### 不适用

| 模型族 | 原因 |
|---|---|
| **DeepSeek-V2/V3/V4、Kimi K2 等 MLA 模型** | KV cache 存的是压缩 latent(kv_c_normed 512/576 维 + k_pe),不存在 head_dim=128 的逐 head K/V;TurboQuant 的旋转+QJL 量化语义不适用于 latent cache |
| **Gemma-2/3** | head_dim=256 |
| **Qwen3.5/Qwen3.6、Qwen3-Next** | head_dim=256,且为 Gated DeltaNet 混合线性注意力架构 |
| **GLM-4.5/4.6/5.x (glm_moe_dsa)** | head_dim=92/192,DSA 稀疏注意力结构 |
| **gpt-oss** | head_dim=64 |
| **MiniMax-M2** | 线性注意力混合架构,KV 路径不同 |

## 三、结论

- **核心价值场景**:head_dim=128 的 GQA 模型在长上下文、大 batch 推理时 KV cache 显存占比高;2-4 bit TurboQuant 可将 KV cache 压缩 4~10.7 倍。
- **最大适用面**:Qwen2.5/Qwen3 dense 全系 + Qwen3-MoE + Llama-3 系 + GLM-4(0414)系,覆盖当前昇腾上最常部署的开源模型。
- **不适用边界**:MLA/latent-cache 模型(DeepSeek/Kimi)与 head_dim≠128 的模型;若需支持 head_dim=256(Qwen3.5 系、Gemma),需扩展算子 tiling(K 维度 256)。