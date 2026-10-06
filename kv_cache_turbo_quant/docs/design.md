# KvCacheTurboQuant 算子设计文档

> 提交人:liuchenbing-2026;日期:2026-10-05。本文覆盖设计与真机自验证结果,并附 vllm-ascend 端到端接入收益验证(扩展工作)。

# 需求背景(required)

## 需求来源

本设计依据[9月社区任务 kv_cache_turbo_quant 算子开发任务书](https://www.hiascend.com/activities/task-center/details/6d3df013343845608677c85f3c58d4d6)及其附件 `op.json`、`case.json`、`golden.py`。任务书要求:面向标准 MHA/GQA KV cache,用 Ascend C 开发 aclnn 在线向量量化编码算子;验证硬件 Atlas 800T A2,CANN 9.1.0+。标杆为附件 `golden.py:calc_expect_func` 的 PyTorch eager 参考实现;任务未提供同名 TBE 算子,故不填虚构 TBE 对比路径。目标代码仓为 `cann/ops-transformer/experimental/attention`。

## 背景介绍

标准 MHA/GQA 推理按 token 保存 K/V 向量,BF16、`head_dim=128` 时 256 B/head。任务书规定首版采用统一 `mse_bits` MSE 主编码 + 1-bit QJL 残差编码:默认 `mse_bits=3`、`qjl_dim=128` 时 idx 48 B + qjl 16 B + norm/gamma 各 2 B,合计 **68 B/head ≈ 4.25 bit/channel,理论压缩比约 3.76x**。混合 3.5-bit outlier 分组模式与 MLA latent cache 不在首版范围。

### 参考流程

```text
x(BF16) -> x_norm=||x||2 -> u=x/x_norm -> y=H@u
  -> idx=scalar_quantize(y,mse_bits) 位打包           (MSE 主编码)
  -> r=y-dequant(idx) -> gamma=x_norm*||r||2
  -> qjl=sign(S@r) 位打包                              (QJL 残差 sketch)
```

# 需求分析(required)

## 需求描述

aclnn 工程化自定义算子 `KvCacheTurboQuant`:输入 BF16 `[T,H,128]` K/V 向量、FP32/BF16 旋转矩阵 H `[128,128]`、FP32/BF16 投影矩阵 S `[Q,128]`,属性 `mse_bits∈{2,3,4}`(默认 3);输出 4 个张量(见下表)。量化在线完成,面向 prefill 逐 chunk 与 decode 逐 token 写入路径。

| 参数 | 方向 | 类型 | 形状 |
|---|---|---|---|
| kv_vectors | 输入 | BF16 | [T,H,128],T>=1,H∈[4,32] |
| rotation_matrix | 输入 | FP32/BF16 | [128,128],调用方保证正交 |
| qjl_matrix | 输入 | FP32/BF16 | [Q,128],Q=qjl_dim |
| quant_idx | 输出 | uint8 | [T,H,128*mse_bits/8] |
| quant_qjl | 输出 | uint8 | [T,H,Q/8] |
| quant_norm | 输出 | BF16 | [T,H] |
| quant_gamma | 输出 | BF16 | [T,H] |

## 需求拆解

1. 与 `golden.py` 在相同输入下逐位一致(质心表、打包字节序、范数量化均对齐);
2. 双矩阵乘(128x128 旋转、Qx128 投影)走 Cube 单元,标量量化/打包走 Vector 单元;
3. 大 T 场景(预填 2048+ 行)按行分块多核流水,避免片上放不下整批;
4. mse_bits 2/3/4 共用一套 kernel,仅切换质心表与打包位宽。

# 详细设计(required)

## 算子分析

### 数学公式

`x_norm=||x||2; u=x/x_norm; y=H@u; idx=argmin_c||y-c|| (逐维最近质心); y_hat=centroid[idx]; r=y-y_hat; gamma=x_norm*||r||2; qjl=sign(S@r)`。质心为任务书给定的标准正态最优 Lloyd-Max 量化点(2/3/4-bit 分别 4/8/16 级),边界为中点;打包按低位在前的小端位序。

### host 侧设计

- 入参校验、推导输出形状;`T*H` 展平为行数 rows;
- tiling:每 tile 128 行(TILE_ROWS=128),每个 AI Vector 核处理 64 行(HALF_ROWS),AI Cube/Vector 按 1:2 配对;`usedPairs = min(总核数/2, ceil(rows/128))` 行序切分;
- 两个 matmul(128x128 与 Qx128,FP32)的 TCubeTiling 在 host 用 `MatmulApiTiling` 预算,随 tiling data 下发;GM workspace 按 pair 双缓冲(U0/U1/Y0/Y1/R0/R1/P0/P1,每 buf 64 KB),tiling 记录 stride。

### kernel 侧设计

单 kernel 内 Cube+Vector 流水:

1. CopyIn BF16 x(64 行)-> FP32;Vector 求平方和得 x_norm(BF16 写出),归一化得 u 写入 GM workspace;
2. Cube matmul `Y = U @ H^T`(128x128),结果落 GM;Vector 读回 Y,查质心表逐维最近邻得 idx,同时减出残差 r;
3. idx 按 mse_bits 位打包(uint32 累加器 Or 移位),写出 quant_idx;r 求 ||r||2 得 gamma=x_norm*||r||(BF16 写出);
4. Cube matmul `P = R @ S^T`(Qx128),Vector 取符号位打包写出 quant_qjl;
5. U/Y/R/P 四级 GM workspace 双缓冲,Cube 与 Vector 事件同步,相邻 tile 重叠搬运与计算。

## 自验证结果(Atlas 800T A2,CANN 9.1.0)

- **精度**:任务附件 5 组 case(bits 2/3/4、N=1~1024)与 golden.py **逐位一致**(bit-exact);另做 200 轮随机形状回归与 race 测试通过。
- **性能**:逐 chunk 在线量化实测开销见第 6 节 A/B;算子本体吞吐满足在线写入路径要求。
- **真实链路误差**:接入 vllm-ascend 后用真实 K/V 测重构相对误差,mse_bits=3 逐 token 最大 ~0.19-0.27,符合 TurboQuant 3-bit+QJL 预期量级。

# vllm-ascend 端到端接入与收益验证(扩展,超出任务书要求)

接入方式与全部数据见 `docs/vllm_ascend_integration.md`,复现脚本与原始 bench JSON 在 `code/`、`results/`。要点:

- **接入**:OPP 包安装 + torch 胶水(`torch.ops.turboquant.kv_cache_turbo_quant`)+ `vllm.general_plugins` 插件在 EngineCore 子进程安装 patch;已验证 Qwen3-1.7B/8B、Qwen3-30B-A3B(TP=2)、GLM-4-9B 正常生成且输出与基线一致。
- **写路径开销(影子模式 A/B,vllm bench serve,16k/32k x 并发 1/4/16/32)**:TTFT +3~6%,TPOT 高并发 +5~8%,总吞吐 -5~8%。
- **容量收益(store 模式,KV 真实压缩落盘 66 B/向量)**:同 23.72 GiB/卡 KV 显存,Qwen3-30B-A3B 可驻留 518,144 -> 2,009,728 tokens(**3.88x**),40,960 上下文最大并发 12.65x -> 49.07x;等容量省显存约 19 GiB/卡。
- **边界**:decode 读带宽收益依赖融合解压读 kernel(未含在本算子范围);store 模式读侧当前为 torch 暂存实现,不做吞吐对照。

# 风险与后续

1. 融合解压 kernel(`kv_cache_turbo_dequant` 或 paged-attention 内联解压)是吞吐收益解锁项,建议另行立项;
2. 混合 3.5-bit outlier 分组模式、MLA 适配按任务书说明不在首版;
3. 合入 ops-transformer 时将按仓规范精简为 op_host/op_kernel/op_api 标准目录并补仓内 readme。