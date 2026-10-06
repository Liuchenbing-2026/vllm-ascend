# kv_cache_turbo_quant 提交(liuchenbing-2026)

9月社区任务 09:kv_cache_turbo_quant 算子开发(Ascend C / aclnn 在线向量量化编码)。

## 目录

- **`docs/reproduce.md` — 新环境完整复现指南(从零环境到全部验收结论,含预期数字)**
- `docs/design.md` — 算子设计文档(含真机自验证结果与 vllm-ascend 接入收益验证)
- `docs/vllm_ascend_integration.md` — vllm-ascend 接入验证完整记录
- `docs/applicable_models.md` — 适用模型分析
- `code/op/` — OPP 算子源码(op_host / op_kernel / op.json)
- `code/integration/` — vllm-ascend 接入(torch 胶水、影子/store patch、插件、冒烟与精度单测)
- `code/bench/` — 部署、拉起、压测、清理脚本
- `results/` — 全部 bench 原始 JSON、服务/压测日志、聚合脚本 `aggregate.py`

## 结论速览(Atlas 800T A2, Qwen3-30B-A3B TP=2)

- 算子精度:5 组验收 case 与 golden.py 逐位一致(bit-exact)
- 容量收益(store 模式真实落盘):同显存 KV 518,144 -> 2,009,728 tokens(3.88x),40k 上下文最大并发 12.65x -> 49.07x
- 写路径开销(影子模式 A/B):TTFT +3~6%,TPOT 高并发 +5~8%,总吞吐 -5~8%
- 已在 Qwen3-1.7B/8B、Qwen3-30B-A3B、GLM-4-9B 上完成 vllm-ascend 端到端冒烟

复现请直接按 `docs/reproduce.md` 执行。