开发高性能 NPU Triton 算子。请先阅读以下 skill 文件获取完整指导：

1. **主文件**: `.agents/skills/npu-triton-dev/SKILL.md` — 全流程 playbook（需求分析→编写→验证→性能分析→优化→集成）
2. **代码模板**: `.agents/skills/npu-triton-dev/references/templates.md` — 9 种 kernel 模板（标准循环/2D Grid/持久化/二级分块/Block Pointer/融合/Fake/Autotune/Heuristics）
3. **性能分析**: `.agents/skills/npu-triton-dev/references/perf-analysis.md` — Roofline 模型 + msprof + PipeUtilization 瓶颈判定
4. **问题排查**: `.agents/skills/npu-triton-dev/references/troubleshooting.md` — UB overflow/CoreDim/数据类型/精度/性能问题

NPU Triton 核心约束（必须遵守）：
- Grid ≤ 物理核数：`get_vectorcore_num()`（纯向量）或 `get_aicore_num()`（含 tl.dot）
- 核内循环：`tl.range(pid, total, num_programs)`
- UB ≤ 192KB：预估 tile 占用，超出则二级分块
- 无 uint64/float64/链式布尔
- 对齐：VV 32B / CV 512B
- 中间计算 float32，存储转回原 dtype

用户需求: $ARGUMENTS
