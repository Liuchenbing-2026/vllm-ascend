# 128K输入 / 1K输出：TQ开关对比

状态：2026-10-10 03:18+08:00，正式矩阵仍在运行。本报告会在全部12格取得完整结果或实际失败记录后更新；当前不能得出整组性能结论。

## 固定测试条件

新runtime容器 `tq-128k-ab-20261010-runtime`，.19物理卡0、1，2×Ascend910B4-1（64GiB/卡）；Qwen3-30B-A3B BF16权重、TP=2、eager。vLLM和vLLM-Ascend均从固定源码在新任务目录完成构建，实际模块与native库路径已核验。TQ为固定提交的真实store 4-bit，shadow关闭；原58文件和量化/读写算法未改。

每格同时发起N个请求，N=1/2/4/8/16/32；每请求输入131072个固定种子的整数token IDs，输出1024 tokens，temperature=0、ignore_eos=true。两侧相同payload SHA256；成功记录必须实际usage为131072/1024。单格仅一组，不作为稳态吞吐或可信P99。输入为合成token IDs，未测长文本任务准确率。

两侧统一KV预算25769803776 bytes（24.00GiB/卡）、chunked prefill=4096、max_num_seqs=32，关闭prefix cache与async scheduling。原模型上限40960；两侧统一YaRN factor4/original40960，服务总长度132096。这是性能实验配置，不据此认定128K语义质量通过。

## 容量实测

| 服务 | 初始free（两卡） | KV预算/卡 | 日志GPU KV cache size | 132096总长度最大驻留并发 |
| --- | --- | --- | --- | --- |
| BF16 | 60.57 / 60.58 GiB | 24.00 GiB | 524288 tokens | 3.97x |
| TQ store 4-bit | 60.57 / 60.58 GiB | 24.00 GiB | 2033536 tokens | 15.39x |

容量比为3.878662109375。两侧启动日志的manual KV预算和初始可用显存一致；不是按不同gpu_memory_utilization所得容量。容量比与输出吞吐比单独计算。

## 性能结果与计量

阶段结果见[selected-summary.md](selected-summary.md)。原始JSON保留实际请求usage、完成状态、时延、payload哈希；[selection.json](selection.json)显式选择各格来源，[selected/selection-provenance.json](selected/selection-provenance.json)记录原路径与文件SHA256，不自动用较新重试覆盖旧失败。

输出吞吐=N×1024/整组耗时，从HTTP发送到最后响应结束，包含排队、prefill、decode，排除输入构造和服务启动。TTFT为首次非空SSE文本到达；TPOT=(最后文本−首次文本)/(实际输出tokens−1)，不是逐token ITL。未完成组不计算完整工作吞吐或收益比；SSE文本事件数不当作已生成token数。

已完成BF16 C1/C2/C4/C8的HTTP时限为1800s，均在时限前完整结束。后续TQ及BF16高并发时限7200s；实际命令与每格timeout保留。计划切换中断的旧BF16 C16不计入正式完成结果，原记录保留。

## 已验证与限制

原5组胶水bit-exact案例全部通过（idx/qjl/norm/gamma）；4-bit roundtrip脚本通过。其报告的重建相对L2误差为MSE-only均值0.0957、最大0.1805，带QJL参考均值0.0878；这些是诊断指标，不是MSE数值，也不是本脚本单独的误差验收门槛。实际store v1不保存或使用QJL残差，不能用带QJL参考代表实际store质量。Q侧旋转等价最大相对差0.00166<0.01。未做整模型任务准确率评估。

源码确认每层每解码步解压完整历史KV，再调用BF16注意力，见固定源码[读取路径](https://github.com/Liuchenbing-2026/vllm-ascend/blob/8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b/kv_cache_turbo_quant/code/integration/kvtq_store.py#L148)。当前长上下文解码慢与该路径一致，但未采集正式运行的kernel耗时分解，不能量化各阶段开销占比。

基础镜像 `vllm-ascend:dspark-a2-028` 为宿主既有本地镜像，完整image ID与实际版本见[README.md](README.md)及[environment.json](environment.json)。RepoDigests为空，无可提供的registry digest或拉取地址；未导出镜像。模型配置/tokenizer/index已记录哈希，16个权重文件仅记录尺寸、未验全量weight哈希及来源revision。依赖元数据仍有已披露冲突，不能说pip check全绿。

构建、实际启动、压测、独立精度命令及所有问题台账见[README.md](README.md)。整合复现脚本经过静态核对；本轮实际编译、依赖修正、精度与服务均分步执行，不声明在干净镜像从零一次性复跑通过。
