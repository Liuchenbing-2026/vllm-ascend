# C++ 解码调用实验，2026-10-16

| 项目 | 状态 |
|---|---|
| 改动 | 将原 FP32 query旋转、BF16 FIA、FP32输出旋转及原位BF16写回合入一个C++入口；真实66B packed KV与量化/读核不变 |
| NPU验证 | 10组逐位一致（含inference、非连续query/output及哨兵），20组非默认流逐位一致 |
| 局部筛选 | 旋转/FIA/输出段，短368.624→274.335µs、长420.588→283.078µs；每形状5预热/30次，一个event区间，包含提交间隙；不是纯kernel或整网收益 |
| 整网 | 发布时新三模式684请求/440 IDs矩阵待运行，目标未达成；此前Byte TQ比BF16短/长吞吐低10.74%/13.59% |
| 验收 | 同批BF16下20/20 C16吞吐/TTFT/TPOT均不衰减；2000/200 C60吞吐正收益且延迟不衰减；保留全部慢轮，不改golden/容差 |

基础提交`d322006031957796e6d5a601ce4d23a219933915`，任务分支`perf/tq-decode-cpp-20261016`。配套既有Byte读核的执行代码`abbe29ef0520d8210457338da13d101d89edb92c`；设备ELF SHA398a52ac7a8e2220875561e89fceb3288cb1e7bbe94a4e7fc5aefadb1bea3b98。框架源码未改动：vLLM2cf0a6915ce544dc493a0990f2ea38d81601128a，Ascend96df623103f921df1a4488170d106f36689acb59；仅新编译本C++扩展。

实际容器`tq-byte-model-20261015-host3`，镜像完整tag`vllm-ascend:dspark-a2-028`，本地ImageID`sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`，RepoDigests为空，无可获取registry digest证明。Ascend910B4-1双卡TP2，driver26.0.rc1，CANN9.1.0，Python3.12.13，torch2.10.0+cpu/torch_npu2.10.0.post4/TritonAscend3.2.2/transformers5.14.1/numpy1.26.4。保留已安装环境，本轮不是干净镜像从头复现；框架源码及模型校验由后台前置门禁重核。

## 实际编译与独立精度命令

```bash
TORCH_EXTENSIONS_DIR=/ws/decode_cpp_20261016/torch-extensions MAX_JOBS=4 /ws/.venv/bin/python /ws/decode_cpp_20261016/integration/torch_ext/build_decode.py
source /usr/local/Ascend/cann-9.1.0/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 /ws/.venv/bin/python /ws/decode_cpp_20261016/verify_decode_cpp.py
```

## 整网复测脚本和完整服务参数

部署脚本创建独立root`/data02/tq-opt-20261011/decode_cpp_model_20261016`，保存host/monitor/finalizer PID、源码/二进制SHA、完整命令、资源监视和原始请求；异步服务模型回归需要真实NPU，局部bitwise不能替代。候选candidate用本分支，optimized用上轮Byte，bf16关闭TQ。每场景每模式1预热+3正式轮，采集轮不计吞吐。

```bash
TASK_NPU_CARDS=2,5 bash /ws/decode_cpp_model_20261016/scripts/serve.sh candidate
/ws/.venv/bin/python -m vllm.entrypoints.openai.api_server --model /models/Qwen3-30B-A3B --served-model-name qwen3-30b-tq-ab --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager --no-enable-prefix-caching --no-async-scheduling --max-model-len 8192 --max-num-batched-tokens 4096 --max-num-seqs 512 --kv-cache-memory-bytes 25769803776 --gpu-memory-utilization 0.92 --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}' --host 127.0.0.1 --port 18377
/ws/.venv/bin/python /ws/decode_cpp_model_20261016/scripts/bench_client.py --mode candidate --concurrency 16 --input-len 20 --output-len 20 --timeout 3600 --output /tmp/tq_cpp_short.json
/ws/.venv/bin/python /ws/decode_cpp_model_20261016/scripts/bench_client.py --mode candidate --concurrency 60 --input-len 2000 --output-len 200 --timeout 3600 --output /tmp/tq_cpp_long.json
/ws/.venv/bin/python /ws/decode_cpp_model_20261016/scripts/check_model_output.py --mode candidate --output /tmp/tq_cpp_greedy.json
```

沿用原压测、输出token和聚合口径。脚本静态核对、真实算子验证和从头复跑分别报告。新8kC256/128K场景、BF16语义数据集和CI未验证。旧量化GM/FFTS告警与退出2MiB泄漏仍未闭环。本候选是host编排优化，未开发新Ascend C attention融合核，也没有BF16 shadow cache。
