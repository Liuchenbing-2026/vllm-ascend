# 本轮实验与可复现记录

| 项目 | 状态 |
|---|---|
| 当前结论 | 入队/直接接口均15组逐位、20异步、2项LUT更新及40次访存通过；2MiB退出泄漏另列。长批读取约快4%，短批入队退化15.04%、直接接口退化5.08%；两者未采用，入队整网发布门禁主动拒绝。 |
| 任务分支 | `perf/tq-read-constants-20261014` |
| 基础提交 | `9e188464533435b6b29035d2bee71324cb47e6e5` |
| 交付SHA | 推送后以该分支与发布回读清单为准；编译源码由snapshot/source-manifest固定 |
| 基础镜像 | `vllm-ascend:dspark-a2-028`；本地image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`；RepoDigests为空，未证明可公开拉取 |
| 已执行 | 当前专用容器内构建/上述设备检查；具体命令与exit保留在snapshot/*/command.json |
| 未执行 | 干净镜像从头复现、上游CI、新语义质量数据集；整网收益以完整BF16对照为准 |

容器`tq-opt-20261011-model`，物理卡2/5，2×910B4-1，driver26.0.rc1，CANN9.1.0；Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post4、TritonAscend3.2.2、transformers5.14.1、numpy1.26.4。vLLM固定`2cf0a6915ce544dc493a0990f2ea38d81601128a`，vLLM-Ascend固定`96df623103f921df1a4488170d106f36689acb59`。框架本轮未改源码，未重新声称构建或CI通过。

实际挂载：宿主`/data02/tq-opt-20261011`→容器`/ws`，宿主`/root/tq-opt-20261011`→`/ws/opt_20261011`，模型只读`/models/Qwen3-30B-A3B`。复用先前固定的量化器、原生读vendor、`.venv`及`/ws/opt_20261012/verify`参考数据；这些前提须按任务23既有复现指南准备，不能只下载本目录就宣称干净环境复现已通过。新增依赖仅本分支构建的独立vendor和Torch扩展，实际sha保留在build-manifest。

## 实际启动与验证

后台运行由宿主`python3 -I -u host.py`负责，执行命令、输入源码hash、build-manifest、NPU状态、日志和退出码保留于各snapshot。诊断器退出0不等于访存通过；独立memory-review要求正覆盖。失败目录保留，重跑需新attempt目录。

下列服务脚本保留完整参数；仅paired实验的新整网在此轮实际启动，其余仅历史基线模板或被门禁拒绝，不能作为新整网性能结果。

```bash
#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
mode=$1
source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash
integration=/root/kvtq_integration
store=1
case "$mode" in
  bf16) store=0 ;;
  candidate) integration=/ws/queued_model_20261014/candidate_write/integration ;;
  original) integration=/ws/c128_20261012/baseline/integration ;;
  optimized) integration=/ws/native_combined_20261013/candidate_write/integration ;;
  *) exit 2 ;;
esac
export PYTHONPATH="$integration:/ws/source/vllm:/ws/source/vllm-ascend:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${TASK_NPU_CARDS:-2,5}" VLLM_ASCEND_KVTQ=0 VLLM_ASCEND_KVTQ_BITS=4
export VLLM_ASCEND_KVTQ_STORE="$store" TORCH_EXTENSIONS_DIR=/ws/torch-extensions OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=/ws/c128_20261012/model-triton-cache
unset ASCEND_LAUNCH_BLOCKING
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model,ascend_model_loader,kv_cache_turbo_quant_shadow
profile_args=()
if [ -n "${2:-}" ]; then
  profile_args=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$2\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"delay_iterations\":0,\"max_iterations\":0}")
fi
exec /ws/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B --served-model-name qwen3-30b-tq-ab \
  --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager \
  --no-enable-prefix-caching --no-async-scheduling \
  --max-model-len 8192 --max-num-batched-tokens "${3:-4096}" --max-num-seqs 512 \
  --kv-cache-memory-bytes 25769803776 --gpu-memory-utilization 0.92 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}' \
  --host 127.0.0.1 --port 18377 "${profile_args[@]}"
```

压测由`actual-model-scripts/run_write_ab.py`（paired目录）启动：20/20 C16、2000/200 C60、8000/1 C256，candidate/BF16/此前原生TQ，每场景1预热+3正式，正式计时不打开profiling。所有请求成功、usage/payload完整核对后才能计算整网结果；单输出TPOT不适用。完整启动command与压测命令由snapshot保存。

精度命令为snapshot对应`/ws/.venv/bin/python -u scripts/verify*.py --precision-only --output-dir <独立目录>`；原版独立golden、BF16逐位比较和异步消费者保留。paired原始脚本曾缺少参考模块搜索路径，首次失败保留；实际通过的为`verify_attempt2.py`。未调整golden、容差、请求成功标准或吞吐分母。

## 未解决项

原量化器的工具GM/FFTS告警、2MiB进程退出泄漏和本地磁盘/旧checkout问题仍保留；当前原生读的正覆盖不等于整条TQ路径所有问题已关闭。源码分支用于保留实验与复现，不代表已合入、已达到BF16收益或上游CI通过。
