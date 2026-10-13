# 原生Ascend C paged读候选（NPU验证排队中）

| 项目 | 本轮状态 |
|---|---|
| 分支/基础 | perf/tq-native-paged-read-20261013 / 4a88742bfc663b9863dd1703201d6c2326c9e91e |
| 实际构建 | 新TqPagedRead OPP与Torch glue编译/安装成功，独立/ws/native_read_20261013/opp/vendors/tqread，不覆盖原quantize OPP |
| NPU精度/微测 | 等paired_write_20261013_attempt3正式矩阵退出后执行严格BF16 bits对照与固定原Triton读延迟；未称通过 |
| 整网 | 只有位一致且120k token微测更快，native_model_20261013才启动；微测不是整网收益 |
| 目标 | TQ开关TTFT/TPOT/吞吐，20/20无退化、2000/200收益；未达成 |

保留原66B/head真实4bit、BF16 norm、BF16 centroid舍入和每步dense工作区。新增原生page job按物理页分配，32 rows/UB tile，96B只是UB行对齐（GM存储仍66B），256项packed-byte LUT以一个FP32 word搬运两个BF16位模式；四byte planes重排后BF16->FP32乘norm再BF16 RNE，不改量化codebook。物理页总是从当前block_table读，元数据只复用不含物理页ID的逻辑job描述，不保留BF16历史缓存。

目前长读的初始分流阈值65536 aggregate tokens待micro确认，短写阈值1024来自首轮20/20退化与4K prefill改善的筛选，正式收益仍需验证；2/3-bit原路径不变。原分支9e4c0af已有KvTqReadBench，但其96B GM padding、常量LUT和checksum不产生真实paged结果，不能作为本候选数值或整网证据。

## 固定环境与完整运行入口

vLLM2cf0a6915ce544dc493a0990f2ea38d81601128a、Ascend96df623103f921df1a4488170d106f36689acb59未改，真实Qwen3-30B-A3B BF16、TP2、910B4-1 physical2/5、driver26.0.rc1、CANN9.1.0、torch2.10.0+cpu、torch_npu2.10.0.post4、triton-ascend3.2.2。复用tq-opt-20261011-model，基础镜像vllm-ascend:dspark-a2-028，本地image ID sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14，RepoDigests=[]。公共pull与干净镜像从头复跑未验证；基础源码构建/额外依赖与模型准备沿用[前阶段完整复现](../prefill_store_20261013/README.md)。

实际新算子源码部署/ws/native_read_20261013/code，以下构建CPU运行，不占NPU；严格NPU门禁排队，不能在正式矩阵中混跑micro。

```bash
# 容器内，独立构建入口（完整CMake/vendor/编译flags见脚本）
bash /ws/native_read_20261013/scripts/build_native_read.sh
# 空闲同卡后的独立NPU门禁
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=2,5 VLLM_ASCEND_KVTQ_BITS=4
export PYTHONPATH=/ws/native_read_20261013/code/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH
/ws/.venv/bin/python /ws/native_read_20261013/scripts/verify_native_read.py
# 宿主queued模型监督器，先等gate.exit0与micro改善，再等实际空闲卡
nohup python3 -I -u /data02/tq-opt-20261011/native_model_20261013/scripts/host.py > /data02/tq-opt-20261011/native_model_20261013/logs/host.log 2>&1 < /dev/null &
```

完整模型服务命令/环境变量在[scripts/serve.sh](scripts/serve.sh)，同maxseq512/batched4096/maxlen8192/KV24GiB/card，eager/prefix/async关闭；8000/1C256、20/20C16、2000/200C60三场景，各模式1warmup+3formal，独立greedy440 IDs。原quant OPP与三个框架kernel hashes保留；新OPP/glue另记录来源和哈希，不宣称它们与旧native相同。

构建失败保留：首次unset PYTHONPATH导致msopgen导入失败；第二次uint32 API scalar用有符号字面量导致模板冲突；第三次msopgen重复生成非空目录失败；修后复用模板、unsigned常量、保留CANN Python path，attempt4编译安装成功。这不等于NPU精度或性能通过。

未创建上游PR；没有放宽golden/容差，独立对照固定原read bit结果。图模式仍不支持（store本来要求eager），Meta只提供输出形状，不宣称支持capture。小/长分流阈值、实际model收益与mssanitizer等验收尚待补齐，不作为生产采用版本。

## Combined candidate after strict reader verification

The reader has passed all 15 original BF16 bit-equality cases, including physical-page reuse. The measured read path improved for every nonempty fixture; activate the native path for all positive token counts. Combine the already verified paired K/V quantizer/writer, preserving the original 66-byte layout and native quantizer. This combination requires fresh writer gates and matched model measurements; no ON/OFF gain is claimed yet.

Retain failed gate attempts 1 and 2 (missing reference/read module paths). Attempt 3 precision and microcases passed. The first memcheck CLI parsed Python flags as tool flags and returned zero without executing cases; it was rejected. Use the explicit argument separator and require all 15 cases plus all 14 native invocations to finish with the SDK positive no-error line. The unfiltered memory attempt was interrupted because it instrumented unrelated reference operators; preserve its partial log. A filtered attempt checks the exact native kernel.
