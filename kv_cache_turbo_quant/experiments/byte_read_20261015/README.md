# 字节查表Ascend C读核候选

| 项目 | 本轮实证 |
|---|---|
| 源仓/分支 | Liuchenbing-2026/vllm-ascend / `perf/tq-byte-read-20261015`；交付SHA由任务23及发布回读元数据固定 |
| 基础SHA | `cbe337489df9e5f5e4e1fc2762b85875463bb4c3` |
| 改动 | 保留66-byte store4/KV两路；uint8→half→int32准确转换替代四轮Shift/And/Gather及重排；共享reader构造参数选择独立op，无新环境变量 |
| 当前实测 | 15组bitwise、20异步、2 LUT更新、4 inference-mode均通过；40次目标核memcheck正覆盖无访问错误，原2MiB退出泄漏保留 |
| 局部微测 | 单形状1个NPU事件区间，5预热+20调用，含Host间隙：120000 tokens两路683.267→484.652µs(-29.07%)；256000 tokens1457.053→1032.115µs(-29.16%) |
| 短批 | 320tokens94.107→95.526µs(+1.51%)，单样本不足以验收短场景不衰减；完整整网目标未达到 |
| 整网 | 本新读核整网尚未验收；CBE/BF16上一批完整开关1992请求仍低11.11%/13.64%/7.40%；不混为本改动收益 |

本轮在新容器`tq-profile-20261014-host18`真实编译Ascend C和Torch扩展。镜像`vllm-ascend:dspark-a2-028`，本地image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`，RepoDigests空，未证明公开可拉取。宿主`/data2/tq-profile-20261014`→容器`/ws`，只读模型`/data1/Qwen3-30B-A3B`→`/models/Qwen3-30B-A3B`。CANN9.1.0、Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post4、TritonAscend3.2.2；框架仍vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a`、Ascend `96df623103f921df1a4488170d106f36689acb59`。复用的框架/量化器未重新编译，未做干净镜像从头复现、语义质量数据集或CI。

## 实际执行与失败保留

首次root`/ws/byte_read_20261015`编译exit0，但缺旧fixture使functional在kernel调用前失败，host.exit1。恢复root`/ws/byte_read_attempt2_20261015`从已归档固定版本复制原始paged_read_torch.py和kvtq_store_reference.py，不修改golden或容差；functional和memory均exit0，原40calls门禁全部保留。随后其他任务占用2/3卡，micro前资源检查拒绝，host.exit1；不能将整个root写成exit0。独立`/ws/byte_read_micro_20261015`在检查空闲4卡后完成同一已编译核的新旧对照，host.exit0。所有微测都在同一4卡比较旧pair与新byte，不与2卡门禁数据拼性能。原始日志/失败和版本在任务23归档。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash
source /ws/paired_read_20261014/opp/vendors/tqreadpair/bin/set_env.bash
# 实际构建入口；输入源码固定在source-manifest，依赖上述既有venv和toolkit：
bash /ws/byte_read_20261015/scripts/build_byte_read.sh
source /ws/byte_read_20261015/opp/vendors/tqreadbyte/bin/set_env.bash
# 已实跑独立精度：
ASCEND_RT_VISIBLE_DEVICES=2 VLLM_ASCEND_KVTQ_BITS=4 /ws/.venv/bin/python /ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py --precision-only --output-dir /tmp/byte_precision
# 已实跑目标核访存；known2MiB退出泄漏单独解释：
ASCEND_RT_VISIBLE_DEVICES=2 VLLM_ASCEND_KVTQ_BITS=4 mssanitizer --tool=memcheck --kernel-name=TqPagedReadByte_85b34c13b2a1e76b7a16bfe98a6e3a5d_0 --leak-check=yes --log-file=/tmp/byte_memory.log -- /ws/.venv/bin/python /ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py --precision-only --output-dir /tmp/byte_memory
# 已实跑微测与原全部门禁复核，实际输出路径见command.json：
ASCEND_RT_VISIBLE_DEVICES=4 VLLM_ASCEND_KVTQ_BITS=4 /ws/.venv/bin/python /ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py --output-dir /tmp/byte_micro
```

服务整网沿用TP2/eager、BF16权重、24GiB KV/card、len8192、budget4096、max_num_seqs512、prefix/async关闭；TQ/BF16同配置。源码需挂载到独立integration并使用新byte vendor及Torch扩展，原量化customize必须先初始化；完整新整网后台启动、压测和验收命令由任务23保存。尚未实跑的模板不作为实测。

UB tile保持32行，删除shift/mask/plane/join buffer，新增byteHalf及扩大index后总静态UB小于原读核，未提高队列深度；不是MC2或核间同步算子。当前CANN dav_c220头文件确认两段Cast支持；CPU整数布局检查5组通过，不代替NPU精度。原量化GM/FFTS告警、原退出泄漏、外部资源抢占均未根治。
