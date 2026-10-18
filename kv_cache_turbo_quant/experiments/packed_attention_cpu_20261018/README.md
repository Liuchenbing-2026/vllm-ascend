# Store4 packed attention CPU candidate

| Item | Status |
|---|---|
| Scope | One-query GQA decode mathematical prototype; no serving dispatch changes |
| Base | 87b101ba2690c55e4314798092d93af095d8c38f |
| Task branch | perf/tq-packed-attention-cpu-20261018 |
| CPU result | 10 new shape/boundary cases and 13 invalid-contract cases passed; final BF16 output compared with zero tolerance |
| Device result | No NPU kernel implemented, compiled, or tested; no device performance claim |
| Delivery goal | Existing full-model speed failure and quality/stability/reproduction gaps remain open |

The old 16:14 read-only notice governed that evidence-recheck round. The current
user instruction restores local development and CPU verification. It does not
authorize an NPU job, container, service, lease, PR, or associated-branch push.

## Frozen layout and interface

The prototype reuses `code/integration/kvtq_store.py`; it does not implement a new
codec, centroid table, quantizer, or rotation generator. Source provenance is the
base above and that file's existing SHA256
`9e80293023796907c35d369c1b7a578d33a781c98c9dd1eec1307fd3e02bf031`.

| Field | Contract |
|---|---|
| Query | CPU BF16 `[batch, query_heads, 128]`, one query per sequence |
| K and V | Separate contiguous uint8 `[physical_blocks, page_size, kv_heads, 66]` |
| Bytes 0..63 | Two nonuniform-centroid indices per byte: dimension 2j in the low nibble and 2j+1 in the high nibble |
| Bytes 64..65 | Stored BF16 norm, little endian; no QJL residual or gamma in this layout |
| Codec rounding | Existing BF16 centroid times stored BF16 norm, with BF16 output; no uniform INT4 reinterpretation |
| Block table | int32/int64 `[batch, max_pages]`; active physical pages must be valid; unused entries are ignored |
| Lengths | One positive integer per nonempty sequence, bounded by the table; empty batch supported |
| GQA | Query head count divisible by KV head count; primary target Hq16/Hkv2/D128 |
| Rotations | Existing FP32 Q times R-transpose, rounded to BF16 before attention; attention output rounded to BF16 before FP32 inverse rotation and final BF16 write |
| Output | BF16 query shape; optional existing CPU output view is written in place |

`packed_decode_cpu` gathers only the active logical tile and delegates its
dequantization to the unchanged store reference through a temporary tile view.
It updates per-head maximum, denominator, and numerator with online softmax.
It does not materialize full-length dense K/V within the candidate. This CPU
allocation structure is not an NPU UB/L1, DMA, synchronization, or tiling proof.

The independent attention oracle materializes K/V with the same existing codec
and uses PyTorch dense SDPA. Both attention calculations intentionally use FP64
to check the recurrence and layout without claiming FIA-equivalent device
reductions. Their final BF16 outputs must be bitwise equal. Passing this new
CPU gate does not relax or satisfy the existing NPU zero-tolerance gate, packed
kernel precision, model quality, stability, or full-model speed requirements.

## New cases and reproducible CPU entry

The new cases cover B16/L20, B60/L2000, B1/L8192, lengths
1/63/64/65/127/128/129/1999/2000/2001, tile sizes 31/64/127/128,
permuted physical pages, unused negative table entries, zero and finite large
norms, alternate GQA ratios, page size64, noncontiguous query/table/output, empty
batch, and unchanged inputs. Invalid lengths, active pages, dtype, GQA, tile
size, and scale must be rejected. These are new candidate checks, not reruns of
the historical model matrix, token comparison, or NPU verification scripts.

Actual local environment: macOS, Python3.9.6, torch2.8.0, numpy2.0.2; two CPU
threads. No container or base image was used for this CPU phase, and no packages
were installed. A future device phase must separately fix its image and native
dependencies; the old device image is not represented as this run's environment.

From the task worktree, with a fresh private report path:

```bash
/usr/bin/python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/verify_cpu.py \
  --report /absolute/private/output/cpu_math_unique_run.json
```

The verifier records source hashes, cases, failures and the CPU-only scope.
Reports are created exclusively. Any failed input is preserved in a separate
fixture rather than overwritten. To run only a new or previously failed case:

```bash
/usr/bin/python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/verify_cpu.py \
  --case ragged_tile_page_tails --report /absolute/private/output/new_case_unique.json
```

Full successful new checks are not rerun without a changed implementation or
new unresolved concern. Results and execution commands remain in the private
task evidence directory, not embedded as historical NPU or full-model evidence.

## Remaining device work and closure

The next implementation needs a real device kernel that consumes the same 66B
pages, preserves the codec rounding, and combines QK, softmax and PV without a
full-length BF16 K/V buffer. NPU FP32 reduction behavior, boundaries, nondefault
streams, memory safety and the existing precision gate remain unverified.
CPU FP64 arithmetic cannot establish any of those device properties.

Only after a concrete kernel/build plan is reviewable should a bounded device
window be requested from the designated resource coordinator. No device window
or lease is scheduled here. Real-model quality and stability and the original
short no-regression / long positive-gain speed targets still require their own
evidence. No old failure or precision threshold is removed or weakened.

## 设备阶段增量：ABI、tiling、workspace与OPP入口

本节接续同一分支，不创建重复任务。e5423f98的10组shape与13项CPU契约检查
保持原证据，本轮没有重跑。新增内容是设备接口与host规划代码，**尚无设备kernel
实现、CANN编译、OPP安装、NPU精度或性能通过记录**。

| 产物 | 路径 | 当前边界 |
|---|---|---|
| 算子schema | `code/op/TqPackedAttention.json` | 输入/输出/属性顺序已固定，等待CANN生成核验 |
| Host定义与tiling | `code/op/op_host/tq_packed_attention.cpp` | shape/dtype推导、平台UB/AIV查询、workspace声明已准备；未用CANN编译 |
| 共享tiling规则 | `code/op/op_host/tq_packed_attention_plan.h` | Host与本机probe复用同一份预算/选择规则 |
| 设备tiling ABI | `code/op/op_kernel/tq_packed_attention_tiling.h` | v1、80B、8B对齐；字段offset由static_assert固定 |
| 新规划核对入口 | `device_plan.cpp`、`check_device_contract.py` | 只核对新增ABI、UB边界、溢出与构建缺源阻断，不调用旧CPU数学检查 |
| 构建/OPP入口 | `opp_entry.py` | 默认只输出plan；build与install分离，缺实际kernel先拒绝，不生成假核冒充实现 |
| BF16与后续验收契约 | `device_validation_contract.json` | 全部设备/质量/稳定性/性能项仍待执行；门槛不变 |

### 设备kernel ABI v1

固定核函数参数顺序为：

```cpp
__global__ __aicore__ void tq_packed_attention(
    GM_ADDR query_rot, GM_ADDR key, GM_ADDR value,
    GM_ADDR table, GM_ADDR lengths, GM_ADDR centroids,
    GM_ADDR attention_rot, GM_ADDR status,
    GM_ADDR workspace, GM_ADDR tiling);
```

| 字段 | dtype / shape | 语义 |
|---|---|---|
| query_rot | BF16 `[B,Hq,128]` | 原FP32 Query旋转后按原路径转BF16；旋转不塞入首版核心核 |
| key / value | uint8 `[P,S,Hkv,66]` | 两个独立cache；原64B非均匀centroid索引+末尾2B小端BF16 norm |
| table | int32 `[B,W]` | 连续ND；仅访问逻辑长度实际使用的页；无效填充项不得读取 |
| lengths | int32 `[B]` | 每个非空序列KV长度为正且不超过表覆盖；B=0为无工作快返回 |
| centroids | BF16 `[16]` | 直接复用原表的BF16值，不替换为均匀INT4反量化 |
| attention_rot | BF16 `[B,Hq,128]` | core输出；之后仍走原FP32逆旋转并写BF16最终输出 |
| status | int32 `[B,Hkv]` | 每个任务独占；0成功，1长度无效，2物理页越界，3norm非有限 |
| scale / api_scratch_bytes | float属性 / int属性 | scale沿用原调用值；scratch须来自实际SDK/API需求证明，当前尚未取得 |

真实任务编号 `job=b*Hkv+kv_head`，负责该KV head对应的连续G=`Hq/Hkv`个Query heads。
所有GM地址乘法使用64位；读取前检查长度和物理页，按`min(tile,页内剩余,实际剩余)`
切段，避免跨物理页做连续DMA。66B行使用非对齐搬运，UB每行按96B规划，
不能从最后一个物理行额外读取不存在的68B/96B尾部。实际DataCopyPad overload、
stride单位、尾部及PIPE同步仍要在目标CANN9.1头文件和设备上验证。

status是原型错误可见性的输出；非零必须判失败，不能把零输出当成功。
未来wrapper必须处理错误，相关同步/回读开销应计入整段attention，不能藏在计时外。
PTA/Meta和服务分流尚未接入，本节不表示生产路径已切换到该核。

### 首版tiling与内存预算

首版方案为AIV-only：按`(B,Hkv)`分工，核内循环G个Query heads和KV tiles；
不做split-KV或跨核归约，不占用L1/L0/Cube工作区。blockDim为
`min(B*Hkv,平台GetCoreNumAiv())`，空batch使用单block快返回。
B1/Hkv2只有2个任务，核利用率可能不足；这是待验证的性能风险，没有最优或加速结论。

tile候选依次为128/64/32/16，选择包含所有buffer后能放入平台实际UB的最大候选；
实际容量通过`GetCoreMemSize(UB)`获取，不能把示例192KiB当成所有芯片的配置。
G=8、tile=64、**仅用于规划的32KiB SDK scratch预留**时预算如下：

| Buffer | 字节 | 生命周期/用途 |
|---|---:|---|
| K/V packed双缓冲，共4槽 | 24,576 | 每槽64行×96B；不能提前覆盖仍在DMA或解码中的槽 |
| 解码K FP32 | 32,768 | 含原BF16舍入后再转FP32的K，供该tile的G个Q heads复用 |
| 解码V BF16 | 16,384 | 保留原centroid×norm的BF16输出边界 |
| phase scratch | 32,768 | 解码临时值、QK乘积与PV阶段分时复用；重叠阶段不得别名 |
| 16行索引/norm展开scratch | 8,192 | 分chunk解码，索引Gather结束后方可复用为norm展开 |
| Query BF16/FP32与FP32分子 | 10,240 | G×128；online累加器跨KV tiles保留 |
| 统计量、scores、LUT与控制 | 608 | 每次一个Q head的scores；G组m/l/carry等状态 |
| SDK/API scratch预留 | 32,768 | **预留值，不是已查询到的实际需求或已验证足够** |
| 合计 | **158,304** | 示例UB输入196,608B时选择tile64；不是设备实测 |

解码索引/norm临时区、phase scratch的重用按阶段串行安排；后续kernel必须落实
MTE2→Vector、Vector→MTE3和队列槽位生命周期，不能只靠这份预算宣称同步安全。
真实Reduce/Softmax API的scratch需求确定后须复核预算；不能把预留调小以制造“放得下”。
本机probe已覆盖UB恰好容纳tile64、少1B降为tile32、恰好容纳tile16及少1B拒绝等边界。

首版**用户GM workspace为0B**，无全长BF16 KV或跨核partial归约缓冲；
这不代表ACLNN总workspace为0。实际调用必须执行生成的GetWorkspaceSize并按返回值分配。
状态输出和attention输出是正式输出tensor，不混作“零workspace”的隐藏临时区。

### 构建与隔离OPP复现入口（本轮均未执行）

入口沿用本仓msopgen工程方式，但不执行旧脚本中固定容器、清空目录及系统OPP安装动作。
默认plan只读取源码与哈希；当前会列出唯一缺失的设备实现文件
`code/op/op_kernel/tq_packed_attention.cpp`。build在检查到它缺失后立即拒绝，
不能让msopgen生成的空壳核通过打包冒充完成。

```bash
/usr/bin/python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/opp_entry.py \
  --action plan --record /absolute/private/fresh-plan.json
```

以下仅供**设备实现完成且共享执行已获有效分配后**使用；本轮未新建申请，也未执行：

```bash
python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/opp_entry.py \
  --action build --cann-root /usr/local/Ascend/cann-9.1.0 \
  --python-bin /ws/.venv/bin/python --work-dir /absolute/task/fresh-opp-build \
  --record /absolute/task/fresh-build-result.json

python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/opp_entry.py \
  --action install --package /absolute/task/generated-package.run \
  --package-sha256 ACTUAL_PACKAGE_SHA256 --opp-prefix /absolute/task/fresh-opp-prefix \
  --record /absolute/task/fresh-install-result.json
```

build只生成/编译/打包，不安装、不运行UT/ST/模型；install另行校验包SHA并要求全新prefix。
它不复用或覆盖已有vendor树，安装后的预期入口为
`vendors/customize/bin/set_env.bash`、`op_api/include/aclnn_tq_packed_attention.h`、
`op_api/lib/libcust_opapi.so`；以实际生成/安装文件为准，缺失直接报错。
全部SDK命令、完整stdout/stderr、源文件/包/安装文件哈希应保留，失败目录不删除。
本轮没有Linux CANN编译环境或OPP产物可供宣称已通过，设备镜像/SDK实际可用性
也须在后续有效分配时固定，不能把历史镜像当成本轮构建证明。

新增host规划检查入口（不调用旧10shape/13契约）：

```bash
/usr/bin/python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/check_device_contract.py \
  --out-dir /absolute/private/fresh-device-contract-run
```

它仅用本机clang编译不依赖CANN的80B ABI与共享预算逻辑，核对schema顺序、溢出及
缺kernel阻断。该结果不是host OpDef的CANN编译，也不是NPU kernel验证。

### BF16对照判据与下一节点

`device_validation_contract.json`固定两层零容差对照：先比较相同packed KV经
原解码+真实BF16 FIA得到的旋转空间输出，再比较包含原两侧旋转的完整attention输出；
均要求BF16逐位相等、rtol=0/atol=0。不得为了适配candidate改变原FIA的scale、
golden或容差。FP32在线归约与FIA归约顺序可能不同，是否满足该门槛仍未知；
若失败，保留反例并定位，不用CPU FP64通过作替代。

原始未量化BF16 KV的整网语义质量是独立验收项；四组新旧TQ的440 tokens不代表该项通过，
历史684请求只证明有限矩阵完成。稳定性、从头复现和原速度目标均不变。
未来性能比较应计入两侧旋转、必要status处理、workspace、host dispatch及同步，
不能只报告核心核耗时或容量收益。

**下一实际节点**：实现尚缺的AIV设备kernel，按本ABI落实66B精确搬运、BF16舍入、
FP32 online softmax/PV和状态输出；确认实际SDK scratch，再完成CANN构建及设备验证入口。
共享执行仍须指定资源协调人的有效分配，本轮没有新增申请或自动排队；
没有设备kernel、OPP实物和设备精度证据之前，整网速度FAIL与完整交付阻断状态不变。
