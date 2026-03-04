# NPU Triton 性能分析参考

## 1. 硬件参数速查（Atlas 800T/I A2 / Ascend 910B）

| 指标 | 数值 |
|------|------|
| GM（HBM）带宽 | ~1.8 TB/s |
| Vector 算力（FP16） | ~11.06 TOPS |
| Vector 算力（FP32） | ~5.53 TFLOPS |
| Cube 算力（FP16） | ~320 TFLOPS |
| Cube 算力（INT8） | ~640 TOPS |
| UB 容量 | 192 KB / Core |
| AI Core 数 | ~25 |
| Vector Core 数 | ~40 |
| Vector 平衡点 | ~6 FLOPs/Byte |
| Cube 平衡点 | ~178 FLOPs/Byte |

## 2. 理论极限计算

### Memory-bound 算子（大多数 Triton 算子）

```
理论极限 = 总搬运量(Bytes) / GM 带宽(1.8 TB/s)
总搬运量 = 输入读取 + 输出写回
         （融合 kernel 的中间结果不算，留在 UB）
```

### Compute-bound 算子（含 tl.dot）

```
理论极限 = 总计算量(FLOPs) / 算力峰值
对于 FP16 矩阵乘: FLOPs = 2 × M × N × K
```

### 融合算子

```
理论极限 = max(搬运理论耗时, 计算理论耗时)
融合的收益 = 减少的 HBM 访问次数 × 单次访问耗时
```

## 3. msprof 命令速查

```bash
# On-device profiling（需要硬件）
msprof op --kernel-name=<kernel_name> python3 test.py

# Simulation profiling（无需硬件）
msprof op simulator --kernel-name=<kernel_name> --soc-version=Ascend910B1 python3 test.py

# 环境变量
export TRITON_PRINT_AUTOTUNING=1       # 打印 autotuning 结果
export TRITON_BENCH_METHOD=npu         # NPU benchmark 方法
export TRITON_DISABLE_LINE_INFO=0      # 启用代码级热点分析
export MLIR_ENABLE_TIMING=1            # 编译耗时统计
```

## 4. PipeUtilization 指标解读

| 指标 | 对应硬件 | 高值含义 |
|------|----------|----------|
| `vec_ratio` | Vector Unit | 向量计算繁忙 |
| `mac_ratio` | Cube Unit | 矩阵计算繁忙 |
| `scalar_ratio` | Scalar Unit | 标量控制繁忙（通常不好） |
| `mte2_ratio` | MTE2 (GM→UB) | 数据搬入繁忙 |
| `mte3_ratio` | MTE3 (UB→GM) | 数据搬出繁忙 |
| `mte1_ratio` | MTE1 (L1→L0) | Cube 数据搬入繁忙 |
| `icache_miss_rate` | 指令缓存 | 指令缓存未命中（不好） |

### 瓶颈判定

```
mte2+mte3 >> vec  →  搬运瓶颈  →  融合/multibuffer
vec >> mte2+mte3  →  计算瓶颈  →  低精度/减计算
scalar 高          →  标量瓶颈  →  增大 BLOCK/向量化
所有都低           →  流水空泡  →  检查 tiling
vec ≈ mte2 且高   →  理想状态  →  接近极限
```

## 5. 性能比判定标准

```
性能比 = 实际耗时 / 理论极限

≤ 1.2   → 优秀，已达极限
1.2~1.5 → 良好，ROI 较低
1.5~2.0 → 可接受，值得优化
> 2.0   → 必须优化
```

## 6. 优化手段优先级

| 优先级 | 手段 | 适用场景 | 预期效果 |
|--------|------|----------|----------|
| P0 | `multibuffer=True` | 搬运与计算未重叠 | 消除流水空泡 |
| P0 | 算子融合 | 多个相邻 kernel | 减少 HBM 访问 50~70% |
| P1 | 增大 BLOCK_SIZE | scalar_ratio 高 | 减少循环/标量开销 |
| P1 | 对齐保证 | VV 32B / CV 512B | 消除对齐惩罚 |
| P2 | heuristics | 有条件分支 | 编译时消除死分支 |
| P2 | do_not_specialize | 变化频繁的参数 | 减少重编译 |
| P3 | 向量化标量 | i64 比较等 | scalar→vector |
| P3 | num_warps/stages 调优 | 通用 | 经验搭配 warps=4/stages=3 |

## 7. 已达极限的确认标志

- [ ] 性能比 ≤ 1.2
- [ ] HBM 带宽利用率 > 80%
- [ ] vec_ratio + mte2_ratio ≈ 1.0（计算搬运重叠）
- [ ] 增大/减小 tile 后性能不再变化
- [ ] multibuffer 开关无差异
- [ ] scalar_ratio < 15%
- [ ] Bank conflict < 5%

## 8. 典型算子性能参考

| 算子 | 典型 shape | 瓶颈类型 | 性能比参考 |
|------|-----------|----------|-----------|
| RMSNorm | (128, 7168) bf16 | Memory-bound | 1.1~1.5 |
| RoPE | (128, 64, 128) bf16 | Memory-bound | 1.2~1.8 |
| SwiGLU+Quant | (128, 18432) bf16→int8 | Memory-bound | 1.3~2.0 |
| tl.dot matmul | (128, 512) × (512, 64) fp16 | Compute-bound | 1.2~1.8 |
| 融合 RMSNorm+RoPE | (128, 7168) bf16 | Memory-bound | 1.1~1.3（融合优势） |
