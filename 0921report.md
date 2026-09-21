# report.md — 本周 FP8 推理加速实验汇总（含推理时间数据）

本文件汇总本轮（本周）所有**有实际推理时间数据**的实验，按“实验名称 / 核心改进方法 / 推理时间”
列表，并在文末为效果最好的一组实验单独写一小节说明核心修改。

**固定测量条件**（所有数据可比）：RTX 4080（SM89）、Python 3.14.7、torch 2.13.0、CUDA 13.0、
B=1、输入 12×720×1280、seed 3407、FP16 外围、channels-last、
`torch.compile(fullgraph=True, mode=max-autotune)`、fused attention、warmup 20 / iterations 100、
每次 forward 用一对 CUDA event 计时（不含输入生成、校准与编译）。

**对照基准**：唯一 FP16 baseline = `reports/baseline_4080.json`（20.4321 ms，只读锁定，本周未重跑）；
“相对 FP16 baseline”= 20.4321 / 实验 mean。第二轮实验还额外对照“父配置”
（第一轮的静态 FP8 MLP，18.9485 ms，只读历史值）。

---

## 一、实验汇总表 4080

| 轮次 | 实验名称 | 核心改进方法 | 推理时间 mean (ms) | 推理时间 p95 (ms) | 相对 FP16 baseline |
| --- | --- | --- | ---: | ---: | ---: |
| 参照 | FP16 baseline （上周 4080 实验结果） | 全部算子 FP16 + SDPA + channels-last + max-autotune | 20.4321 | 20.5538 | 1.0000× |
| 第一轮 | 动态 FP8 `mlp_fc1` | MLP 第一层投影改 FP8，scale 每帧动态求 amax | 20.6223 | 20.7062 | 0.9908× |
| 第一轮 | 动态 FP8 MLP（fc1+fc2） | MLP 两层都用 FP8，scale 仍动态 | 20.2608 | 20.3295 | 1.0085× |
| 第一轮 | 静态 FP8 `mlp_fc1` | 改为“校准一次、固定复用”的静态 per-tensor scale | 19.8430 | 19.8903 | 1.0297× |
| 第一轮 | **静态 FP8 MLP（fc1+fc2）→ 父配置** | 两层 MLP 均 FP8 且静态 scale，去掉每帧求 scale 开销 | 18.9227 | 18.9862 | 1.0798× |
| 第一轮 | 静态 MLP + `attn_v` | 在 MLP 基础上把 V 投影也换 FP8 | 19.4928 | 19.6132 | 1.0482× |
| 第一轮 | 静态 MLP + `attn_out` | 在 MLP 基础上把 attention 输出投影换 FP8 | 19.0477 | 19.1348 | 1.0727× |
| 第一轮 | 静态 MLP + `attn_q`/`attn_k` | 在 MLP 基础上把 Q/K 投影各自换 FP8（两次独立量化） | 19.1786 | 19.2927 | 1.0654× |
| 第一轮 | 静态 FP8 MLP 三次复测（父配置最终值） | 同上，独立进程 3 次 | 18.9485（3 次均值） | 19.0454 | 1.0783× |
| 第二轮 S2 | A1 `qk_shared_cast` | Q/K 共享**一次**激活量化，再用同一 E4M3 张量跑两个 96×96 GEMM | 18.6851 | 18.7321 | 1.0935× |
| 第二轮 S2 | **A2 `qk_merged_gemm`** | 权重重排为 `[192,96]`，Q/K 合并成**一次** 96→192 GEMM，输出两半为视图（零拷贝） | 18.3381（3 次均值） | 18.4485 | 1.1142× |
| 第二轮 S3 | B1 `b1_fused_norm_partition_v` | Triton 生产者：LayerNorm + 8×8 窗口划分 + V 输入 E4M3 写回，三合一，V 投影转 FP8 | 19.5918（4 次均值） | 19.9334 | 1.0429× |
| 第二轮 S3 | B2 `b2_fused_reorder_proj_cast` | Triton 生产者：SDPA 输出 head 合并/窗口重排 + proj 输入 E4M3 写回，proj 转 FP8 | 19.2933 | 19.3436 | 1.0590× |
| 第二轮 S4 | C1 `c1_fp8_qk` | 自定义 Triton 窗口注意力核心替换 SDPA：Q/K 片上转 E4M3，QKᵀ 用 FP8 MMA，softmax 统计 FP32 | 14.5178（2 次均值） | 14.5617 | 1.4074× |
| 第二轮 S4 | C2 `c2_fp8_av` | 同一核心：QKᵀ 保持 FP16，P（固定 scale 1/448）与 V 片上转 FP8，A·V 用 FP8 MMA | 14.7489（2 次均值） | 15.1330 | 1.3853× |
| 第二轮 S4 | **C3 `c3_fp8_core`（本周最佳）** | 同一核心内 Q/K/V/P 全部片上转 FP8，QKᵀ 与 A·V 都走 FP8 MMA，attention 矩阵不落显存 | **14.5104（3 次均值）** | **14.5859** | **1.4081×** |
| 第二轮 S6 | combo1 = A2 + C3 | 组合两个通过项，Q/K 合并 GEMM 的输出直接喂 FP8 核心（无新增融合） | 14.3374（3 次均值） | 14.9626 | 1.4251× |
| 第二轮 S6 | combo2 = A2 + C3 + B2 边界 | FP8 核心直接写 E4M3，proj 作为消费者走 FP8 GEMM（跳过 B2 的重排 kernel） | 14.3275（3 次均值） | 15.3001 | 1.4261× |

## 效果最好的几组实验 4090 数据
| 轮次 | 实验名称 | 核心改进方法 | 推理时间 mean (ms) | 推理时间 p95 (ms) | 相对 FP16 baseline |
| --- | --- | --- | ---: | ---: | ---: |
| 参照 | FP16 baseline （上周 4090 实验结果） | 全部算子 FP16 + SDPA + channels-last + max-autotune | 11.8092 | 11.8253 | 1.0000× |
| 第一轮 | **静态 FP8 MLP（fc1+fc2）→ 父配置** | 两层 MLP 均 FP8 且静态 scale，去掉每帧求 scale 开销 | 11.9168 | 11.9937 | 0.9910× |
| 第二轮 S2 | **A2 `qk_merged_gemm`** | 权重重排为 `[192,96]`，Q/K 合并成**一次** 96→192 GEMM，输出两半为视图（零拷贝） | 11.5218 | 11.6030 | 1.0249× |
| 第二轮 S4 | **C3 `c3_fp8_core`（本周最佳）** | 同一核心内 Q/K/V/P 全部片上转 FP8，QKᵀ 与 A·V 都走 FP8 MMA，attention 矩阵不落显存 | **8.9827** | **9.0148** | **1.3147×** |

---

## 二、效果最好的实验：`c3_fp8_core` 核心修改说明

**一句话**：把六个 block 里的 `scaled_dot_product_attention` 换成自定义 Triton 核心
`_window_attention_kernel`——在 8×8 窗口内，Q/K/V/P 在片上转成 E4M3，
QKᵀ 与 A·V 都走 FP8 tensor-core MMA，softmax 统计保持 FP32，attention 矩阵从不写入显存。

具体改动（只改实现方式与数据组织，不改网络结构、参数名与数学语义）：

1. **核心算子替换**：新增 `fp8_kernels.py::_window_attention_kernel`
   （grid = 窗口数 × 头数 = 3600×6，`num_warps=4`），一次 kernel 内完成
   `QKᵀ → softmax → A·V`，替换原来的库 SDPA 调用。父配置实测 backend 是
   `pytorch_flash::flash_fwd_kernel`（6 次/forward），启用后该 kernel 完全消失。
2. **片上 FP8 转换，零额外访存**：Q/K/V 由核心自己从 FP16 GEMM 输出转 E4M3
   （`cvt.rn.satfinite.e4m3x2`），不做“先量化写回全局显存、再读回”的额外往返；
   概率 P 用固定 scale `1/448`（P∈[0,1]，无需行内缩放）。
3. **FP8 tensor-core MMA**：编译产物中含 16 条 `mma.sync.aligned.m16n8k32`（E4M3）指令，
   累加器全 FP32；softmax 的行 max/sum 也在 FP32 下完成，只有表示层进 FP8。
4. **布局零拷贝**：核心直接消费投影输出的 `[windows, tokens, 96]`（token-major、head-major）
   张量，因此 SDPA 路径上的 `view → permute(0,2,1,3) → transpose(1,2).reshape(...)` 拷贝链
   全部消失；合并 GEMM（A2）产生的“token stride = 2·dim”视图也能直接吃（stride 作为参数传入）。
5. **head_dim=16 → K=32 零填充**：Triton 的 FP8 MMA 要求 K≥32（能力探针实测 K=16 无法编译），
   因此把 16 维补零到 32，用 mask 保证只写回有效列。
6. **静态 scale 复用**：Q/K/V 的 per-tensor scale 由一次未计时校准得到并写入
   `reports/fp8_other_parts/scales/c3_fp8_core.json`（含配置身份校验，跨配置复用会被拒绝）；
   计时区间内没有任何 scale 计算。
7. **可选 E4M3 写回**：同一个核心可以把结果直接写成 E4M3 交给 proj 的 FP8 边界
   （combo2 复用了这条生产者/消费者边界，省掉单独的重排 kernel）。

**收益结构**（为什么快）：C1/C2/C3 三次测量差异 ≤0.24 ms，而父配置 → 核心的跳变约 4.4 ms，
说明收益主要来自**核心结构与数据流**（去掉拷贝/物化、合并 kernel、片上转换），
而不是“FP8 的 MMA 本身更快”；FP8 的作用是在同样的访存预算下少搬一半数据。


```bash
python benchmark.py \
  --device cuda:0 --batch 1 --height 720 --width 1280 --seed 3407 \
  --precision fp16 --fused-attention --channels-last \
  --compile --compile-mode max-autotune --warmup 20 --iterations 100 \
  --fp8-targets mlp_fc1,mlp_fc2 --fp8-scale-mode static-tensor \
  --baseline-report reports/baseline_4080.json \
  --candidate c3_fp8_core \
  --scale-file reports/fp8_other_parts/scales/c3_fp8_core.json \
  --experiment-id c3_fp8_core \
  --output reports/fp8_other_parts/04_attention_core/c3_fp8_core/run_1.json
```

**改动文件**：`fp8_kernels.py`（新增核心 kernel）、`model.py`（`_core_attention` 路径、
候选开关与校准/缓存生命周期）、`benchmark.py`（`--candidate`/`--scale-file` 与报告字段）。
`state_dict` 键集合与 543,555 参数不变，checkpoint 仍严格加载；`smoke_test.py` PASS。

**注意（限制）**：本轮按要求不评估任何画质指标；静态 scale 用固定测试输入校准，
真实输入范围更大时会被截断；父配置与 FP16 baseline 均为历史只读值，非同场 A/B；
本环境存在会话级漂移（同一命令 14.50 → 14.90 ms，约 2.6%），
因此小于该波动的差值（例如 combo1/combo2 相对 C3 的约 1.2% 均值优势）不算已证实收益。

