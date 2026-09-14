## 1. 对比实验总表

RTX4090，基准（baseline）为 fp16 eager（27.857 ms）。加速比 = 基准 mean_ms ÷ 该配置 mean_ms（越大越快）。

| 配置 | mean_ms | fps_from_mean | 加速比 | 峰值显存 (MiB) |
|------|--------:|--------------:|-------:|---------------:|
| baseline：fp16 eager | 27.857 | 35.90 | 1.00× | 774.7 |
| baseline：channels_last | 22.384 | 44.67 | 1.24× | 774.7 |
| baseline：compile reduce-overhead | 16.422 | 60.89 | 1.70× | 36.2 |
| baseline：compile max-autotune | 14.836 | 67.40 | 1.88× | 36.2 |
| 方案一：SDPA 融合注意力 | 23.706 | 42.18 | 1.18× | 522.6 |
| 方案二：CUDA Graph 捕获 | 27.653 | 36.16 | 1.01× | 791.0 |
| 方案三：FP8 量化（eager） | 39.507 | 25.31 | 0.71× | 775.2 |
| 方案四：FP8 + SDPA | 35.260 | 28.36 | 0.79× | 585.3 |
| 方案五：FP8 + compile reduce-overhead | 17.385 | 57.52 | 1.60× | 36.7 |
| **组合最优**：SDPA + compile(max-autotune) + channels_last | **11.802** | **84.73** | **2.36×** | 36.2 |


## 2. 实现原理

- **baseline_fp16_eager**：fp16 半精度、无任何优化，所有算子逐次 eager 执行，作为对比基准。
- **baseline_fp16_channels_last**：将张量内存布局改为 channels_last（NHWC），提升卷积与访存局部性、减少转置开销。
- **baseline_fp16_compile_reduce-overhead**：用 `torch.compile(mode="reduce-overhead")` 把整图编译为 Triton 大 kernel 做算子融合，并以 CUDA graph 消除逐 kernel 的 CPU launch 开销。
- **baseline_fp16_compile_max-autotune**：`torch.compile(mode="max-autotune")` 在整图编译基础上对每个算子穷举自动调优，选出最优 Triton/cuBLAS kernel 实现。
- **方案一 scheme1_fused_attention**：用 `scaled_dot_product_attention`（FlashAttention/内存高效后端）融合 QK^T + softmax + A·V，不再物化 141 MiB 的注意力矩阵。
- **方案二：torch.compile 编译优化（算子融合 + CUDA graph）**：
使用 `torch.compile(mode="max-autotune" / "reduce-overhead")` 对整图做算子融合、重排与静态图优化，将大量小 kernel（尤其是占 32.8% 的 LayerNorm+add、线性层）融合为 Triton 大 kernel，并以 CUDA graph 消除逐 kernel 的 CPU 启动开销。
- **方案三 scheme3_cuda_graph**：用 `torch.cuda.CUDAGraph` 捕获整段 eager 前向并回放，把数百次 kernel launch 合并为一次，消除 CPU→GPU 调度开销。
- **方案四 scheme4_fp8_eager**：将 6 个 block 的 36 个 Linear 权重/激活量化到 `float8_e4m3fn`（逐张量动态 scale），用 `torch._scaled_mm` 在 FP8 tensor core 上做 GEMM；权重只量化一次并缓存。
- **scheme4_fp8_fused_attention**：FP8 量化 Linear 与 SDPA 融合注意力叠加（方案四 + 方案一）。
- **scheme4_fp8_compile_reduce-overhead**：FP8 量化 Linear 后再叠加 `torch.compile(reduce-overhead)`。
- **scheme4_fp8_compile_max-autotune**：FP8 量化 Linear 后再叠加 `torch.compile(max-autotune)`。
- **combined_fused_compile_channelslast**：方案一（SDPA）+ 方案二（compile max-autotune）+ channels_last 三者叠加，为最终最优组合。



# 3. combined_fused_compile_channelslast 阶段级性能分析报告

**实验对象**：最优组合配置 `combined_fused_compile_channelslast`
（`--precision fp16 --fused-attention --compile --compile-mode max-autotune --channels-last`），
硬件 RTX 4080 / torch 2.13.0 / CUDA 13.0，输入 `B×12×720×1280`，仅模型 forward。

**打点方式**：`model.py` 新增 `--stage-timing` 手动打点（CUDA Event 对 + 同名 NVTX range），
默认关闭、不改变任何算子语义（`smoke_test.py` PASS）。

---

## 1. 两次运行与产物清单

| 运行 | 配置 | 总耗时 mean | 用途 |
|---|---|---:|---|
| A 配置 | `max-autotune`（fullgraph=True + CUDA graphs），无打点 | **21.759 ms** | 4080 |
| B 打点采集 | `max-autotune-no-cudagraphs` + fullgraph=False（自动降级，见 §5），nsys 采集 | **21.909 ms**（+0.7%） | 47 个阶段的 CUDA Event 计时 + NVTX 可视化 |


| 文件 | 内容 |
|---|---|
| `combined_fused_compile_channelslast.json` | 运行 A：官方配置 benchmark 输出 |
| `combined_fused_compile_channelslast_stage_timing.json` | 运行 B：含 `stage_timing.stages`（47 阶段 CUDA Event 统计：mean/median/p90/min/max/占比） |
| `combined_stage_timing.nsys-rep` | 运行 B 的 nsys 采集（用 nsys-ui 打开） |
| `combined_stage_timing.sqlite` | nsys-rep 的 SQLite 导出 |
| `combined_stage_timing_nsys_analysis.md` | sqlite 分析：47 阶段 GPU busy 表 + Top kernel 表（**仅统计 100 个稳态迭代**） |
| `combined_stage_timing_nsys_stats.md` | `nsys stats` 标准报表存档（含 warmup 污染警告） |
| `analyze_nsys.py` | 上述分析的可复现脚本 |
| `run_stage_timing_nsys.log` | 运行 B 的完整日志 |

---

## 2. 网络一次 forward 的执行过程

输入 4 张 `B×3×720×1280`（TbrColor_1 / TbrColor_3 / Albedo / Normal）。一次前向按顺序经过
以下阶段（括弧内为 NVTX range 名与**每次 forward 的 GPU busy 均值**，来自运行 B 的稳态统计）：

1. **输入拼接与 padding**（`stage/input_cat_pad`，0.042 ms）：
   color 域（TbrColor_1+TbrColor_3 → 6ch）与 gbuffer 域（Albedo+Normal → 6ch）分别 cat 并
   replicate-pad 到 16 的倍数（720×1280 已满足，pad=0，仅为 cat 的一次拷贝）。
2. **输入嵌入卷积**（`stage/conv_color` 0.171 ms + `stage/conv_gbuffer` 0.194 ms）：
   `Conv2d(6→96, k4, s2) + LeakyReLU`，720p 下采样到 **360×640×96** 并转成 BHWC，
   即 **230400 个 token**。
3. **6 × GBufferTransformerBlock**（`stage/blk0` … `stage/blk5`，每层 ≈ 2.82 ms）：
   | block 内顺序 | NVTX range | 含义 | GPU busy（6 层合计） |
   |---|---|---|---:|
   | ① | `blkN/norm1` | norm1_color(color) 与 norm1_gbuffer(gbuffer) 两个 LayerNorm | 1.453 ms |
   | ② | `blkN/window_partition` | BHWC 切成 8×8 窗口：3600 windows × 64 tokens（color、gbuffer 各一次 permute+contiguous） | 1.792 ms |
   | ③ | `blkN/qkv` | 三个 Linear(96→96)：**Q、K 来自 gbuffer，V 来自 color**，再 reshape 成 6 头 × head_dim 16 | 2.651 ms |
   | ④ | `blkN/sdpa` | `F.scaled_dot_product_attention`（flash 融合 kernel） | 6.482 ms |
   | ⑤ | `blkN/attn_out` | 输出投影 Linear(96→96) + window_reverse 还原 BHWC | 1.346 ms |
   | — | （无 range，间隙） | residual add：`color = color + attention`（计入 stage 间隙，见 §4） | — |
   | ⑥ | `blkN/norm2` | LayerNorm | 0.430 ms |
   | ⑦ | `blkN/mlp` | fc1(96→192) + GELU + fc2(192→96) | 2.730 ms |
   | — | （无 range，间隙） | residual add：`color = color + mlp_out` | — |

   > gbuffer 特征在 block 内不被更新，逐层复用。
4. **上采样**（`stage/upsample`，1.547 ms）：BHWC→BCHW 后
   `ConvTranspose2d(96→48, k4, s2) + LeakyReLU + Conv2d(48→3, k3)`，恢复到 720×1280×3 并裁剪掉 padding。
5. **输出残差**（`stage/output_color1_plus_residual`，0.035 ms）：`TbrColor_1 + residual` 得到最终输出。

总账：GPU busy 21.36 ms / wall 21.97 ms，stage 内合计 18.87 ms（88.3%），
stage 间隙（12 次 block 内 residual add）2.49 ms（11.7%）。


## 按阶段类型归并（6 个 block 求和）

| 阶段组 | 说明 | GPU busy 均值 ms | 占 GPU busy |
|---|---|---:|---:|
| `blocks/*/sdpa (x6)` | 6 层窗口内融合注意力 `F.scaled_dot_product_attention`（flash kernel，seq=64、head_dim=16）——全网最大单点 | 6.482 | 30.3% |
| `blocks/*/mlp (x6)` | 6 层 MLP：fc1(96→192) + GELU + fc2(192→96)，带宽受限 GEMM | 2.730 | 12.8% |
| `blocks/*/qkv (x6)` | 6 层 Q/K/V 投影：三个 Linear(96→96)，Q、K 来自 gbuffer，V 来自 color | 2.651 | 12.4% |
| `blocks/*/window_partition (x6)` | 6 层 BHWC→8×8 窗口切分（3600 窗口 × 64 token），color/gbuffer 各一次 permute+contiguous 纯拷贝 | 1.792 | 8.4% |
| `upsample` | BHWC→BCHW 后 ConvTranspose2d(96→48, k4, s2)+LeakyReLU+Conv2d(48→3, k3)，恢复到 720×1280×3 并裁掉 pad | 1.547 | 7.2% |
| `blocks/*/norm1 (x6)` | 6 层注意力前归一化：norm1_color(color) 与 norm1_gbuffer(gbuffer) 两个 LayerNorm | 1.453 | 6.8% |
| `blocks/*/attn_out (x6)` | 6 层注意力输出投影 Linear(96→96) + window_reverse 把窗口拼回 BHWC | 1.346 | 6.3% |
| `blocks/*/norm2 (x6)` | 6 层 MLP 前 LayerNorm | 0.430 | 2.0% |
| `conv_gbuffer` | gbuffer 域嵌入：Conv2d(6→96, k4, s2)+LeakyReLU，720p→360×640，转 BHWC | 0.194 | 0.9% |
| `conv_color` | color 域嵌入：Conv2d(6→96, k4, s2)+LeakyReLU，720p→360×640，转 BHWC | 0.171 | 0.8% |
| `input_cat_pad` | 4 张输入拼接为 color(6ch)/gbuffer(6ch) + replicate pad（720p 下 pad=0，仅 cat 拷贝） | 0.042 | 0.2% |
| `output_color1_plus_residual` | 最终输出 = TbrColor_1 + residual（逐点相加） | 0.035 | 0.2% |

## 详细执行时间表
| stage（NVTX range 名） | 说明 | GPU busy 均值 ms | 中位 ms | 占 GPU busy |
|---|---|---:|---:|---:|
| `stage/input_cat_pad` | 4 张输入（TbrColor_1/3、Albedo、Normal）拼接为 color(6ch)/gbuffer(6ch) 两个张量并 replicate pad 到 16 倍数（720p 下 pad=0，仅剩 cat 拷贝） | 0.042 | 0.042 | 0.2% |
| `stage/conv_color` | color 域嵌入：Conv2d(6→96, k4, s2)+LeakyReLU，720p 下采样到 360×640，NCHW→BHWC（得到 230400 个 token） | 0.171 | 0.171 | 0.8% |
| `stage/conv_gbuffer` | gbuffer 域嵌入：Conv2d(6→96, k4, s2)+LeakyReLU，同样得到 360×640×96 的 BHWC 特征 | 0.194 | 0.194 | 0.9% |
| `stage/blk0/norm1` | 注意力前归一化：norm1_color(color) 与 norm1_gbuffer(gbuffer) 两个 LayerNorm | 0.280 | 0.275 | 1.3% |
| `stage/blk0/window_partition` | BHWC 特征切分成 8×8 不重叠窗口（3600 窗口 × 64 token），color/gbuffer 各一次 permute+contiguous 拷贝 | 0.297 | 0.293 | 1.4% |
| `stage/blk0/qkv` | 三个 Linear(96→96) 投影：Q、K 来自 gbuffer，V 来自 color；reshape 成 6 头 × head_dim 16 | 0.451 | 0.442 | 2.1% |
| `stage/blk0/sdpa` | 窗口内融合注意力 F.scaled_dot_product_attention（flash kernel，seq=64） | 1.084 | 1.066 | 5.1% |
| `stage/blk0/attn_out` | 注意力输出投影 Linear(96→96)，再 window_reverse 把窗口拼回 BHWC 特征图 | 0.228 | 0.225 | 1.1% |
| `stage/blk0/norm2` | MLP 前的 LayerNorm | 0.073 | 0.071 | 0.3% |
| `stage/blk0/mlp` | 两层 MLP：fc1(96→192)+GELU+fc2(192→96) | 0.467 | 0.455 | 2.2% |
| `stage/blk1/norm1` | 同 blk0/norm1（第 1 层） | 0.236 | 0.233 | 1.1% |
| `stage/blk1/window_partition` | 同 blk0/window_partition | 0.297 | 0.293 | 1.4% |
| `stage/blk1/qkv` | 同 blk0/qkv | 0.438 | 0.432 | 2.1% |
| `stage/blk1/sdpa` | 同 blk0/sdpa | 1.086 | 1.063 | 5.1% |
| `stage/blk1/attn_out` | 同 blk0/attn_out | 0.228 | 0.227 | 1.1% |
| `stage/blk1/norm2` | 同 blk0/norm2 | 0.073 | 0.072 | 0.3% |
| `stage/blk1/mlp` | 同 blk0/mlp | 0.455 | 0.448 | 2.1% |
| `stage/blk2/norm1` | 同 blk0/norm1（第 2 层） | 0.237 | 0.230 | 1.1% |
| `stage/blk2/window_partition` | 同 blk0/window_partition | 0.297 | 0.293 | 1.4% |
| `stage/blk2/qkv` | 同 blk0/qkv | 0.444 | 0.433 | 2.1% |
| `stage/blk2/sdpa` | 同 blk0/sdpa | 1.074 | 1.052 | 5.0% |
| `stage/blk2/attn_out` | 同 blk0/attn_out | 0.225 | 0.223 | 1.1% |
| `stage/blk2/norm2` | 同 blk0/norm2 | 0.073 | 0.068 | 0.3% |
| `stage/blk2/mlp` | 同 blk0/mlp | 0.461 | 0.445 | 2.2% |
| `stage/blk3/norm1` | 同 blk0/norm1（第 3 层） | 0.234 | 0.231 | 1.1% |
| `stage/blk3/window_partition` | 同 blk0/window_partition | 0.299 | 0.294 | 1.4% |
| `stage/blk3/qkv` | 同 blk0/qkv | 0.435 | 0.429 | 2.0% |
| `stage/blk3/sdpa` | 同 blk0/sdpa | 1.077 | 1.060 | 5.0% |
| `stage/blk3/attn_out` | 同 blk0/attn_out | 0.223 | 0.219 | 1.0% |
| `stage/blk3/norm2` | 同 blk0/norm2 | 0.072 | 0.071 | 0.3% |
| `stage/blk3/mlp` | 同 blk0/mlp | 0.447 | 0.441 | 2.1% |
| `stage/blk4/norm1` | 同 blk0/norm1（第 4 层） | 0.233 | 0.230 | 1.1% |
| `stage/blk4/window_partition` | 同 blk0/window_partition | 0.301 | 0.294 | 1.4% |
| `stage/blk4/qkv` | 同 blk0/qkv | 0.445 | 0.433 | 2.1% |
| `stage/blk4/sdpa` | 同 blk0/sdpa | 1.069 | 1.054 | 5.0% |
| `stage/blk4/attn_out` | 同 blk0/attn_out | 0.221 | 0.219 | 1.0% |
| `stage/blk4/norm2` | 同 blk0/norm2 | 0.068 | 0.067 | 0.3% |
| `stage/blk4/mlp` | 同 blk0/mlp | 0.455 | 0.439 | 2.1% |
| `stage/blk5/norm1` | 同 blk0/norm1（第 5 层，最后一层） | 0.233 | 0.230 | 1.1% |
| `stage/blk5/window_partition` | 同 blk0/window_partition | 0.301 | 0.295 | 1.4% |
| `stage/blk5/qkv` | 同 blk0/qkv | 0.437 | 0.431 | 2.0% |
| `stage/blk5/sdpa` | 同 blk0/sdpa | 1.093 | 1.059 | 5.1% |
| `stage/blk5/attn_out` | 同 blk0/attn_out | 0.220 | 0.218 | 1.0% |
| `stage/blk5/norm2` | 同 blk0/norm2 | 0.071 | 0.070 | 0.3% |
| `stage/blk5/mlp` | 同 blk0/mlp | 0.446 | 0.442 | 2.1% |
| `stage/upsample` | BHWC→BCHW 后 ConvTranspose2d(96→48, k4, s2)+LeakyReLU+Conv2d(48→3, k3)，恢复到 720×1280×3 并裁掉 pad | 1.547 | 1.530 | 7.2% |
| `stage/output_color1_plus_residual` | 最终输出 = TbrColor_1 + residual（逐点相加） | 0.035 | 0.035 | 0.2% |
| **合计（stage 内）** | | **18.873** | | **88.3%** |
| **未归属（stage 间隙）** | 12 次 block 内 residual add（`color+attention`、`color+mlp_out`），未单独打 range | **2.491** | | **11.7%** |
---

## 4. 性能瓶颈分析

### 4.1 阶段分级（每次 forward 的 GPU busy，稳态）

| 排名 | 阶段 | 耗时 | 占比 | 性质 |
|---:|---|---:|---:|---|
| 1 | **SDPA ×6**（flash attention） | 6.48 ms | **30.3%** | 小 tile（seq 64 / head_dim 16）→ flash kernel 算力利用率仅 ~5%（5.7 GFLOP/层 ÷ 1.08 ms ≈ 5.2 TFLOPS），被 occupancy/带宽卡住 |
| 2 | **MLP ×6**（fc1+GELU+fc2） | 2.73 ms | 12.8% | 带宽受限 GEMM（每层搬运 ≈265 MB，≈580 GB/s，接近 HBM 上限 ~717 GB/s） |
| 3 | **QKV 投影 ×6** | 2.65 ms | 12.4% | 带宽受限瘦 GEMM（M=230400, K=N=96，≈600 GB/s） |
| 4 | **block 内 residual add ×12**（stage 间隙） | 2.49 ms | 11.7% | 纯 elementwise，每次搬 132 MB，≈640 GB/s，带宽受限 |
| 5 | **LayerNorm ×18**（norm1+norm2） | 1.88 ms | 8.8% | 带宽受限 |
| 6 | **window_partition ×6** | 1.79 ms | 8.4% | permute+contiguous 纯拷贝 |
| 7 | **upsample** | 1.55 ms | 7.2% | 其中 dgrad 卷积 0.74 ms（≈46 TFLOPS，全网最接近算力上限的 kernel）+ LeakyReLU/bias 0.53 ms |
| 8 | **attn_out ×6**（proj GEMM 0.69 + window_reverse 拷贝 0.65） | 1.35 ms | 6.3% | GEMM + 纯拷贝 |
| 9 | conv_color + conv_gbuffer | 0.37 ms | 1.7% | 很小 |
| 10 | input_cat_pad / output 残差 | 0.08 ms | 0.4% | 可忽略 |

### 4.2 结论：瓶颈在哪里

1. **这是一个显存带宽（HBM）受限的网络，不是算力受限**。除 SDPA 与上采样 dgrad 外，
   所有耗时大项（GEMM、LayerNorm、residual add、partition/reverse 拷贝）实测都在
   550–650 GB/s，已贴近 RTX 4080 的 ~717 GB/s 有效带宽。**Transformer block 内部合计
   ≈19.4 ms，占 90.7%**；输入/输出卷积合计 <2.5%，不是瓶颈。
2. **单点最大项是 SDPA（30.3%）**：窗口只有 64 token、head_dim 仅 16，flash kernel 的
   tensor core 利用率天然低（~5%），它本质也是"小算力 + 多次读写 Q/K/V/O"的带宽型 kernel。
3. **纯搬运类开销合计 ≈32%**（residual 11.7% + partition 8.4% + reverse 3.1% + LN 8.8%）——
   这是 230400 token × 96ch fp16（44 MB/张量）反复读写 HBM 的直接代价。


### 4.3 下一步优化方向（均不改变算子数学语义）

- **融合 residual add**：12 次 add（2.49 ms）可并入上游 GEMM epilogue 或 LayerNorm 主循环
  （官方 fullgraph 编译已部分做到；eager/打点版它们是独立 kernel）。
- **QKV 合并 GEMM**：Q、K 同输入 gbuffer，可 concat 权重成一次 GEMM；省一次 44 MB 激活读 + 2 次 launch。
- **常驻窗口布局**：相邻 block 间 window_reverse→norm→window_partition 是"还原再切分"的往返拷贝
  （合计 ≈3.4 ms），若让中间特征常驻窗口域布局可消除大半。
- **SDPA backend 调优**：对比 flash / memory-efficient / cuDNN attention 后端在
  seq=64、head_dim=16 下的速度（纯后端选择，不改语义）。
- **upsample 融合**：LeakyReLU 与 bias add 在 graph break 边界未被融合（0.53 ms），
  官方 fullgraph 配置下应已融合；可换 channels_last 友好的 conv 实现进一步压缩 dgrad 时间。

---