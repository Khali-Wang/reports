# TbrTransNet 全屏 Transformer 推理加速 — RTX 4090 对比实验报告

## 环境

| 项目 | 值 |
|------|----|
| 硬件 | NVIDIA GeForce RTX 4090 (24 GiB) |
| Python | 3.13.2 |
| PyTorch | 2.7.0+cu126 |
| CUDA runtime | 12.6 |
| 输入 | `1×12×720×1280`（720p，batch 1） |
| 计时口径 | 仅模型 forward（不含输入生成与 host/device 拷贝），warmup 20 / iterations 100 |

## 1. 对比实验总表

基准（baseline）为 fp16 eager（27.857 ms）。加速比 = 基准 mean_ms ÷ 该配置 mean_ms（越大越快）。

| 配置 | mean_ms | fps_from_mean | 加速比 | 峰值显存 (MiB) |
|------|--------:|--------------:|-------:|---------------:|
| baseline：fp16 eager | 27.857 | 35.90 | 1.00× | 774.7 |
| baseline：channels_last | 22.384 | 44.67 | 1.24× | 774.7 |
| baseline：compile reduce-overhead | 16.422 | 60.89 | 1.70× | 36.2 |
| baseline：compile max-autotune | 14.836 | 67.40 | 1.88× | 36.2 |
| 方案一：SDPA 融合注意力 | 23.706 | 42.18 | 1.18× | 522.6 |
| 方案三：CUDA Graph 捕获 | 27.653 | 36.16 | 1.01× | 791.0 |
| 方案四：FP8 量化（eager） | 39.507 | 25.31 | 0.71× | 775.2 |
| 方案四：FP8 + SDPA | 35.260 | 28.36 | 0.79× | 585.3 |
| 方案四：FP8 + compile reduce-overhead | 17.385 | 57.52 | 1.60× | 36.7 |
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

## 3. 为什么加入 FP8 后反而更慢

FP8（`float8_e4m3fn`）本意是把矩阵乘搬到 FP8 tensor core 上、换取约 2× 的 GEMM 算力，
但对该模型却普遍变慢（eager 39.507 vs 27.857 ms、+SDPA 35.260 vs 23.706 ms、
+compile reduce-overhead 17.385 vs 16.422 ms）。原因有：

1. **网络是显存带宽受限，matmul 不是瓶颈。** profiling 显示 GPU 时间大头是 LayerNorm
   （约 32.8%）、softmax、注意力矩阵与 window 转置拷贝等访存密集算子，Linear 的 GEMM
   本身占比有限。FP8 只能降低 GEMM 的"算力"消耗，对访存主导的瓶颈没有帮助。

2. **动态量化引入了额外显存往返。** 每个 Linear 前向都要对激活做 `abs().amax()` 归约、
   除以 scale、再转 fp8；FP8 量化本身相当于"读 fp16 写 1 字节 fp8、再读 fp8 做 GEMM"，
   这些额外读写（再加上 fp8 结果要乘回 scale 才得到 fp16 输出，使用 fp16 是因为 pytorch 框架中，LayerNorm、softmax、GELU、残差相加（逐元素加法）等均不直接接受 FP8 张量，通常需要先将 FP8 转换回 float16/float32 再执行这些操作）抵消了 fp8 GEMM 省下的时间。

3. **只量化了部分算子。** 本次仅量化 36 个 Linear（Q/K/V/proj、fc1/fc2），而卷积、
   LayerNorm、softmax 以及注意力里的 QK^T、A·V 仍是 fp16。FP8 的收益被限制在一个子集里，
   但量化开销却作用在每一个 Linear 上。

4. **matmul 尺寸小（dim=96），tensor core 优势有限。** 即便用 `torch.compile` 把量化融合
   进大 kernel，FP8+compile 仍略慢于 fp16+compile（reduce-overhead 17.39 vs 16.42 ms），
   进一步佐证该模型并非算力受限。


## 4. 结论

- 最优组合 **SDPA + torch.compile(max-autotune) + channels_last** 达到 **11.802 ms / 84.73 FPS**，
  相对 eager 加速 **2.36×**，峰值显存由 774.7 MiB 降至 36.2 MiB（-95%）。
- `torch.compile` 是单方案收益最大者（max-autotune 1.88×），且与 SDPA 叠加后效果进一步提升。
- 该网络为显存带宽受限：CUDA Graph（1.01×）与 FP8（0.71×~1.60×）几乎无益甚至变慢，
  收益主要来自 kernel 融合与消除中间张量物化，而非降低 matmul 精度或减少 launch 次数。
