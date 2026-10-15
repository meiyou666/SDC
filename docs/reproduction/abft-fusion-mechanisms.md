# ABFT 融合机制与后续问题

本轮针对性核对支持三个可借用机制：在已加载的输入片段上生成校验、在输出转换前归约 FP32 累加器、按算术强度选择局部或全局校验。它们均已有先例。后续值得验证的是这些机制如何与 NVFP4 分块尺度、故障位置和实际 CUTLASS 数据路径配合，不能把融合本身表述为首次提出。

## 论文与代码对应

本地外部源码位于项目总目录 `external/`，不在本仓库内。下表记录实际阅读的入口；没有运行这些外部实现。

| 来源 | 已核对机制与代码入口 | 可借用部分 | 当前限制 |
| --- | --- | --- | --- |
| [FT-SGEMM 作者代码](https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs) | `kernel/ft_sgemm/include/ft_sgemm_small.cuh`：先加载 `prefetch_vector_tile_A/B`，再直接求 `A_c/B_r`，通过 shuffle 归约，用另一输入片段乘校验并保存到共享内存 | 在输入预取后复用寄存器，避免为 checksum 单独重读全局内存；按 tile 保留局部证据 | 此处是 FP32 SGEMM，不能直接推出 FP8/NVFP4 数值效果；也不能仅凭源码结构宣布性能收益 |
| [Arithmetic-Intensity-Guided Fault Tolerance，SC 2021](https://doi.org/10.1145/3458817.3476184)，[作者代码](https://github.com/Thesys-lab/arithmetic-intensity-guided-abft) | `cutlass-abft-thread/include/cutlass/gemm/warp/mma_tensor_op.h` 的 one-sided 分支：对 `ptr_B` 做 `__half2` 求和，再以 `checkB` 和 `ptr_A` 执行额外 MMA，写入 `accum_abft`。README 对应论文 §4–5，并区分 thread/global 实现 | 输入片段上的局部校验；依据算术强度选择校验组织 | 被读代码使用 half 校验，不能视作 FP32 checksum；README 的部分 global 开销排除了可与下一层重叠的 dot-product kernel，不能与完整串行计时直接比较 |
| [FT-Transformer](https://doi.org/10.1145/3712285.3759853)，[作者代码](https://github.com/HuangliangDai/FT-Transformer) | README 描述 strided 编码与 checksum reuse；`src/flash_attn_v2/Optimized/flash_fwd_kernel.h` 创建 `checksum_o/o2`，传给 `gemm_A_in_regs`，随后与 `acc_o` 归约值比较 | 按 MMA 数据分布组织局部校验；跨相邻线性计算复用校验的设计思路 | 源码明确标注一个告警分支 `for overhead analysis only`，并直接改写 `acc_o(0,0,0)`。该性能路径不能作为已完成正确纠错的证据 |
| [AEGIS，OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/lei) | §5.1.1：`compute the checksum of C in the float32 accumulator`，避免最终 BF16 截断；§5.1.3 历史残差自适应阈值；§5.3 用保存上下文重算确认；§6 分 fwd/dgrad 行校验和 wgrad 行列校验 | 输出 cast 前观察 FP32 累加器；感知和确认分离 | 所读原文没有明确说明 Matmul 输入 checksum 随加载复用；FlashAttention 输入加载的描述不能移作该证据。单独 FP32 比较不等于完整 AEGIS |

## 本轮补充的两篇原始文献

通过项目 Docker 内标准库读取以下公开 arXiv HTML；只检索相关章节，未声称全文复现。Transformer Engine 文档候选 URL 返回 404，因此没有用它支撑结论。

| 文献 | 核对内容 | 与本项目的关系 |
| --- | --- | --- |
| [Microscaling Data Formats for Deep Learning](https://arxiv.org/html/2310.10537v3)，§2–3 | 一个 block 由共享尺度 `X` 与码值 `P_i` 构成，逻辑值为 `X P_i`；表中 MXFP4 为 32 元素/E8M0 尺度；尺度与码值的物理布局不作强制规定 | checksum 应对应逻辑值，不能混淆码值求和与带尺度求和；此文支持表示方式，不支持 ABFT 检测率结论 |
| [Pretraining Large Language Models with NVFP4](https://arxiv.org/html/2509.25149v2)，§2、§4.2–4.4、附录 B | NVFP4 采用 16 元素 block、E4M3 block scale、FP32 tensor scale；训练还涉及 2D scaling、随机 Hadamard 变换和随机舍入 | 可研究尺度读取/传播与校验组织；必须区分实际运行的 1D/2D 缩放和前后向布局。此文是量化训练机制来源，未被读作 ABFT 方法 |

官方实现入口：[CUTLASS example 72a](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu)。本地同名源码已存在，声明 NVFP4 输入与 BF16 输出。后续修改应基于仓库固定 CUTLASS 版本，不能用网页 main 分支取代版本记录。

## 可否定的候选问题

本阶段只考虑计算故障，输入码值与显存中的尺度保持正确；显存翻转不纳入当前基线。

| 候选问题（尚未验证） | 支持这一问题的机制 | 必须控制的反证与否定条件 |
| --- | --- | --- |
| 输入加载后生成 checksum，是否比单独校验 kernel 更省总时间？ | FT-SGEMM 和 AI-guided 已有片段复用实例 | 额外归约、寄存器和同步可能降低 occupancy；在同算子、同保护范围下，计入全部 kernel 后若无稳定收益，则否定本配置的优化收益 |
| 局部 checksum 是否改善 NVFP4 正常残差与故障残差的可分性？ | 局部范围减少归约长度，已有 thread/strided 组织可借用 | 局部多次判定增加每次 GEMM 的误报机会；固定 GEMM 级 FPR、同故障事件后若无 TPR/开销收益，则否定优势。不能用单个 tile FPR 代替 GEMM FPR |
| 能否按 block16 尺度边界传播校验，减少独立反量化或尺度重读？ | NVFP4 的表示及官方 block-scaled GEMM 提供明确尺度边界 | 不同输出列/行通常具有不同尺度，不能把尺度任意移出求和；若校验需要等量或更多尺度访存、同步和计算，则否定性能假说 |
| 码值 × scale 计算或分支中的尺度传播错误，是否需要更早检查？ | 尺度参与多个逻辑值的计算，相关计算错误可能影响多个输出 | 在显存输入正确的条件下，分别注入缩放计算、主计算与校验分叉后、累加器中的错误；若现有独立参考已在相同成本下覆盖，则否定新增检查的必要性 |
| 不改变主 GEMM 流水线级数，能否通过 checksum 资源分配降低总开销？ | FP8 融合中已观察到 scratch 2048 超过 SM120 共享内存上限；改为 1024 并将参考校验放入寄存器后，保持原 6 stage 编译通过 | 尚无 GPU 性能结果；额外寄存器可能降低 occupancy。需比较总时间、寄存器和共享内存占用；若总时间无改善，则否定性能收益。资源组织优化已有相关先例，具体对应仍需进一步检索 |

尺度传播应从真实布局推导。例如 `A_ik = g_A sA_(i,b(k)) qA_ik` 时，行校验预测仍需计算 `sum_k A_ik sum_j B_kj`；当 `B` 的尺度随 `j` 变化，`sum_j sB_(j,b(k)) qB_kj` 内的尺度不能统一提出。该式是逻辑值恒等式，浮点实现仍有归约次序误差。

优先顺序：先复现输入片段复用与 cast 前累加器校验，再比较 global/tile/block16 的同条件版本。每次只改一个机制，记录误报、漏报、故障命中与总开销。当前没有新性能、覆盖率或首次性结论。

## 限定补检：FP8/FP4 与流水线资源

2026-10-15 通过项目 Docker 的 Python 标准库访问公开检索接口，未安装依赖或运行编译。GitHub repository search 的五组查询均返回零个仓库匹配：

- [ABFT FP8](https://github.com/search?q=ABFT+FP8&type=repositories)
- [ABFT NVFP4](https://github.com/search?q=ABFT+NVFP4&type=repositories)
- [ABFT FP4](https://github.com/search?q=ABFT+FP4&type=repositories)
- [ABFT register pressure](https://github.com/search?q=ABFT+register+pressure&type=repositories)
- [checksum GEMM pipeline stages](https://github.com/search?q=checksum+GEMM+pipeline+stages&type=repositories)

另用上述主题的三个组合查询尝试公开网页搜索。Google 请求发生 TLS EOF；Bing RSS 对 FP8 仅返回无关结果，对 FP4/NVFP4 和流水线资源组合没有返回条目。因此，本轮没有新增可核对原文的 ABFT 论文或作者代码。GitHub 仓库元数据搜索没有覆盖代码全文、所有 README 或论文全文；网页搜索也未形成有效文献覆盖。该结果只能说明本轮未找到，不能据此判断这些机制未被研究，更不能推导首次性。

当前可直接借用的依据仍是前表中的 FT-SGEMM、AI-guided ABFT、FT-Transformer 和 AEGIS。NVFP4 文献补充了尺度与量化语义，没有提供 FP4 ABFT 检测或开销结论。保持流水线级数的资源分配仍是待测工程方案。
