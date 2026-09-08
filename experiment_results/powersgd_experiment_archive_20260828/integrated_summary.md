# PowerSGD 实验统一汇报

更新时间：2026-09-01

## 1. 统一对比口径

- 对比窗口：step 301–1000（700 个对齐 step）。
- 每个压缩方案与同模型、同 step100 checkpoint 的 dense 基线对齐。
- 训练配置：seed=42、BF16、batch 配置沿用对应模型、rank=1、PowerSGD+ restart period=50；冻结 `v` 从 optimizer step 301 激活。
- 指标：action loss MAE、MAPE、最大绝对误差及其 step、平均 step time、平均吞吐。显存仅在实验结束时确认 8 卡均已释放；实验目录未单独记录峰值显存。

## 2. 主要结果

| 模型 | 方案 | MAE | MAPE | 最大绝对误差（step） | 平均 step time | 平均吞吐 |
|---|---|---:|---:|---:|---:|---:|
| Pi0.5 | dense | 0 | 0 | 0 | 0.9755 s | 12.3022 |
| Pi0.5 | PowerSGD+ rank1 | 0.00767826 | 4.8958% | 0.0223525（318） | 3.4293 s | 3.8529 |
| Pi0.5 | PowerSGD+ + 冻结 `v` | 0.00481505 | 3.0823% | 0.0286356（580） | 2.8407 s | 4.6225 |
| FastWAM | 纯 PowerSGD+ rank1 | 0.00066485 | 0.04836% | 0.00683236（317） | 1.0751 s | 0.9414 |
| FastWAM | dense | 0 | 0 | 0 | 0.8358 s | 1.2061 |
| Cosmos3 | dense | 0 | 0 | 0 | 2.0150 s | 0.5020 |
| Cosmos3 | 纯 PowerSGD+ rank1 | 0.01570647 | 120.2158% | 0.182260（482） | 2.2613 s | 0.4558 |
| Cosmos3 | PowerSGD+ + 冻结 `v` | 1.1635943 | 9418.37% | 6.1166744（963） | 2.2720 s | 0.4538 |

Pi0.5 的冻结 `v` 相比纯 PowerSGD+：MAE 下降约 37%，MAPE 下降约 37%，平均 step time 下降约 17%，吞吐提升约 20%；但仍慢于 dense，且最大单点误差更大。

FastWAM 的纯 PowerSGD+ 与 dense 非常接近（MAPE 0.04836%），但平均 step time 慢约 28.7%，吞吐低约 21.9%。

Cosmos3 的冻结 `v` 日志确认 31/31 个候选压缩参数在 optimizer step 301 进入冻结路径，rank=1 近似重启正常发生；然而动作损失相对 dense 严重偏离，当前实现不具备可接受的精度。

Cosmos3 纯 PowerSGD+ 补跑已完成：warmup_dense、warmup_powersgd_plus、dense 和纯 PowerSGD+ 均成功到 step1000。step301–1000 的 MAE=0.01570647、MAPE=120.2158%，最大绝对误差=0.18225979（step482）；rank=1 压缩在全局 step300（局部 iter199）启动，之后按50步周期重启，共15次。该结果明显优于 Cosmos3 冻结 `v` 版本，但仍显著偏离 dense。

## 3. Cosmos3 问题与修复

早期版本曾因模型仍处于 meta tensor 时直接调用 `.to(device)` 失败（`Cannot copy out of meta tensor; no data!`），另有视频后端 `opencv` 不可用问题。修复版本在 DDP 包装前显式物化 meta tensor，并切换到已安装的 `pyav` 后端；v4 随后完成 dense、冻结-v warmup、dense 主段和 PowerSGD+ 冻结-v 主段全部 step1000。原失败目录和日志均保留，不覆盖成功结果。

FastWAM v1 的 PowerSGD+ warmup 曾被调度中断（SIGTERM），未归类为算法错误；在新目录 v2 幂等重跑后全部分支成功完成。

所有最终成功运行均未出现 NaN、OOM 或 CUDA/NCCL 错误。

## 4. 其它已归档实验

归档还包含 Pi0.5 的 ACP、误差平均、Accordion、Oracle error budget、学习率窗口、固定学习率、EF21、安全 rank1/dense 等实验，以及 FastWAM/Cosmos3 的低学习率窗口实验。它们的原始路径、状态和可复现入口见 `manifest.csv`；统一数值入口为 `integrated_summary.csv`。

## 5. 文件入口

- `manifest.csv`：所有实验远端路径、状态、方案和失败原因。
- `integrated_summary.csv`：可直接用于表格分析的最终数值。
- `remote_snapshot/`：从 A800 保存的文本、JSON/JSONL、CSV 和日志摘录（不含 checkpoint）。
- `progress_20260828.md`：按时间记录的执行过程与连接中断证据。

## 6. Pi0.5 FP8 梯度粒度参考实验（2026-09-08）

从共同 step300 checkpoint 跑到 step1000。满足 `ndim > 1` 且大于 64 KiB 的梯度 tensor 使用量化，其余梯度 exact all-reduce。461 个 tensor、144 个 bucket 被选中，覆盖约 99.975% 梯度字节。

| 方案 | MAE | MAPE | 最大误差 | 最终 action loss |
|---|---:|---:|---:|---:|
| Dense | 0 | 0 | 0 | 0.133954480 |
| Gradient-row | 2.43191e-4 | 0.15535% | 2.24522e-3（step582） | 0.133855477 |
| Gradient-block-128 | 2.74373e-4 | 0.17441% | 1.68952e-3（step838） | 0.134350508 |

Block 首次长跑发生 CUDA OOM；日志保留后仅增加 CUDA allocator 设置重跑成功，量化语义未改变。该实验是 gradient-only 精度参考实现：量化后反量化，再做 exact all-reduce；计时不代表生产 FP8 通信带宽。
