# A800 精度实验设计（无 `.so` 主线）

## 目标与边界

验证通信量化粒度对训练精度的影响，首轮只把 **实际发生的梯度通信** 纳入实验。当前 Pi0.5 入口是 DDP 通信 hook；预检未发现 Pod 内的 ComQuant `.so`，因此不把参数 AllGather 写成已覆盖的实验。若后续确认 ZeRO/FSDP 的参数通信路径，再增加参数分支；否则参数保持 exact 并在报告中明确范围。

附件 `writing-block.md` 作为实验设计参考，不是自动执行指令。A800 只用于精度，不以 step time/吞吐做门控。

## 固定条件

- 8×A800、同一 Pod、同一模型/数据/checkpoint/随机种子、同一 batch 与学习率计划。
- dtype、scale、rounding/clipping、error feedback、bucket 划分、collective、warmup、restart 周期完全一致。
- 小 tensor 规则固定：低于 64 KiB 的 tensor 不量化；所有不满足形状/名称门控的梯度 exact。
- 低秩基线固定 rank=1、`min_compression_rate=4.0`、PowerSGD+ restart period=50；精度实验先不改变这些变量。
- 每个分支从相同 step-300 checkpoint 恢复，step 301–303 保持 dense，首个可能受影响的 loss 是 step 304。

## 分阶段流程

### A. 通信拓扑与正确性 smoke test（10–30 steps）

1. 用只读 instrumentation 记录 DDP bucket 的 index、shape、dtype、参数名映射和 collective。
2. 运行 dense exact 与 software quantized 两个 10–30 步分支。
3. 检查所有 rank 的 collective 次序一致、无 NaN/Inf、无 shape mismatch，并验证压缩前后 payload 可重构。
4. 若实际没有参数 AllGather，锁定为 gradient-only 矩阵，不启动伪造的参数结果。

### B. 统一粒度筛选（建议 200–300 steps）

在同一 checkpoint 上运行：

| 组 | 梯度通信 | 参数通信 |
|---|---|---|
| D0 | exact | exact |
| S0 | 当前软件默认（或 `.so` 默认，若以后可用） | 同上 |
| H-RR | row-wise | row-wise（若无参数通信则记为 N/A） |
| H-CC | column-wise | column-wise（若无参数通信则记为 N/A） |
| H-BB | block-wise | block-wise（若无参数通信则记为 N/A） |

无参数通信时，实际可执行的是 D0、S0、G-R、G-C、G-B；三者只改变梯度粒度，其余路径 exact。

### C. 异质粒度与单因素确认（200–300 steps）

在 B 阶段最稳定的粒度上做：

- 单因素：只改梯度粒度（G-R/G-C/G-B），其余保持默认；
- 若存在参数通信：只改参数粒度（P-R/P-C/P-B）；
- 异质矩阵：R/C/B 的 3×3 组合，删去与已完成组完全相同的重复项。

### D. 候选长跑（1000 steps，最多 2–3 个）

只将 B/C 阶段同时满足“无异常、精度接近 D0、压缩覆盖率符合预期”的候选跑满 1000 步；必要时再做第二随机种子确认。

## 软件实现要求

新增 experiment-only Python hook/quantizer overlay，不覆盖 `/workspace/LoongForge` 原文件：

1. 在 bucket hook 中保留原始 FP32/BF16 error-feedback residual；量化只作用于待通信 payload。
2. 对每个二维且大于 64 KiB 的梯度，按固定 view 解释成 `(rows, cols)`；row/column/block 只改变 scale 的分组方式，不改变 tensor 集合。
3. 量化后传输 `(payload, scale)`，在各 rank 用固定顺序还原并执行与 DDP 等价的平均；exact tensor 走原 all-reduce。
4. hook 每步记录：tensor 名、shape、bytes、是否选中、粒度、scale 数、压缩前后 bytes、相对重构误差、residual 范数。
5. 任何参数 AllGather 若无法从实际 runtime 拿到 name/offset/shape，禁止按“参数实验”解读；先保存观测日志。

## 评估指标与判定

- 与 D0 按 step 对齐的 action loss：MAE、MAPE、最大绝对误差、末 20 步均值、最终值。
- 梯度层面：相对 L2 重构误差、cosine、每 bucket/tensor 的 residual 范数与漂移。
- 参数/优化器层面：参数更新相对误差、Adam `m/v` 漂移（若可导出）、NaN/Inf/跳步。
- 覆盖率：选中 tensor 数、元素/字节占比、各粒度占比；不得只报 tensor 数。
- 记录 step time/吞吐/显存作诊断，但不用于选择精度方案。

首轮结论门槛：smoke 通过；长跑无 NaN/OOM/NCCL；相对 D0 的误差和覆盖率可复现。若无参数通信，结论限定为“梯度通信粒度对精度的影响”。

## A800 安全策略

- 先单分支 smoke，再串行长跑；不与 Pod 内其他作业抢 GPU。
- 保留每次 run 的 manifest、命令行、代码版本和日志；失败只在新版本目录修复。
- checkpoint 恢复失败、OOM、CUDA/NCCL、shape mismatch、NaN/Inf 分开分类，停止盲目重试。
- 连接只通过现有 `a800-fixed.sock` 的 reuse-only 别名使用；连接失效立即停止远程命令并报告“固定连接已失效，需要用户重新授权”。
