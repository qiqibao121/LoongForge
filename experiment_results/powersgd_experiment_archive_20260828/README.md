# PowerSGD 实验统一归档

本目录是本轮 PowerSGD/PowerSGD+、Accordion、误差补偿与冻结 `v` 实验的固定汇总入口。

## 归档原则

- 远端 A800 的原始实验目录保持不变；本目录保存可追溯的配置、状态、逐步指标、分析结果和日志摘录。
- 不复制模型 checkpoint、TensorBoard event、core dump 或仓库缓存；需要复现实验时使用 `manifest.csv` 中的远端原路径。
- `remote_snapshot/` 保留从 A800 拉回的文本/JSON/CSV/JSONL/日志证据，目录层级与远端路径一致。
- `integrated_summary.csv` 和 `integrated_summary.md` 是跨模型、跨方法的统一比较入口；缺失指标明确标记为 `pending`，不填估计值。
- 当前活动验证的阶段性状态见 `progress_20260828.md`；实验完成后会把其中的 pending 项更新为最终统计。
- 长期对象存储按 `PROJECT_MEMORY.md` 中的 BOS 前缀保存；同步时不得使用 `--delete`。

## 模型与验证范围

- Pi0.5：已有 dense、标准 PowerSGD、PowerSGD+、误差平均、Accordion、Oracle/Tensor 选择、冻结 `v` 等结果。
- FastWAM：准备同配置的 dense 与 PowerSGD+ 冻结 `v` 配对验证；冻结 `v` 仅在明确有 rank-1 压缩 tensor 的 step 生效。
- Cosmos3：先完成框架/优化器兼容性检查。其默认 TorchFusedAdamW 与现有冻结 `v` 路径若不兼容，不以失败运行冒充验证结果。
- Pi0.5 FP8 粒度：已完成 gradient-only 的 Dense/row/block-128 1000-step 精度参考对比；列粒度在 Stage B 筛选后未进入长跑。该实现执行本地量化/反量化后 exact all-reduce，不能作为生产带宽测量。

## 固定远端前缀

`/raid0/fastwam/experiments/`

所有实验的原始路径、状态和归档时间记录在 `manifest.csv`。成功结果不得覆盖；失败尝试保留并标注原因。
