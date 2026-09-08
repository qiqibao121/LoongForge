# 其它模型兼容性记录

本次 FastWAM 验证使用 A800 Pod 内 `/workspace/LoongForge` 的冻结 `v` 实现，并补齐了该 workspace 原来缺失但配置路由所需的：

- `configs/models/embodied/fastwam.yaml`
- `configs/models/embodied/cosmos3/nano.yaml`
- `examples/embodied/fastwam/run_fastwam_sft_ddp_zero1_finetune.sh`

这些文件从 `/raid0/fastwam/LoongForge_window` 复制；如果目标文件原先存在，远端保留 `.bak_before_freeze_v_models_20260828`。本验证的训练脚本为 `source/run_fastwam_powersgd_plus_freeze_v_fair_a800.sh`，采用两套 0–100 warmup 来生成各自的 DCP step100，随后分别续跑 dense 和冻结 `v` 到 step1000。

冻结 `v` 的有效性判据不是命令行参数出现，而是训练日志必须出现 `PowerSGD+ freeze-v activated`，且日志/metrics 中存在 rank-1 压缩 tensor；只有满足该条件才会将分支标记为成功。

Cosmos3 的默认优化器/训练入口仍需单独 smoke test；若其参数分组或状态恢复不满足冻结 `v` 约束，应保留失败证据而不强行把它当作有效跨模型结论。
