# Round 2 Method Pre-experiment

## Long–short native Camera Head fine-tuning

`pre_experiments.long_short_camera_head` fine-tunes VGGT's own Camera Head from
cached 500-frame Camera tokens. Short-window predictions and raw ScanNet poses
exist only in separate privileged training/evaluation sidecars; deployed
inference accepts only the long-token shard and a Camera Head checkpoint. The
H20 entry point is `scripts/h20/run_long_short_camera_head.sh`, and formal
artifacts are written below
`/data/yjh/output/vggt/long_short_camera_head/<run_id>`. The runner first performs
a one-scene smoke test, then trains matched GT-only and quality-weighted
long–short variants on eight scenes and evaluates both on two locked-replay
scenes.

## Variational Camera latent candidate selector

`pre_experiments.variational_camera_selector` trains a prediction-only listwise ranker over
one no-op plus 32 frozen VRFM directions at seven nonzero step sizes. Long-window inputs and
candidate scores are physically separated from GT utility sidecars. The H20 entry point is
`scripts/h20/run_variational_camera_selector.sh`; formal outputs are written only below
`/data/yjh/output/variational_camera_selector/<run_id>`. Each validation scene exports both
prediction-only score grids and the actually selected corrected Camera latents; the privileged
report additionally records score-to-utility calibration without copying labels into those
prediction-only artifacts.

## 当前入口：CVA02

当前正在推进的前置实验是 Camera Velocity Ambiguity 02。它使用独立的协议、
输出和结论边界，主入口为：

- [`CVA02 当前状态`](../docs/camera_velocity_ambiguity_02_status.md)
- [`CVA02 冻结设计`](../docs/camera_velocity_ambiguity_02_design.md)
- [`CVA02 执行计划`](../docs/superpowers/plans/2026-08-24-camera-velocity-ambiguity-02.md)

实现包预期位于 `pre_experiments/camera_velocity_ambiguity_02/`，由协议、输入门控、
预测、几何证据和报告模块逐步建立。ScanNet 完整性校验和 calibration gate 未通过
前，不运行正式 GPU 推理。旧的 `local_global_consistency/` 是继承工具和历史实验，
不是 CVA02 当前结果目录。

`study_type: method_pre_experiment`

`common/` contains the minimal shared runtime used by
`local_global_consistency/`. Retired Round 1, Round 1.5, and Round 1.6
implementations are intentionally excluded from this worktree.

Metrics containing predictions use aligned values for primary conclusions;
raw values and recovered scale are diagnostics. Pure GT baselines use raw data.
Round 2 detection scores are prediction-only. GT may be used only by separately
named validation outputs and is always kept raw.
