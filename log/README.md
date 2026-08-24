# 历史过程日志索引

`log/` 是按日期记录的实验过程、诊断和决策证据。日志不是实时监控面板，也不
自动代表 CVA02 当前结果。当前进度只维护在
[`../docs/camera_velocity_ambiguity_02_status.md`](../docs/camera_velocity_ambiguity_02_status.md)。

## 时间线

### 2026-07-16 至 2026-07-21：早期相机实验

- [`2026-07-16_camera_iteration.md`](2026-07-16_camera_iteration.md)：camera iteration；
- [`2026-07-21_autodl_three_stage_setup.md`](2026-07-21_autodl_three_stage_setup.md)：运行环境和三阶段脚本；
- [`2026-07-21_round1_camera_iteration_results.md`](2026-07-21_round1_camera_iteration_results.md)：Round 1 结果；
- [`2026-07-21_round1_5_context_consistency.md`](2026-07-21_round1_5_context_consistency.md)：上下文一致性；
- [`2026-07-21_round1_6_camera_head_amplification.md`](2026-07-21_round1_6_camera_head_amplification.md)：camera-head amplification。

### 2026-07-21 至 2026-07-29：Local/global 与拼接

- [`2026-07-21_round2_local_global_consistency.md`](2026-07-21_round2_local_global_consistency.md)：Round 2 local/global；
- [`2026-07-21_numeric_result_publishing.md`](2026-07-21_numeric_result_publishing.md)：数值证据发布；
- [`2026-07-28_local_global_worktree_cleanup.md`](2026-07-28_local_global_worktree_cleanup.md)：工作树清理；
- [`2026-07-29_local_global_stitching_and_rotation.md`](2026-07-29_local_global_stitching_and_rotation.md)：stitching、旋转评价与最终旧版决策。

## 如何使用历史日志

历史日志可以回答“当时做了什么、为什么做、旧实验观察到什么”，但不能回答
“当前 CVA02 是否已经完成”。引用其中数值时，应同时注明对应提交、数据目录、
运行目录和实验编号；不得把旧的 40-scene evaluation 称为 CVA02 的 fresh holdout。
