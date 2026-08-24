# 历史设计与实现计划索引

`doc/` 保存 2026-07 期间形成的设计、实现计划和数据管线说明。它们解释了
当前 CVA02 为什么继承某些工具，但不是当前协议、进度或结果的 source of truth。
当前入口请先看 [`../docs/README.md`](../docs/README.md) 和
[`../docs/camera_velocity_ambiguity_02_status.md`](../docs/camera_velocity_ambiguity_02_status.md)。

## Local/global 主线

- [`2026-07-21_Local_Global_Consistency_Design.md`](2026-07-21_Local_Global_Consistency_Design.md)：015 的核心设计；
- [`2026-07-21_Local_Global_Consistency_Implementation_Plan.md`](2026-07-21_Local_Global_Consistency_Implementation_Plan.md)：旧实现计划；
- [`2026-07-21_Local_Global_Dataset_Construction_Design.md`](2026-07-21_Local_Global_Dataset_Construction_Design.md)：数据构造和泄漏控制；
- [`2026-07-28_ScanNet50_Local_Global_Validation_Design.md`](2026-07-28_ScanNet50_Local_Global_Validation_Design.md)：ScanNet-50 验证设计；
- [`2026-07-28_ScanNet50_Local_Global_Validation_Implementation_Plan.md`](2026-07-28_ScanNet50_Local_Global_Validation_Implementation_Plan.md)：旧验证管线计划；
- [`2026-07-29_Scene0150_Short_Sequence_Exception_Design.md`](2026-07-29_Scene0150_Short_Sequence_Exception_Design.md)：430-frame 特例。

## 相机迭代与运行设置

- [`2026-07-16_Camera_Iteration_Worktree_Design.md`](2026-07-16_Camera_Iteration_Worktree_Design.md)：早期工作树和相机迭代；
- [`2026-07-21_Camera_Head_Amplification_Design.md`](2026-07-21_Camera_Head_Amplification_Design.md)：camera-head amplification 设计；
- [`2026-07-21_Camera_Head_Amplification_Implementation_Plan.md`](2026-07-21_Camera_Head_Amplification_Implementation_Plan.md)：对应实现计划；
- [`2026-07-21_AutoDL_Three_Script_Setup_Implementation_Plan.md`](2026-07-21_AutoDL_Three_Script_Setup_Implementation_Plan.md)：旧 AutoDL 三脚本设置。

## ScanNet、FastVGGT 与可视化

- [`2026-07-27_FastVGGT_ScanNet50_Download_Design.md`](2026-07-27_FastVGGT_ScanNet50_Download_Design.md)：下载、断点续传和远端布局；
- [`2026-07-27_FastVGGT_ScanNet50_Download_Implementation_Plan.md`](2026-07-27_FastVGGT_ScanNet50_Download_Implementation_Plan.md)：下载实现计划；
- [`2026-07-29_Trajectory_Overlay_Visualization_Design.md`](2026-07-29_Trajectory_Overlay_Visualization_Design.md)：轨迹 overlay 设计，当前 CVA02 要求严格复现上游；
- [`2026-07-21_Numeric_Result_Publishing_Implementation_Plan.md`](2026-07-21_Numeric_Result_Publishing_Implementation_Plan.md)：旧数值结果发布流程；
- [`2026-07-21_Numeric_Result_Publishing_Design.md`](2026-07-21_Numeric_Result_Publishing_Design.md)：数值证据和发布约束。

## 工作树与维护

- [`2026-07-28_Local_Global_Worktree_Cleanup_Plan.md`](2026-07-28_Local_Global_Worktree_Cleanup_Plan.md)：旧工作树清理；
- `scene0150`、可视化和发布文档中的路径以形成日期为准，若与当前 H20 路径冲突，
  以 CVA02 状态页为准。

历史文档不删除、不重写结论；如果需要修正当前实验语义，应在 CVA02 设计或状态页
中明确说明，而不是回写历史记录。
