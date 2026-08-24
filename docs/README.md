# 文档索引

这是仓库的唯一文档导航页。项目总览从根目录 [`README.md`](../README.md)
开始；CVA02 的实时状态只维护在
[`camera_velocity_ambiguity_02_status.md`](camera_velocity_ambiguity_02_status.md)。

## 1. 当前 CVA02 主线

| 文档 | 用途 | 状态 | 可作为当前结论依据 |
|---|---|---|---|
| [`camera_velocity_ambiguity_02_status.md`](camera_velocity_ambiguity_02_status.md) | 当前分支、门控、数据和下一步 | 当前入口 | 是，状态信息 |
| [`camera_velocity_ambiguity_02_design.md`](camera_velocity_ambiguity_02_design.md) | 理论问题、证据边界和实验定义 | 冻结设计 | 是，协议依据 |
| [`superpowers/plans/2026-08-24-camera-velocity-ambiguity-02.md`](superpowers/plans/2026-08-24-camera-velocity-ambiguity-02.md) | 14 个实现任务和三人分工 | 执行计划 | 是，实施依据 |
| [`pre_experiments/README.md`](../pre_experiments/README.md) | 前置实验代码语义和当前入口 | 代码说明 | 仅作为运行说明 |
| `pre_experiments/camera_velocity_ambiguity_02/` | CVA02 协议、输入、预测和几何代码 | 随任务逐步建立 | 通过测试后才可用 |

CVA02 的固定身份、ScanNet 完整性门控、H20 路径和结论资格以这组文档为准。

## 2. 历史实验与设计

| 文档/目录 | 用途 | 状态 | 可作为当前结论依据 |
|---|---|---|---|
| [`doc/2026-07-21_Local_Global_Consistency_Design.md`](../doc/2026-07-21_Local_Global_Consistency_Design.md) | 015 local/global 方法设计 | 历史设计 | 仅作继承依据 |
| [`doc/2026-07-28_ScanNet50_Local_Global_Validation_Design.md`](../doc/2026-07-28_ScanNet50_Local_Global_Validation_Design.md) | ScanNet-50 验证设计 | 历史设计 | 不能替代 CVA02 输入 |
| [`doc/2026-07-29_Trajectory_Overlay_Visualization_Design.md`](../doc/2026-07-29_Trajectory_Overlay_Visualization_Design.md) | 轨迹可视化设计 | 历史设计 | 仅作可视化参考 |
| [`doc/2026-07-28_Local_Global_Worktree_Cleanup_Plan.md`](../doc/2026-07-28_Local_Global_Worktree_Cleanup_Plan.md) | 旧工作树清理 | 历史计划 | 否 |
| [`doc/2026-07-16_Camera_Iteration_Worktree_Design.md`](../doc/2026-07-16_Camera_Iteration_Worktree_Design.md) | 早期相机迭代 | 历史设计 | 否 |
| [`doc/2026-07-21_Camera_Head_Amplification_Design.md`](../doc/2026-07-21_Camera_Head_Amplification_Design.md) | camera-head 放大实验 | 历史设计 | 否 |

`doc/` 下的实现计划与设计文件保持原路径，不做物理归档；详见
[`doc/README.md`](../doc/README.md)。

## 3. 数据、评测与工具

| 文件 | 用途 | 状态 |
|---|---|---|
| [`configs/fastvggt_scannet50.txt`](../configs/fastvggt_scannet50.txt) | FastVGGT 官方 50-scene 顺序 | 固定输入 |
| [`doc/2026-07-27_FastVGGT_ScanNet50_Download_Design.md`](../doc/2026-07-27_FastVGGT_ScanNet50_Download_Design.md) | ScanNet 下载和断点续传设计 | 历史/运行参考 |
| [`doc/2026-07-27_FastVGGT_ScanNet50_Download_Implementation_Plan.md`](../doc/2026-07-27_FastVGGT_ScanNet50_Download_Implementation_Plan.md) | 下载实现步骤 | 历史/运行参考 |
| [`doc/2026-07-28_ScanNet50_Local_Global_Validation_Implementation_Plan.md`](../doc/2026-07-28_ScanNet50_Local_Global_Validation_Implementation_Plan.md) | 旧版 ScanNet 评测管线 | 历史管线 |
| [`configs/fastvggt_scannet50.txt`](../configs/fastvggt_scannet50.txt) | 50 场景清单 | 当前协议输入 |

大型数据、权重和正式输出不在 Git 中，固定放在 H20：

```text
/data/yjh/share/datasets/ScanNet
/data/yjh/share/pretrained/VGGT-1B/model.safetensors
/data/output/camera_velocity_ambiguity/<run_id>/
```

## 4. 过程记录与数值证据

| 文件 | 用途 | 状态 |
|---|---|---|
| [`log/2026-07-29_local_global_stitching_and_rotation.md`](../log/2026-07-29_local_global_stitching_and_rotation.md) | stitching、旋转评价和历史决策 | 已完成历史证据 |
| [`log/2026-07-21_round2_local_global_consistency.md`](../log/2026-07-21_round2_local_global_consistency.md) | Round 2 local/global 过程记录 | 历史记录 |
| [`log/2026-07-21_round1_camera_iteration_results.md`](../log/2026-07-21_round1_camera_iteration_results.md) | Round 1 相机结果 | 历史记录 |
| [`results/local_global_consistency/scannet50/stitching_analysis/README.md`](../results/local_global_consistency/scannet50/stitching_analysis/README.md) | 旧结果目录说明 | 历史结果 |

日志中的“结果”只说明对应历史运行，不自动成为 CVA02 的当前结果。
实时进度只写入 CVA02 状态页，避免多个互相矛盾的进度来源。

## 5. 辅助工作区与审计副本

本地 `.codex_runtime` 下还存在以下辅助目录：

```text
.codex_runtime/cva02_dev       # 主开发工作树
.codex_runtime/cva02_local     # 本地整理/同步副本
.codex_runtime/velocity_repo_audit  # 速度相关代码审计副本
.codex_runtime/FastVGGT_audit  # FastVGGT 上游复现审计资料
```

只有 `cva02_dev` 的当前分支可以接受主线代码提交。辅助目录用于比对、审计或
暂存，不是独立研究分支，也不应被当作最新结果来源。

## 6. 维护规则

1. 新实验先写设计和计划，再写代码；
2. 当前协议、进度和下一步只在 CVA02 入口维护；
3. 历史文档不改写，若语义已过时就在入口标明状态；
4. 结论必须注明数据来源、代码提交、运行目录和是否通过完整性门控；
5. 不把大型 H20 产物复制回本地文档目录。
