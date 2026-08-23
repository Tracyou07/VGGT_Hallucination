# Theory Foundation

本目录是 [VGGT 多短窗修复三文档](../VGGT_VRFM_DOCUMENT_SET.md)中的第一份：

> 从相机轨迹多解到修复速度歧义：VGGT 长短上下文修复的理论基础

它同时说明两条不同的问题线：固定完整观测后的物理解空间，以及给定 global/local
VGGT evidence 后的修复速度分布。V-RFM 主要对应第二条线；完整解空间拓扑不再被
写成使用 latent 的唯一前提。

当前状态：`protocol in progress; no scientific conclusion yet`。前置实验尚未冻结，
本文不声称已经观察到多模态修复速度。

目标读者是需要先统一“多解、窗口差异、修复速度多模态”含义的 VGGT 研究者。
本文不是 V0 的运行手册，也不是 V-RFM 的网络实现合同；下游证据见
[V0 前置实验](../camera_velocity_ambiguity_preexperiment/README.md)，条件方法见
[V-RFM 方法设计](../variational_rectified_camera_refiner/README.md)。

共享术语遵循总入口：global prediction 是完整 500 帧预测，local prediction 是
100 帧短窗预测，candidate repair 只有通过冻结评价后才能升级为 valid repair。

## 文件

- `camera_trajectory_solution_space.md`：便于理论讨论和内容评审的主稿；
- `camera_trajectory_solution_space.tex`：正式 PDF 排版源；
- `scannet_fixed_observation_experiment_plan.md`：旧的固定 RGB-D 解空间实验计划，作为
  长期物理解空间研究线保留，不是 V0 前置实验；
- `../references/camera_refiner_references.bib`：三文档共享文献库；
- `../references/vggt_vrfm_report.sty`：三文档共享视觉样式；
- `../../output/pdf/camera_trajectory_solution_space_theory.pdf`：生成的 PDF。

## 构建

在仓库根目录运行：

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document theory --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document theory
```

XeLaTeX/BibTeX 缓存写入 `tmp/pdfs/theory/`，逐页 PNG 写入
`tmp/pdf-renders/theory/`。正式数值实验继续在 H20 执行；本文档构建不触碰实验代码
或 artifact。
