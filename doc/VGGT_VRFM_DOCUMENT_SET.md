# VGGT 多短窗修复与 V-RFM：三文档入口

这组三份材料围绕同一个问题展开：**VGGT 的多个短上下文预测，究竟只是重复/噪声，
还是给出了多个都有效、但不能安全平均的长序列修复方向？**

当前统一状态：`protocol in progress; no scientific conclusion yet`。
三份文档都不把“多个窗口输出”提前写成“多个物理真解”，也不把 V-RFM 当成已选定方法。

## 三份文档各自回答什么

| 文档 | 回答的问题 | 当前状态 | Markdown | LaTeX | PDF 输出 | 依赖实验结果 |
|---|---|---|---|---|---|---|
| 理论基础 | “多解”可能指什么？多个短窗候选为什么还不是多解证据？ | 概念边界已整理；无科学结论 | [主稿](camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.md) | [排版源](camera_solution_space_01_theory_foundation/camera_trajectory_solution_space.tex) | `output/pdf/camera_trajectory_solution_space_theory.pdf` | 否；最终判断要引用 V0 |
| V0 前置实验 | 两个重叠短窗给出的修复方向，是选择问题、连续冗余，还是值得建模的速度多模态？ | 协议已冻结；实验结果待回填 | [主稿](camera_velocity_ambiguity_preexperiment/camera_velocity_ambiguity_preexperiment.md) | [排版源](camera_velocity_ambiguity_preexperiment/camera_velocity_ambiguity_preexperiment.tex) | `output/pdf/camera_velocity_ambiguity_preexperiment.pdf` | 是；它本身就是证据门控 |
| 方法设计 | 如果 V0 支持速度多模态，如何用 sequence-level latent 的 V-RFM 生成完整、连贯的修复？ | 条件方法草案；尚未进入实现 | [主稿](variational_rectified_camera_refiner/variational_rectified_camera_refiner_method.md) | [排版源](variational_rectified_camera_refiner/variational_rectified_camera_refiner_method.tex) | `output/pdf/variational_rectified_camera_refiner_method.pdf` | 是；只有 V0 与 target-builder 两道门都通过才进入训练 |

## 阅读顺序

```text
理论基础：划清物理解空间、窗口差异和修复速度分布
    ↓
V0 前置实验：用冻结评价判断候选之间究竟是什么关系
    ↓
方法设计：只在证据支持时，用 V-RFM 表示多个有效修复速度
```

最重要的两个门槛是：

1. V0 必须先看到“双端有效、方向分离、内部平均/插值变差”的稳定证据；
2. 多个局部候选还必须能组装成多个完整、连续、独立评价有效的 sequence targets。

第一道门回答“是否存在值得建模的速度歧义”，第二道门回答“训练所需的多个有效目标
从哪里来”。任一道门不通过，都不应为了使用 V-RFM 而强行制造多模态标签。

## 统一构建

在仓库根目录运行：

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document all --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document all
```

编译缓存位于 `tmp/pdfs/`，逐页渲染位于 `tmp/pdf-renders/`，最终 PDF 位于
`output/pdf/`。这些都是可复现生成物，不进入 Git；版本控制保存 Markdown、LaTeX、
共享样式、共享文献库和构建脚本。

## 共享术语

- **global prediction**：VGGT 对完整 500 帧序列的一次预测；
- **local prediction**：长度 100、stride 50 的短窗口预测，共九个；
- **candidate repair**：短窗对 global 轨迹提出的候选残差，不等于有效 target；
- **valid repair velocity**：通过冻结、独立评价的修复方向；
- **selector**：在已有候选/样本中判断哪个更好；
- **generator**：产生多个完整修复样本；V-RFM 在本文中只承担这一角色。

