# Camera Velocity Ambiguity Pre-experiment

本目录是 VGGT 多短窗修复三文档中的第二份，也是最详细的一份：

> 多个短上下文 VGGT 预测是否产生多模态相机修复速度？

它定义 `V0` 前置实验：对一条 500 帧 global prediction 和九个长度 100、stride 50
的 local predictions，在相邻窗口的共享 50 帧上构造 `d_L`、`d_R`，并用冻结评价
区分 `NOT_SUPPORTED`、`SELECTOR_PROBLEM`、`CONTINUOUS_REDUNDANCY` 和
`MULTIMODAL_VELOCITY_SUPPORTED`。

当前状态：`protocol in progress; no scientific conclusion yet`。

结果章节只能从已提交的 result card 更新，至少需要 branch、commit、input manifest
hash、split、run ID、command、artifact path/hash、样本数、四类比例、scene-level
bootstrap CI 和代表性失败案例。未提交的 shell 输出、临时 artifact 或口头观察不得
进入正式结论。

## 构建

在仓库根目录运行：

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document experiment --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document experiment
```

该目录只维护协议与冻结报告，不拥有 H20 上的实验实现和运行 artifact。

