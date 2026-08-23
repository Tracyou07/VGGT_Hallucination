# Variational Rectified Camera Refiner

本目录是 [VGGT 多短窗修复三文档](../VGGT_VRFM_DOCUMENT_SET.md)中的第三份：

> 用 Variational Rectified Flow 融合 VGGT 多短窗修复方向

它是一份 6--10 页目标的概念设计文档，用于组会、合作者沟通和方法早期评审，不是
网络层数、训练命令或数据工程的实现合同。

当前状态：`protocol in progress; no scientific conclusion yet`。只有配套 V0 前置实验
得到冻结的 `GO_VRFM`，才进入正式实现。若结果更支持 selector 或连续融合，应走对应
路线而不是强行使用 V-RFM。

目标读者是需要理解“如果 V0 通过，VGGT 与 V-RFM 怎样连接”的研究者。本文不承诺
具体网络宽度、训练命令或部署 scorer，也不把多个 local windows 直接当成多个训练
target。上游的概念边界见[理论基础](../camera_solution_space_01_theory_foundation/README.md)，
证据门控见[V0 前置实验](../camera_velocity_ambiguity_preexperiment/README.md)。

共享术语中，V-RFM 是 generator，不是 selector；sequence-level latent 在一条完整
500 帧序列上只采样一次，以避免逐帧切换修复模式。

## 构建

在仓库根目录运行：

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document method --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document method
```

第一版只生成 camera-center residual；rotation 和 FoV 保持 global VGGT 输出。

输出路径：`output/pdf/variational_rectified_camera_refiner_method.pdf`；编译缓存与逐页
渲染分别位于 `tmp/pdfs/method/` 和 `tmp/pdf-renders/method/`。
