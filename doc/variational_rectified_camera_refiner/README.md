# Variational Rectified Camera Refiner

本目录是 VGGT 多短窗修复三文档中的第三份：

> 用 Variational Rectified Flow 融合 VGGT 多短窗修复方向

它是一份 6--10 页目标的概念设计文档，用于组会、合作者沟通和方法早期评审，不是
网络层数、训练命令或数据工程的实现合同。

当前状态：`protocol in progress; no scientific conclusion yet`。只有配套 V0 前置实验
得到冻结的 `GO_VRFM`，才进入正式实现。若结果更支持 selector 或连续融合，应走对应
路线而不是强行使用 V-RFM。

## 构建

在仓库根目录运行：

```powershell
python scripts/docs/build_vggt_vrfm_pdfs.py --document method --render
python scripts/docs/check_vggt_vrfm_pdf_text.py --document method
```

第一版只生成 camera-center residual；rotation 和 FoV 保持 global VGGT 输出。

