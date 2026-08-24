# 3D Reconstruction Diffusion / VGGT Hallucination

这是一个围绕 VGGT 长短上下文相机重建、局部—全局一致性，以及相机速度
多解性的研究仓库。当前主线是 **Camera Velocity Ambiguity 02（CVA02）**，
它是一个 calibration-first 的 ScanNet-50 前置实验，不是训练脚本，也不会在
前置实验阶段训练 Diffusion、Flow Matching 或 V-RFM。

## 当前主线

主开发工作树是 `cva02_dev`，对应分支：

```text
codex/camera_velocity_ambiguity_02_pre_experiment
```

该分支从 `015-local-global-consistency` 的提交
`a85bcba9356be72d00f970e948ffc461f58c95e8` 继承稳定的相机推理和几何工具，
但使用独立的命名空间、协议、输出目录和结论边界。CVA02 要回答的是：

1. 相邻 100-frame local window 是否只提供同一方向的数值修正；
2. 是否只是窗口选择问题；
3. 两个局部解是否构成连续冗余；
4. 两端都有效但中间存在独立 RGB-D 观测能量障碍时，是否有证据支持多模态相机速度。

第 4 类必须有独立观测能量支持；仅靠 prediction-only 的凸对齐残差不能证明
中点势垒，更不能直接推出存在物理上断开的 IK 分支。

当前规范入口：

- [项目文档索引](docs/README.md)
- [CVA02 当前状态](docs/camera_velocity_ambiguity_02_status.md)
- [CVA02 冻结设计](docs/camera_velocity_ambiguity_02_design.md)
- [CVA02 执行计划](docs/superpowers/plans/2026-08-24-camera-velocity-ambiguity-02.md)

## 研究演进

仓库经历了以下几层工作，旧层仍保留作为可追溯证据：

1. Camera iteration、camera-head amplification 等早期诊断；
2. Round 2 local/global consistency：比较 500-frame global 与 100-frame local；
3. ScanNet-50 与 FastVGGT 评测管线审计；
4. CVA02：在既有代码基础上，重新冻结协议，区分连续冗余、窗口选择和多模态证据。

旧实验的数值结果不能直接替代 CVA02 的 50-scene 输入，也不能被描述为新实验
的 fresh holdout。历史文件的入口见 [`doc/README.md`](doc/README.md) 和
[`log/README.md`](log/README.md)。

## 当前运行边界

正式开发和计算只在 H20 上进行：

```text
代码根目录：/home/ubuntu/yjh/vggt
CVA02 工作树：/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment
ScanNet：/data/yjh/share/datasets/ScanNet
VGGT-1B：/data/yjh/share/pretrained/VGGT-1B/model.safetensors
输出：/data/output/camera_velocity_ambiguity/<run_id>/
```

本地 Windows 目录只保存代码、文档、下载暂存和小型审计材料；大型 H20 结果不
拉回本地。ScanNet 只有在 100/100 资产逐项完成官方清单、Content-Length、本地
文件和 H20 SHA-256 校验后，才允许启动 GPU 推理。

## 实验协议

- FastVGGT 官方 50-scene 顺序固定；
- 49 个普通场景各 500 帧，`scene0150_00` 使用全部 430 帧；
- 共 449 个 length-100、stride-50 local windows；
- 共 399 个相邻 local pairs，其中 primary shared-50 为 398，secondary shared-70 为 1；
- calibration 为 10 个场景，development evaluation 为 40 个场景；
- global-to-GT 只做一次 full-scene Sim(3)；local 左/右窗口只对齐到对应 global segment；
- FastVGGT pose 轨迹图严格复现上游，仅作为 presentation，不进入科学判定指标；
- 只有 calibration 完成并冻结阈值后，才允许进入 development evaluation；
- pre-experiment 不训练生成模型。

## 数据与运行顺序

1. 完成 ScanNet FastVGGT50 下载和完整性验证，生成 `verified_completion.json`；
2. 在 CPU 上通过 protocol、input gate、artifact schema 和几何单元测试；
3. 运行 calibration，冻结阈值和 provenance；
4. 通过门控后运行 40-scene development evaluation；
5. 生成 prediction-only、RGB-D evidence、FastVGGT reproduction plot 和最终报告；
6. 只有第 4 类证据跨场景稳定成立，才讨论后续 V-RFM/生成模型实验。

## 开发检查

在 H20 对当前代码使用对应的 `vggt` 环境运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q pre_experiments
```

下载、校验、历史实验和文档维护说明集中在
[文档索引](docs/README.md)，不要从旧 README 或旧日志推断当前状态。

## 下一步

当前最近的可执行任务是：完成 Task 1 的 CVA02 frozen protocol，然后依次实现
输入门控、frame/artifact schema、prediction runner、几何证据和 FastVGGT 严格
复现。当前进度只更新在 [CVA02 当前状态页](docs/camera_velocity_ambiguity_02_status.md)。
