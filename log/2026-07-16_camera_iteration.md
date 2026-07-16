# 2026-07-16 Camera Iteration Log

## 准备轮：分支与研究边界

- 创建 `camera-iteration-preexperiment` worktree，与现象观察实验分离。
- 将干净 VGGT 基线合入方法分支，移除现象评估代码、结果和下载脚本。
- 固定 10 个 ScanNet 场景、方法预实验目录、输出隔离规则及无下载 AutoDL 契约。
- 总研究指导和总实施计划统一迁回 `main` 维护，本工作树只保留分支复现设计；
  prediction 看 aligned、纯 GT 看 raw 的规则继续写入分支约束。

## 第 0 轮：Camera Head 可观测接口

- 为 `CameraHead` 增加可选逐轮 trace：normalized token、raw 9D pose、raw
  9D delta、delta norm，以及按需保存的 modulated token。
- 增加 `decode_pose_tokens()`，并将迭代次数和 trace 开关透传到
  `VGGT.forward()`。
- 用 CPU 测试确认默认 4 轮调用的返回结构和数值保持兼容，trace 关闭时不创建
  逐轮 trace 列表。

## 第 1 轮：正式 Camera Iteration 预实验代码

- 实现确定性 run ID、完整 Git commit 元数据、输出路径保护和原子 JSON 写入。
- 独立实现 Sim(3) 对齐后的 ATE/ARE/RPE、scale 诊断和逐轮 9D delta 统计。
- 实现 ScanNet raw GT pose/图像读取、坏 pose 跳过、数值帧排序和 nested
  uniform 帧选择。
- 实现本地 `model.safetensors`/`model.pt` 加载及 pose-only VGGT 推理。
- 每个场景/帧数组合只运行一次最大 Camera Head 迭代，输出 JSON、CSV、NPZ、
  精确帧 ID 和最后写入的 `complete.json`；重复调用可续跑。

## AutoDL 复现轮

- 新增 `scripts/autodl/run_camera_iteration.sh`，默认复用现有权重、数据和 conda
  环境；所有路径、帧数和迭代数均可由环境变量覆盖。
- 新增纯标准库 preflight，检查依赖、checkpoint 和 ScanNet processed/raw
  布局；runner 在构造模型前另行检查 PyTorch 与 CUDA。
- 新增分支内 `.sens` 提取器，只导出本实验需要的 color 与 raw GT pose；惰性
  读取帧数据，完整场景跳过，半成品继续提取。

## 本地验证

- 完成 34 个 CPU `unittest`，覆盖默认 API 兼容、trace、指标、CLI、续跑、
  preflight 和提取契约。
- 完成 Python 编译检查、CLI `--help`、临时目录 preflight、`git diff --check`
  和 `bash -n scripts/autodl/run_camera_iteration.sh`。
- 本机没有 VGGT-1B checkpoint、ScanNet 数据和 CUDA，因此今天没有执行真实
  checkpoint 推理，也没有生成或声称任何 Camera Iteration 实验结论。
