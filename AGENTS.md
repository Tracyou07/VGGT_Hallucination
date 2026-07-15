# Repository Guidelines

## Project Structure & Module Organization

`vggt/` 是核心包：`models/` 放 VGGT 与 Aggregator，`heads/` 放 Camera、Depth、Point 和 Track Heads，`layers/` 放 Transformer 组件，`utils/` 放 pose、图像和几何工具。`experiments/scannet_hallucination/` 是 ScanNet 评估入口，`probe/tests/` 保存快速单元测试，`probe/runs/` 保存可再生成的研究输出。`configs/` 放场景列表，`scripts/autodl/` 放远端环境脚本，`results/` 保存已提交的基线结果。研究设计与实施计划放在 `doc/`，每日工作记录放在 `log/YYYY-MM-DD.md`。

## Build, Test, and Development Commands

- `pip install -e .`：以可编辑模式安装本包。
- `pip install -r requirements.txt`：安装核心推理依赖。
- `python -c "from vggt.models.vggt import VGGT; print(VGGT.__name__)"`：检查核心包导入。
- `python -m unittest discover -s probe/tests`：运行全部快速测试。
- `python -m experiments.scannet_hallucination.run_eval --help`：查看 ScanNet 评估参数；正式运行需要本地数据、权重和输出目录。
- `Get-Content doc/VGGT_DiT_Implementation_Plan.md`：查看当前分轮实施计划。

## Coding Style & Naming Conventions

使用 Python 3.10+ 和现有 PyTorch 风格：4 空格缩进，函数与变量使用 `snake_case`，类使用 `CamelCase`。涉及张量的接口必须注明形状和坐标约定。导入顺序为标准库、第三方、本地包。仓库未配置统一 formatter；保持邻近代码风格，避免无关格式化和重构。

## Testing Guidelines

测试框架为 `unittest`。新增模型接口或 probe 时，先覆盖纯函数、shape、参数透传、输出 schema 和默认兼容性。快速测试不得依赖 CUDA、下载 checkpoint 或读取 ScanNet。完整权重实验需在变更说明和 `log/` 中记录命令、数据选择与输出路径。

## Metric Interpretation Rules

凡指标含 VGGT 预测量，主结论使用 `aligned`，例如 `depth_absrel_aligned`、`pose_ate_rmse_aligned` 及含预测 depth/pose/point 的 Chamfer；对应 `raw` 与 scale 仅诊断尺度漂移或 gauge mismatch。纯 GT baseline 只看 `raw`。混合项只要含预测量，仍按“aligned 为主，raw/scale 为辅”处理。

## Research Documentation

`doc/VGGT_DiT_Research_Guide.md` 定义研究边界，`doc/VGGT_DiT_Implementation_Plan.md` 定义可执行任务与阶段门。修改研究方向时同步更新相关文档；`log/` 只记录实际完成事项，并明确区分计划、实验结果和未实施工作。

## Commit & Pull Request Guidelines

提交标题使用简短祈使句，如 `Add camera iteration probe`。PR 或变更说明应包含目的、核心文件、验证命令和输出位置。不要提交私有数据集、checkpoint、访问令牌或不可复现的大型输出。
