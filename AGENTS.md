# Repository Guidelines

## Project Structure & Module Organization

`vggt/` 是保留的最小核心包：`vggt/models/` 放 VGGT 与 aggregator，`vggt/heads/` 放 camera/depth/point/track heads，`vggt/layers/` 放 Transformer 层，`vggt/utils/` 放图像加载、pose 与几何工具，`vggt/dependency/` 放 tracker/SfM 辅助代码。`probe/literature/` 保存 DoLa/DAMO 与 VGGT 幻觉定义的文献基线。`probe/h1_geometry_hallucination.py`、`probe/h2_multiframe_hallucination.py` 和 `probe/h3_aggregator_trace.py` 是当前验证脚本，`probe/tests/` 保存单元测试，`probe/runs/` 保存可再生成的实验结果。`examples/kitchen/images/` 是默认小样例输入；`ckpt/VGGT-1B/` 保存本地 checkpoint。

## Build, Test, and Development Commands

- `pip install -e .`：以可编辑模式安装本包。
- `pip install -r requirements.txt`：安装核心推理依赖。
- `python -c "from vggt.models.vggt import VGGT; print(VGGT.__name__)"`：快速检查核心包可导入。
- `Get-Content probe/literature/damo_dola_to_vggt.md`：查看当前幻觉研究的文献起点。
- `python -m unittest discover -s probe/tests`：运行 H1/H2 脚本的快速单元测试。
- `python probe/h1_geometry_hallucination.py --weights local --device cpu`：在当前环境用本地权重跑 H1 无几何输入实验。
- `python probe/h2_multiframe_hallucination.py --weights local --device cpu`：在当前环境用本地权重跑 H2 多帧输入实验。
- `python probe/h3_aggregator_trace.py --weights local --device cpu --max-aa-blocks 8`：追踪 aggregator 前 8 个 alternating-attention block 的跨帧 token 漂移。

## Coding Style & Naming Conventions

使用 Python 3.10+，遵循现有 PyTorch 代码风格：4 空格缩进，函数和变量用 `snake_case`，类用 `CamelCase`。涉及张量维度的函数应在 docstring 或注释中写清形状约定。导入顺序保持为标准库、第三方库、本地包。`pyproject.toml` 未配置统一格式化工具，修改时应贴近相邻代码，避免无关的大规模格式化。

## Testing Guidelines

测试目前使用 `unittest`。新增实验脚本时，应同步添加最小测试，优先覆盖纯函数、统计指标、路径解析和输出格式。H3 的指标以 input-to-final camera drift gain 为主，避免只看单层 attention delta。涉及完整模型推理的改动，应先提供随机权重或小输入的快速检查；使用本地权重或 CUDA 的验证结果应记录在变更说明中。

## Commit & Pull Request Guidelines

提交信息使用简短、明确的祈使句标题，例如 `Add token trace probe` 或 `Trim demo assets`。PR 或变更说明应描述实验目的、涉及的核心文件、运行过的测试命令，以及生成输出的位置。不要把大型输出直接纳入版本控制；需要复现实验时，保留脚本和参数即可。

## Security & Configuration Tips

不要提交私有数据集、本地 checkpoint、Hugging Face token，或大型生成文件。`ckpt/` 用于本地实验权重，移动或共享前先确认许可。新实验输出应放在独立目录，且能由脚本重新生成。
