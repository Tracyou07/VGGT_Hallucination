# 现象刻画与方法预实验分离设计

**日期：** 2026-07-16

**状态：** 对话设计已确认，等待文档审阅

## 目标

将已有的幻觉现象观测、ScanNet 评估代码和有限样本结果，与即将开始的 Camera iteration、SE(3) refinement 和 latent probing 正式预实验明确分开。分类依据是研究目的，而不是数据规模、完成程度或代码新旧。

## 研究类型

### 现象刻画（Phenomenon Characterization）

用于回答“VGGT 出现了什么现象”：深度与 pose 的误差差异、误差随帧数增长的趋势、scale/gauge 表现和不同输出 Head 的关系。

- 状态为持续进行，不是归档内容。
- 当前样本量有限，已有结论必须标注为阶段性观察。
- 后续继续使用同一代码扩展场景数、帧数和数据集覆盖。
- 不用于证明 Camera refinement 新方法有效。

现有资产保持原路径：

```text
experiments/scannet_hallucination/
configs/scannet_hallucination_10.txt
scripts/autodl/
results/scannet_hallucination/
```

### 方法预实验（Method Pre-experiment）

用于回答“某个干预假设是否成立”：Camera Head 迭代次数是否影响误差、GT-free iteration selection 是否有效、显式 SE(3) 修正是否有效、Camera latent 是否存在可优化方向。

- 在运行前固定变量、指标、输出 schema 和通过条件。
- 不与现象刻画结果共用输出目录。
- 初始阶段仍是预实验，不称为主实验或最终方法结果。
- 只有通过阶段门后才进入下一方法阶段。

新增资产使用独立路径：

```text
pre_experiments/
  README.md
  camera_iteration/
    __init__.py
    metrics.py
    run_study.py

results/
  pre_experiments/
    camera_iteration/

probe/tests/
  test_camera_head_trace.py
  test_vggt_camera_options.py
  test_camera_iteration_metrics.py
```

## 边界规则

核心模型 `vggt/` 和通用测试不属于任何一种实验类型。方法预实验可以复用现有 ScanNet 的数据选择与 aligned pose 指标，但不得读取现象刻画结果作为算法输入，也不得把 GT 指标用于 iteration selection 或 refinement objective。

首轮允许从 `experiments.scannet_hallucination.run_eval` 复用稳定纯函数。只有当同一逻辑出现第三个调用方时，才单独计划提取到共享模块；本次分类不附带无关重构。

## 输出与元数据

现象刻画结果继续写入：

```text
results/scannet_hallucination/scene0000_00/frames_100/
```

其他运行保持同一命名规则：第一级是 ScanNet scene ID，第二级是实际帧数 `frames_数字`。

方法预实验结果只能写入：

```text
results/pre_experiments/camera_iteration/20260716T010203Z_2b1e6fc/
```

其中第三级目录固定为 UTC 时间 `YYYYMMDDTHHMMSSZ` 加下划线和 7 位 commit。

每次方法预实验生成 `run_metadata.json`，至少记录：

```json
{
  "study_type": "method_pre_experiment",
  "study_name": "camera_iteration",
  "git_commit": "2b1e6fc3a7e46e8bc4e628c4ce4f8e1a49373032",
  "scene_list": "configs/scannet_hallucination_10.txt",
  "scene_count": 10,
  "frame_counts": [25, 50, 100, 200, 500],
  "sampling": "nested_uniform",
  "primary_metric_policy": "prediction metrics use aligned values"
}
```

`run_id` 由脚本使用 UTC 时间和短 commit 自动生成，不由调用者手写。大型 token dump、checkpoint 和可再生成的中间张量不提交；可审计的配置、summary 和必要图表可以提交。

上面的 JSON 是完整 10-scene 协议示例。`scene_count`、`frame_counts`、`sampling` 和 `scene_list` 必须来自本次实际调用；例如 smoke run 必须记录 `scene_count: 1`，不能沿用完整协议的值。

## 文档与日志

- `experiments/scannet_hallucination/README.md` 标注其类型为“现象刻画”、状态为“持续扩充”，并声明当前样本限制。
- `results/scannet_hallucination/README.md` 说明现有结果不是大规模最终结论。
- `pre_experiments/README.md` 定义正式预实验的运行和结果约束。
- `AGENTS.md` 记录两类研究的定义和输出路径。
- `log/YYYY-MM-DD.md` 中相关条目分别使用 `[现象刻画]` 与 `[方法预实验]` 前缀。
- `doc/VGGT_DiT_Implementation_Plan.md` 中新增运行代码的路径改为 `pre_experiments/camera_iteration/`，结果路径改为 `results/pre_experiments/camera_iteration/`。

## 兼容性与失败保护

- 不移动、不重命名现有现象刻画代码、配置、脚本和结果。
- 现有 AutoDL 与 ScanNet 命令继续工作。
- 方法预实验 CLI 若收到 `results/scannet_hallucination/` 下的输出路径，应直接报错，防止结果混写。
- 恢复运行时必须校验 `run_metadata.json.study_type` 和 `study_name`，不匹配则拒绝复用目录。

## 验收标准

1. 新窗口读取 `AGENTS.md` 后能立即区分两类研究。
2. 旧 ScanNet 评估路径和历史结果保持不变，并明确标注样本限制。
3. Camera iteration 代码与输出均位于方法预实验专用目录。
4. 任何方法预实验输出都带有可追溯的 commit、数据选择和 metric policy。
5. 日志和论文结论不会把阶段性现象观察写成方法预实验结论，反之亦然。

## 非目标

本设计不扩大现象刻画数据量，不实现 Camera trace，不运行预实验，也不改变 aligned/raw 指标规则。这些工作分别由现有 ScanNet 流程和后续实施计划负责。
