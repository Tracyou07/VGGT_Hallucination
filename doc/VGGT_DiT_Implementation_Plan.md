# VGGT Camera Iteration Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结原始 VGGT 的前提下，使 Camera Head 的每轮更新可观测、可配置、可评估，并完成 `num_iterations = 1, 2, 4, 8, 16` 的 training-free 基线实验。

**Architecture:** 保留 Aggregator、Camera Head 和现有 Heads 的默认计算路径。只在 Camera Head 增加可选轻量 trace，在 `VGGT.forward()` 透传迭代参数，并新增独立 probe 复用现有 ScanNet 数据读取与 pose 指标；单次前向运行到最大迭代数，再评估指定中间轮次，避免重复执行 Aggregator。

**Tech Stack:** Python 3.10+、PyTorch 2.3、NumPy、现有 `unittest`、现有 ScanNet evaluation helpers。

## Two Kinds of Iterations

本文中的“轮”有两种含义，实施和分析时必须分开记录。

### Camera Head 内部第 k 轮做什么

Aggregator 只运行一次，并产生固定的 normalized Camera Token `z[B,S,C]`。Camera Head 的每一轮都复用同一个 `z`，变化的是当前 raw pose state：

1. 第 1 轮没有历史 pose，使用 learned `empty_pose_tokens`；第 2 轮起使用上一轮累计的 raw pose，并保持原实现的 detach 行为。
2. 将当前 pose state 投影到 `C` 维，生成 AdaLN 的 `shift`、`scale`、`gate`。
3. 用这些条件调制固定 Camera Token，得到本轮 `pose_tokens_modulated_k`。
4. 通过 Camera Head 的小型 Transformer trunk 和 pose branch，预测 raw 9D 增量 `pose_delta_k`。
5. 累加得到激活前状态 `raw_pose_enc_k = raw_pose_enc_(k-1) + pose_delta_k`；第 1 轮直接令二者相等。
6. 对 translation、quaternion 和 FoV 做 `activate_pose()`，得到可用于相机转换与评估的 `pose_enc_k`。

因此单轮 trace 必须区分四个量：`pose_tokens_modulated_k` 是特征，`pose_delta_k` 是本轮更新，`raw_pose_enc_k` 是激活前累计状态，`pose_enc_k` 是激活后的相机输出。`delta_norm_k` 只是 raw 9D 增量的 L2 统计，不能解释为 SE(3) 距离。

### 研究实施各轮做什么

#### 第 0 轮：建立观测接口

只改接口，不做研究结论。加入可选 trace、normalized token 解码入口和顶层迭代参数透传；用 CPU 单元测试证明默认输出未变化。对应 Task 1–2。

#### 第 1 轮：Camera Iteration Study

冻结全部权重，在每个帧数设置上只运行一次 Aggregator 和 16 轮 Camera Head，然后分别读取第 1、2、4、8、16 轮结果。比较 aligned ATE/ARE/RPE、Sim(3) scale 和 delta norm，回答默认 4 轮是否不足、过度或与主要误差无关。对应 Task 3–6。

#### 第 2 轮：Geometry-aware Iteration Selection

不改变任何 pose，只从第 1 轮已经产生的候选 pose 中选择一轮。选择分数只能使用预测 depth、point map、confidence、时序平滑、focal 稳定和更新幅度，不能使用 GT；GT 只在选择完成后评价。若没有场景依赖的最佳轮次，本轮不启动。

#### 第 3 轮：Training-free SE(3) Pose Refinement

固定 VGGT 的 depth、point map、confidence 和初始 pose，只优化每帧 6D `delta_xi`，首帧固定用于消除 gauge。每一步重新计算几何 residual，并在 SE(3) 上合法更新相机。本轮回答“表征是否基本足够，只是 Camera Head 缺少闭环几何反馈”。

#### 第 4 轮：Low-rank Camera Latent Probing

冻结 VGGT 和 Camera Head，固定低秩基 `B`，只优化每帧低维系数 `a`，再由原 Camera Head 解码 `z + aB^T`。本轮不训练 denoiser，只回答 Camera Token 邻域是否存在能由 GT-free 几何目标找到的更优方向。

#### 第 5 轮：Tiny Latent Denoiser SFT

只有第 4 轮跨场景、跨随机基稳定有效后才训练。冻结所有 VGGT 模块，只训练 zero-init、one-step 的小型 latent residual adapter，并使用 clean/degraded context pair。该轮回答“低秩优化轨迹能否被一个数百万参数以内模块泛化”。

#### 第 6 轮：Latent 与 Pose 两阶段组合

只有第 3 轮和第 5 轮分别独立有效且收益互补时才组合：先更新 latent，经冻结 Camera Head 解码，再做一次 SE(3) correction。该轮只验证组合增益，不再同时改变两个子模块的设计。

## Global Constraints

- 研究方法依据是 `doc/VGGT_DiT_Research_Guide.md`，实验分类依据是 `doc/2026-07-16_Experiment_Separation_Design.md`。
- 首轮 MVP 完全 training-free；不得创建 optimizer，不得修改 checkpoint，不得更新任何模型参数。
- `CameraHead.forward(aggregated_tokens_list)` 与 `VGGT.forward(images, query_points)` 的默认数值结果和现有 prediction keys 必须保持不变。
- 高维 `pose_tokens_modulated` 只在显式请求时保存；默认 trace 仅保留 Camera Token、9 维 pose 累计量、9 维增量和增量范数。
- 所有含预测量的指标以 `aligned` 为主结论；`raw` 与 scale 仅诊断尺度或 gauge；纯 GT baseline 只看 `raw`。
- 单元测试不得依赖 CUDA、下载权重或读取 ScanNet；正式预实验输出只能放在 `results/pre_experiments/camera_iteration/`，不得提交大型 `.npz` 文件。
- 不增加第三方依赖，不重构无关 Head，不在首轮实现 geometry selection、SE(3) 优化、latent optimization 或 SFT。
- `experiments/scannet_hallucination/` 与 `results/scannet_hallucination/` 属于持续扩充的现象刻画，不移动、不重命名，也不接收方法预实验输出。
- 方法预实验必须生成 `run_metadata.json`，其中记录真实调用参数和完整 Git commit；不得把 smoke run 写成完整 10-scene 协议。

---

## Current Baseline

实施前必须按真实代码而非抽象图理解基线：

- `vggt/heads/camera_head.py:73` 已支持 `num_iterations`，但没有返回 raw pose delta、激活前累计 pose 或 token trace。
- `vggt/models/vggt.py:29` 尚未向顶层调用者暴露 `camera_num_iterations`。
- `VGGT.forward()` 已经返回 `pose_enc_list`，因此不得再创建含义重复的字段。
- `experiments/scannet_hallucination/run_eval.py:178` 已实现 `evaluate_pose()`，输出本研究需要的 aligned ATE、ARE、RPE 和 Sim(3) scale；新 probe 应直接复用它。

## File Map

| Path | Responsibility |
|---|---|
| `pre_experiments/README.md` | 定义方法预实验的目录、运行和输出约束 |
| `pre_experiments/camera_iteration/contracts.py` | 验证输出路径、生成 run ID 与 metadata |
| `pre_experiments/camera_iteration/metrics.py` | 纯函数：验证轮次、转换 pose、生成逐轮指标行 |
| `pre_experiments/camera_iteration/run_study.py` | ScanNet CLI、单次最大轮次推理、结果落盘 |
| `vggt/heads/camera_head.py` | 定义 Camera trace contract，并允许直接解码 normalized Camera Tokens |
| `vggt/models/vggt.py` | 透传迭代次数与 trace 开关，保持默认输出兼容 |
| `probe/tests/test_camera_head_trace.py` | CameraHead trace 形状、数值兼容和输入校验 |
| `probe/tests/test_vggt_camera_options.py` | 顶层参数透传和 prediction contract |
| `probe/tests/test_pre_experiment_contract.py` | 方法预实验输出隔离和 metadata contract |
| `probe/tests/test_camera_iteration_metrics.py` | 逐轮指标与轮次索引纯函数测试 |
| `pre_experiments/camera_iteration/README.md` | 运行命令、输出 schema、指标解释 |

---

## Preparation: Separate Study Types

这一准备任务只建立现象刻画与方法预实验的可见边界，不移动旧代码和结果。

### Task 0: Add Study Labels and Pre-experiment Namespace

**Files:**

- Create: `experiments/scannet_hallucination/README.md`
- Create: `results/scannet_hallucination/README.md`
- Create: `pre_experiments/__init__.py`
- Create: `pre_experiments/README.md`
- Create: `pre_experiments/camera_iteration/__init__.py`
- Create: `pre_experiments/camera_iteration/README.md`
- Modify: `AGENTS.md`
- Create: `log/2026-07-16.md`

**Interfaces:**

- Existing ScanNet commands and result paths remain unchanged.
- New code imports through `pre_experiments.camera_iteration`.
- Log entries use `[现象刻画]` or `[方法预实验]` prefixes.

- [ ] **Step 1: Add the active phenomenon-characterization labels**

Both existing README files must state `study_type: phenomenon_characterization`, `status: ongoing`, the current limited-sample caveat, and that future large-scale runs continue in the same paths.

- [ ] **Step 2: Create the method pre-experiment namespace**

`pre_experiments/README.md` must define `study_type: method_pre_experiment`, the allowed result root `results/pre_experiments/`, the aligned/raw metric policy, and the rule that GT is evaluation-only for selection/refinement methods. Package `__init__.py` files contain only a one-line module docstring.

- [ ] **Step 3: Update repository instructions and today's log**

Add a `Study Types` section to `AGENTS.md` with both definitions and paths. In `log/2026-07-16.md`, record this work as `[方法预实验]` and explicitly state that no Camera experiment has run yet.

- [ ] **Step 4: Verify the namespace without moving old assets**

```powershell
Test-Path experiments\scannet_hallucination\run_eval.py
Test-Path results\scannet_hallucination\summary.json
Test-Path pre_experiments\camera_iteration\__init__.py
git diff --check
```

Expected: all three `Test-Path` calls print `True`; `git diff --check` exits 0.

- [ ] **Step 5: Commit the study boundary**

```powershell
git add AGENTS.md experiments/scannet_hallucination/README.md results/scannet_hallucination/README.md pre_experiments log/2026-07-16.md doc/VGGT_DiT_Implementation_Plan.md
git commit -m "Separate method pre-experiments"
```

---

## Round 0: Observation Interface

本轮交付可观测、可测试且默认兼容的 Camera API。它不运行 ScanNet 主实验，也不改变模型预测策略。

### Task 1: Define the Camera Head Trace Contract

**Files:**

- Modify: `vggt/heads/camera_head.py:73`
- Test: `probe/tests/test_camera_head_trace.py`

**Interfaces:**

- Consumes: final aggregated token tensor `[B, S, P, C]`。
- Produces: default `list[Tensor[B,S,9]]`，或 `(pose_enc_list, CameraTrace)`。
- Exposes: `decode_pose_tokens(normalized_pose_tokens, num_iterations, return_trace, trace_pose_tokens)`，供后续低秩 latent probe 复用。

- [ ] **Step 1: Write the failing CameraHead tests**

Create `probe/tests/test_camera_head_trace.py` with three cases:

```python
import unittest

import torch

from vggt.heads.camera_head import CameraHead


class CameraHeadTraceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = CameraHead(
            dim_in=32,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        self.tokens = [torch.randn(2, 3, 5, 32)]

    def test_default_return_is_unchanged(self):
        with torch.no_grad():
            baseline = self.head(self.tokens, num_iterations=3)
            traced, trace = self.head(
                self.tokens,
                num_iterations=3,
                return_trace=True,
            )

        self.assertIsInstance(baseline, list)
        self.assertEqual(len(baseline), 3)
        for expected, actual in zip(baseline, traced):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(trace["delta_norm"].shape, (3, 2, 3))
        self.assertEqual(trace["normalized_camera_tokens"].shape, (2, 3, 32))
        self.assertEqual(trace["pose_tokens_modulated_list"], [])

    def test_full_trace_matches_pose_delta_norms(self):
        with torch.no_grad():
            poses, trace = self.head(
                self.tokens,
                num_iterations=2,
                return_trace=True,
                trace_pose_tokens=True,
            )

        self.assertEqual(len(poses), 2)
        self.assertEqual(len(trace["raw_pose_enc_list"]), 2)
        self.assertEqual(len(trace["pose_delta_list"]), 2)
        self.assertEqual(len(trace["pose_tokens_modulated_list"]), 2)
        expected = torch.stack(
            [delta.float().norm(dim=-1) for delta in trace["pose_delta_list"]]
        )
        torch.testing.assert_close(trace["delta_norm"], expected)

    def test_invalid_trace_options_raise(self):
        with self.assertRaisesRegex(ValueError, "num_iterations"):
            self.head(self.tokens, num_iterations=0)
        with self.assertRaisesRegex(ValueError, "return_trace"):
            self.head(self.tokens, trace_pose_tokens=True)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and confirm the contract is missing**

Run:

```powershell
python -m unittest probe.tests.test_camera_head_trace -v
```

Expected: FAIL because `CameraHead.forward()` does not accept `return_trace` or `trace_pose_tokens`.

- [ ] **Step 3: Add the typed trace and token decoder**

Add `TypedDict` and replace the current `forward()` / `trunk_fn()` boundary with this contract:

```python
from typing import TypedDict


class CameraTrace(TypedDict):
    normalized_camera_tokens: torch.Tensor
    raw_pose_enc_list: list[torch.Tensor]
    pose_delta_list: list[torch.Tensor]
    delta_norm: torch.Tensor
    pose_tokens_modulated_list: list[torch.Tensor]
```

```python
def forward(
    self,
    aggregated_tokens_list: list,
    num_iterations: int = 4,
    return_trace: bool = False,
    trace_pose_tokens: bool = False,
):
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    if trace_pose_tokens and not return_trace:
        raise ValueError("trace_pose_tokens requires return_trace=True")

    tokens = aggregated_tokens_list[-1]
    normalized_pose_tokens = self.token_norm(tokens[:, :, 0])
    return self.decode_pose_tokens(
        normalized_pose_tokens,
        num_iterations=num_iterations,
        return_trace=return_trace,
        trace_pose_tokens=trace_pose_tokens,
    )

def decode_pose_tokens(
    self,
    normalized_pose_tokens: torch.Tensor,
    num_iterations: int = 4,
    return_trace: bool = False,
    trace_pose_tokens: bool = False,
):
    if normalized_pose_tokens.ndim != 3:
        raise ValueError("normalized_pose_tokens must have shape [B, S, C]")
    return self.trunk_fn(
        normalized_pose_tokens,
        num_iterations=num_iterations,
        return_trace=return_trace,
        trace_pose_tokens=trace_pose_tokens,
    )
```

Inside `trunk_fn()`, preserve the existing detach and update order exactly. Append `pred_pose_enc_delta`, the raw cumulative `pred_pose_enc`, and optional `pose_tokens_modulated`; return the existing list alone unless trace was requested:

```python
trace: CameraTrace = {
    "normalized_camera_tokens": pose_tokens,
    "raw_pose_enc_list": raw_pose_enc_list,
    "pose_delta_list": pose_delta_list,
    "delta_norm": torch.stack(delta_norm_list, dim=0),
    "pose_tokens_modulated_list": pose_tokens_modulated_list,
}
return (pred_pose_enc_list, trace) if return_trace else pred_pose_enc_list
```

Do not call `.cpu()`, `.numpy()` or `.detach()` when storing trace fields; the optional trace must remain usable by later differentiable latent probing. Memory control is provided by `return_trace` and `trace_pose_tokens`.

- [ ] **Step 4: Run the focused test**

Run:

```powershell
python -m unittest probe.tests.test_camera_head_trace -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the independently tested contract**

```powershell
git add vggt/heads/camera_head.py probe/tests/test_camera_head_trace.py
git commit -m "Expose camera head iteration trace"
```

---

### Task 2: Expose Iteration Controls Through VGGT.forward

**Files:**

- Modify: `vggt/models/vggt.py:29`
- Test: `probe/tests/test_vggt_camera_options.py`

**Interfaces:**

- Consumes: existing `images` and optional `query_points` plus three keyword-only camera options。
- Produces: existing predictions unchanged by default; adds `predictions["camera_trace"]` only when requested。

- [ ] **Step 1: Write a lightweight forwarding test with fake modules**

The test must not instantiate the 1B model. Build a `VGGT` shell with `nn.Module.__init__()`, a fake aggregator, and a fake camera head. Verify both calls:

```python
default = model(images)
traced = model(
    images,
    camera_num_iterations=8,
    return_camera_trace=True,
    camera_trace_pose_tokens=True,
)
```

Assertions:

```python
self.assertEqual(fake_camera.last_options, (8, True, True))
self.assertEqual(len(traced["pose_enc_list"]), 8)
self.assertIn("camera_trace", traced)
self.assertNotIn("camera_trace", default)
self.assertEqual(default["pose_enc"].shape[-1], 9)
```

- [ ] **Step 2: Run the test and confirm top-level options are absent**

```powershell
python -m unittest probe.tests.test_vggt_camera_options -v
```

Expected: FAIL with an unexpected keyword argument for `camera_num_iterations`.

- [ ] **Step 3: Extend the forward signature without changing defaults**

Use this signature and call contract:

```python
def forward(
    self,
    images: torch.Tensor,
    query_points: torch.Tensor = None,
    *,
    camera_num_iterations: int = 4,
    return_camera_trace: bool = False,
    camera_trace_pose_tokens: bool = False,
):
```

```python
camera_output = self.camera_head(
    aggregated_tokens_list,
    num_iterations=camera_num_iterations,
    return_trace=return_camera_trace,
    trace_pose_tokens=camera_trace_pose_tokens,
)
if return_camera_trace:
    pose_enc_list, camera_trace = camera_output
    predictions["camera_trace"] = camera_trace
else:
    pose_enc_list = camera_output
predictions["pose_enc"] = pose_enc_list[-1]
predictions["pose_enc_list"] = pose_enc_list
```

Document all three options and the conditional trace key in the existing docstring. Keep the `*` so future camera controls cannot be confused with positional `query_points`.

- [ ] **Step 4: Run both camera API test modules**

```powershell
python -m unittest probe.tests.test_camera_head_trace probe.tests.test_vggt_camera_options -v
```

Expected: all tests PASS and no checkpoint is loaded.

- [ ] **Step 5: Commit the top-level API change**

```powershell
git add vggt/models/vggt.py probe/tests/test_vggt_camera_options.py
git commit -m "Expose camera iteration controls"
```

---

## Round 1: Camera Iteration Study

本轮只比较现有 Camera Head 的中间输出，不选择 pose、不优化 pose，也不更新 latent。

### Task 3: Enforce the Method Pre-experiment Run Contract

**Files:**

- Create: `pre_experiments/camera_iteration/contracts.py`
- Test: `probe/tests/test_pre_experiment_contract.py`

**Interfaces:**

- `validate_output_root(path: Path, repo_root: Path) -> Path`
- `make_run_id(git_commit: str, now: datetime) -> str`
- `build_run_metadata(git_commit: str, scene_list: Path, scene_count: int, frame_counts: list[int], sampling: str) -> dict[str, object]`
- `read_git_commit(repo_root: Path) -> str`

- [ ] **Step 1: Write failing contract tests**

Create tests that accept `results/pre_experiments/camera_iteration`, reject `results/scannet_hallucination`, verify `20260716T010203Z_2b1e6fc`, and assert metadata uses the actual scene count and frame counts supplied by the caller:

```python
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_iteration.contracts import (
    build_run_metadata,
    make_run_id,
    validate_output_root,
)


class PreExperimentContractTest(unittest.TestCase):
    def test_output_root_isolated_from_phenomenon_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "results" / "pre_experiments" / "camera_iteration"
            self.assertEqual(validate_output_root(allowed, root), allowed.resolve())
            with self.assertRaisesRegex(ValueError, "method pre-experiment"):
                validate_output_root(root / "results" / "scannet_hallucination", root)

    def test_run_id_and_metadata_use_actual_invocation(self):
        commit = "2b1e6fc3a7e46e8bc4e628c4ce4f8e1a49373032"
        now = datetime(2026, 7, 16, 1, 2, 3, tzinfo=timezone.utc)
        self.assertEqual(make_run_id(commit, now), "20260716T010203Z_2b1e6fc")
        metadata = build_run_metadata(
            commit,
            Path("configs/scannet_hallucination_10.txt"),
            scene_count=1,
            frame_counts=[25],
            sampling="nested_uniform",
        )
        self.assertEqual(metadata["study_type"], "method_pre_experiment")
        self.assertEqual(metadata["scene_count"], 1)
        self.assertEqual(metadata["frame_counts"], [25])
```

- [ ] **Step 2: Run the contract test and confirm the module is absent**

```powershell
python -m unittest probe.tests.test_pre_experiment_contract -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimal contract module**

Use `Path.resolve()` and `Path.is_relative_to()` for routing, UTC for run IDs, and `git rev-parse HEAD` for the full commit. Validate that commit IDs contain exactly 40 hexadecimal characters and that `scene_count == 0` is rejected.

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess


STUDY_TYPE = "method_pre_experiment"
STUDY_NAME = "camera_iteration"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_output_root(path: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    allowed = (root / "results" / "pre_experiments" / STUDY_NAME).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"method pre-experiment output must be under {allowed}")
    return resolved


def _validate_commit(git_commit: str) -> None:
    if _COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a 40-character lowercase hexadecimal id")


def make_run_id(git_commit: str, now: datetime) -> str:
    _validate_commit(git_commit)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{git_commit[:7]}"


def build_run_metadata(
    git_commit: str,
    scene_list: Path,
    scene_count: int,
    frame_counts: list[int],
    sampling: str,
) -> dict[str, object]:
    _validate_commit(git_commit)
    if scene_count < 1:
        raise ValueError("scene_count must be at least 1")
    if not frame_counts or any(count < 1 for count in frame_counts):
        raise ValueError("frame_counts must contain positive integers")
    return {
        "study_type": STUDY_TYPE,
        "study_name": STUDY_NAME,
        "git_commit": git_commit,
        "scene_list": scene_list.as_posix(),
        "scene_count": scene_count,
        "frame_counts": list(frame_counts),
        "sampling": sampling,
        "primary_metric_policy": "prediction metrics use aligned values",
    }


def read_git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    _validate_commit(commit)
    return commit
```

- [ ] **Step 4: Run the contract tests**

```powershell
python -m unittest probe.tests.test_pre_experiment_contract -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit the run contract**

```powershell
git add pre_experiments/camera_iteration/contracts.py probe/tests/test_pre_experiment_contract.py
git commit -m "Add pre-experiment run contract"
```

---

### Task 4: Add Pure Per-Iteration Metric Helpers

**Files:**

- Create: `pre_experiments/camera_iteration/metrics.py`
- Test: `probe/tests/test_camera_iteration_metrics.py`
- Reuse: `experiments/scannet_hallucination/run_eval.py:178`

**Interfaces:**

- `validate_iterations(iterations: list[int], available: int) -> list[int]`
- `build_iteration_rows(scene: str, frame_count: int, requested_iterations: list[int], pose_enc_list: list[torch.Tensor], delta_norm: torch.Tensor, gt_c2w: np.ndarray, image_hw: tuple[int, int]) -> list[dict[str, float | str]]`
- Each output row contains scene, frame count, iteration, aligned pose metrics, Sim(3) scale, and delta mean/p95/max。

- [ ] **Step 1: Write metric helper tests using synthetic cameras**

Construct three GT `c2w` poses translated along x, convert their inverse matrices with `extri_intri_to_pose_encoding()`, and reuse the same exact pose for iterations 1 and 4. Assert:

```python
self.assertEqual([row["iteration"] for row in rows], [1.0, 4.0])
self.assertLess(rows[0]["pose_ate_rmse_aligned"], 1e-5)
self.assertAlmostEqual(rows[1]["delta_norm_mean"], 4.0)
self.assertAlmostEqual(rows[1]["delta_norm_max"], 4.0)
```

Also assert that `[0, 4]`, duplicate iterations, and an iteration larger than `available` raise `ValueError`.

- [ ] **Step 2: Run the test and confirm the helper module is absent**

```powershell
python -m unittest probe.tests.test_camera_iteration_metrics -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the pure helper**

The implementation must reuse the repository's pose conversion and metric code:

```python
from experiments.scannet_hallucination.run_eval import evaluate_pose, to_homogeneous
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
```

For each requested one-based iteration `k`:

```python
pose_enc = pose_enc_list[k - 1]
extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, image_hw)
pred_w2c = to_homogeneous(extrinsic[0].detach().float().cpu().numpy())
delta = delta_norm[k - 1].detach().float().cpu().numpy().reshape(-1)
row = {
    "scene": scene,
    "frame_count_actual": float(frame_count),
    "iteration": float(k),
    "delta_norm_mean": float(delta.mean()),
    "delta_norm_p95": float(np.quantile(delta, 0.95)),
    "delta_norm_max": float(delta.max()),
}
row.update(evaluate_pose(pred_w2c, gt_c2w))
```

Do not compute a new pose alignment implementation in the probe. A single shared `evaluate_pose()` prevents metric drift from the existing ScanNet results.

- [ ] **Step 4: Run the helper tests and the existing suite**

```powershell
python -m unittest probe.tests.test_camera_iteration_metrics -v
python -m unittest discover -s probe/tests
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the metric unit**

```powershell
git add pre_experiments/camera_iteration/metrics.py probe/tests/test_camera_iteration_metrics.py
git commit -m "Add camera iteration metrics"
```

---

### Task 5: Build the ScanNet Camera Iteration Study

**Files:**

- Create: `pre_experiments/camera_iteration/run_study.py`
- Reuse: `experiments/scannet_hallucination/run_eval.py`

**Interfaces:**

- CLI defaults: `--iterations 1 2 4 8 16`, `--sampling nested_uniform`, `--weights local`, `--device auto`。
- One model call per scene/frame selection with `camera_num_iterations=max(iterations)`。
- Dense and track heads are disabled for this pose-only study after checkpoint loading。

- [ ] **Step 1: Define an argument parser that is testable without launching inference**

Use `parse_args(argv: list[str] | None = None)` and include:

```text
--data-dir            required ScanNet process_scannet root
--scene-list          required text file
--scene-limit         default 10
--frame-counts        default 25 50 100 200 500
--iterations          default 1 2 4 8 16
--sampling            prefix|uniform|nested_uniform|regime_step
--weights             local|hub|random
--ckpt-dir            default ckpt/VGGT-1B
--device              cuda|cpu|auto
--preprocess-mode     pad|crop
--out-dir             default results/pre_experiments/camera_iteration
--seed                default 33
--save-camera-tokens  disabled by default
```

Validate iterations immediately with `validate_iterations(iterations, max(iterations))`, then call `validate_output_root(args.out_dir, ROOT)` before loading a model. Generate the run directory with `make_run_id(read_git_commit(ROOT), datetime.now(timezone.utc))`.

- [ ] **Step 2: Implement one-pass inference**

Reuse `load_model()`, `resolve_device()`, frame selection helpers, and `load_and_preprocess_images()`. The core call is:

```python
with torch.no_grad():
    with torch.cuda.amp.autocast(
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        predictions = model(
            images.to(device),
            camera_num_iterations=max(iterations),
            return_camera_trace=True,
            camera_trace_pose_tokens=save_camera_tokens,
        )
```

Immediately after loading weights:

```python
model.depth_head = None
model.point_head = None
model.track_head = None
model = model.to(device).eval()
```

This preserves the exact Aggregator and Camera Head path while avoiding unrelated dense decoding cost.

- [ ] **Step 3: Write stable output artifacts**

At the run root, write `run_metadata.json` before inference. For example, scene `scene0000_00` with 25 frames writes into `scene0000_00/frames_25/` below that run root:

| File | Required content |
|---|---|
| `iteration_metrics.json` | One row per requested iteration |
| `iteration_metrics.csv` | Same rows with stable columns |
| `camera_trace.npz` | `frame_ids`, all `pose_enc_by_iteration[K,S,9]`, `raw_pose_enc_by_iteration[K,S,9]`, `pose_delta_by_iteration[K,S,9]`, `delta_norm[K,S]` |
| `selected_frame_ids.json` | Exact ordered frame IDs |

When `--save-camera-tokens` is set, append `normalized_camera_tokens[S,C]` and `pose_tokens_modulated[K,S,C]` to the NPZ. At the run root, write `summary.csv` and `summary.json` containing all scenes, frame counts, and requested iterations. The metadata values must come from the actual CLI invocation, including a smoke run's reduced scene count.

- [ ] **Step 4: Verify the CLI without model weights**

```powershell
python -m pre_experiments.camera_iteration.run_study --help
python -m py_compile pre_experiments\camera_iteration\run_study.py pre_experiments\camera_iteration\metrics.py pre_experiments\camera_iteration\contracts.py
```

Expected: exit code 0, all documented flags appear, and both files compile.

- [ ] **Step 5: Run a one-scene checkpoint smoke experiment**

On the machine containing the local checkpoint and ScanNet subset:

```powershell
python -m pre_experiments.camera_iteration.run_study `
  --data-dir datasets\scannetv2\process_scannet `
  --scene-list configs\scannet_hallucination_10.txt `
  --scene-limit 1 `
  --frame-counts 25 `
  --iterations 1 2 4 8 16 `
  --sampling nested_uniform `
  --weights local `
  --device cuda `
  --out-dir results\pre_experiments\camera_iteration
```

Expected: a new UTC/commit run directory is created, `run_metadata.json` records `scene_count=1`, exactly 5 metric rows are written, `camera_trace.npz` has `K=16`, every metric is finite, and iteration 4 matches the existing default pose metrics within floating-point tolerance for the same selected frames.

- [ ] **Step 6: Commit the probe**

```powershell
git add pre_experiments/camera_iteration/run_study.py
git commit -m "Add camera iteration study probe"
```

---

### Task 6: Document and Run the MVP Regression Gate

**Files:**

- Modify: `pre_experiments/camera_iteration/README.md`
- Verify: all modified and newly created files above

- [ ] **Step 1: Document reproducibility and metric semantics**

The README must state:

- The probe runs the Aggregator once at the maximum requested iteration.
- Iteration numbers are one-based and refer to `pose_enc_list[k - 1]`.
- `delta_norm` is the L2 norm of raw 9D `pred_pose_enc_delta`, not an SE(3) distance.
- GT pose is used only for final evaluation in this MVP; future geometry-aware selection may not use GT.
- Primary conclusions use aligned ATE/ARE/RPE. `pose_sim3_scale` is diagnostic.
- Camera token dumps are opt-in because their size is `O(KSC)`.
- Every run writes under `results/pre_experiments/camera_iteration/` and records actual invocation metadata.
- The existing `results/scannet_hallucination/` tree is ongoing phenomenon characterization and must never receive method outputs.

- [ ] **Step 2: Run the complete fast verification set**

```powershell
python -m unittest discover -s probe/tests
python -c "from vggt.models.vggt import VGGT; from vggt.heads.camera_head import CameraHead; print(VGGT.__name__, CameraHead.__name__)"
python -m py_compile vggt\heads\camera_head.py vggt\models\vggt.py pre_experiments\camera_iteration\contracts.py pre_experiments\camera_iteration\metrics.py pre_experiments\camera_iteration\run_study.py
```

Expected: all unit tests PASS, import prints `VGGT CameraHead`, and compilation exits 0.

- [ ] **Step 3: Inspect the final diff for compatibility**

```powershell
git diff --check
git diff --stat
git status --short
```

Expected: no whitespace errors; only the files listed in this plan are changed. Preserve the user's existing `AGENTS.md` modification.

- [ ] **Step 4: Commit documentation after verification**

```powershell
git add pre_experiments/camera_iteration/README.md
git commit -m "Document camera iteration experiment"
```

---

## Experiment Protocol

Run the full study on the same 10 ScanNet scenes and nested frame selections for `S = 25, 50, 100, 200, 500`. Report per-scene values and scene medians; do not average all frames across scenes into one unweighted pool.

For each `S` and iteration, report:

- `pose_ate_rmse_aligned`
- `pose_are_mean_deg_aligned`
- `pose_rpe_rot_mean_deg`
- `pose_rpe_trans_mean_aligned`
- `pose_sim3_scale` as a diagnostic
- `delta_norm_mean`, `delta_norm_p95`, `delta_norm_max`
- inference time and peak CUDA memory when running the full experiment

The MVP research result is considered meaningful when a non-default iteration improves median aligned ATE by at least 5% over iteration 4 and is non-worse on at least 7 of 10 scenes. A change visible only in raw scale, one scene, or one frame count is not sufficient.

## Acceptance Criteria

- Existing default calls return the same `pose_enc`, `pose_enc_list`, and non-camera outputs as before.
- Trace-off execution creates no trace lists and has no `O(KSC)` token storage.
- Trace-on execution returns exactly `K` raw poses, deltas, activated poses, and delta-norm slices.
- The study performs one Aggregator pass per frame selection, not five independent model runs.
- All unit tests run on CPU without weights; the documented one-scene CUDA smoke test succeeds when data and weights are available.
- Result interpretation follows the repository aligned/raw rule exactly.

## Gated Follow-up Plans

These stages are deliberately excluded from the first code change. Each begins only after the preceding evidence gate and receives its own implementation plan before code is edited.

### Round 2 Gate: Geometry-aware Iteration Selection

Enter when the iteration curve is U-shaped or different scenes prefer different iterations. Create `pre_experiments/camera_iteration/geometry_score.py`, `pre_experiments/camera_iteration/select_iteration.py`, and `probe/tests/test_camera_geometry_score.py`. The score may consume predicted depth, point map, confidence, focal stability, temporal smoothness, and correction norm, but never GT. Keep an explicitly named oracle-by-ATE result only as an upper bound. Proceed when the selected iteration improves median aligned ATE by at least 5% over fixed iteration 4 on at least 7 of 10 scenes.

### Round 3 Gate: Training-free SE(3) Pose Refinement

Enter when iteration choice alone is insufficient and a GT-free geometry score correlates with aligned pose error. Add `vggt/utils/se3.py`, `pre_experiments/pose_refinement/geometry_residuals.py`, `pre_experiments/pose_refinement/run_refinement.py`, and focused tests. Optimize `[S-1,6]` corrections with frame 0 fixed, left-compose `exp(delta_xi^)` with predicted `w2c`, recompute residuals every step, and retain correction regularization. Reject the method if geometry loss falls while median aligned ATE or aligned point-cloud error consistently worsens.

### Round 4 Gate: Low-rank Camera Latent Probing

Enter after a stable GT-free geometry objective exists. Use the Task 1 `decode_pose_tokens()` API and optimize only coefficients `a[S,d]` in `z' = z + a B^T`, with fixed orthonormal `B[C,d]`, `d in {16,32,64}`, and three random bases. Add `pre_experiments/latent_probing/run_probe.py` plus tests for frozen parameters and gradient flow. Continue only if latent probing beats pose-only refinement by at least 5% median aligned ATE and succeeds for at least two of three bases.

### Round 5 Gate: Small-scale Tiny Latent Denoiser

Enter only after Round 4 passes. Create a separate training plan for a zero-initialized, one-step, width-256 denoiser; freeze Image Encoder, Aggregator, Camera Head, depth Head, and point Head. Train on clean/degraded context pairs with disjoint train/validation/evaluation scene lists. Do not use raw token MSE across unmatched gauges; use pose, relative pose, geometry consistency, relational latent, and `Delta z` regularization losses. Multi-step denoising and a learned pose refiner remain excluded until the one-step adapter provides a reproducible gain.

### Round 6 Gate: Latent and Pose Composition

Enter only when Round 3 and Round 5 each pass their own held-out evaluation and an ablation shows non-overlapping gains. The fixed order is `z -> denoiser -> frozen Camera Head -> one-step SE(3) correction`. Compare the combined result against each component alone with identical frames and metrics; keep the combined system only when it improves median aligned ATE without degrading aligned point-cloud error or producing unstable correction norms.
