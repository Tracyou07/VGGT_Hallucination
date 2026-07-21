# AutoDL Three-Script Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three independent AutoDL preparation scripts for the shared `vggt` conda environment, VGGT-1B weights, and an officially authorized ten-scene ScanNet subset, while keeping the camera experiment runner execution-only.

**Architecture:** Standard-library preflight helpers own checkpoint and complete-scene validation. Three idempotent Bash entrypoints separately prepare environment, weights, and data; `run_camera_iteration.sh` only activates `vggt`, validates prerequisites, and launches the existing Python study. Acquisition artifacts stay under `/root/autodl-tmp`, and durable usage instructions replace the temporary three-script design document after local verification.

**Tech Stack:** Bash, Miniconda, Python 3.10+, `unittest`, PyTorch/CUDA from the AutoDL base image, `huggingface_hub`, official ScanNet `download-scannet.py`.

## Global Constraints

- Work only on `camera-iteration-preexperiment`; keep the worktree attached to that named branch.
- Use one conda environment named `vggt`, cloned from `base` only when absent.
- Never install, upgrade, or pin Torch/CUDA in repository requirements or setup scripts.
- Keep environment, weight, and dataset preparation in three independent scripts; do not add an all-in-one wrapper.
- Require `SCANNET_TOS_ACCEPTED=1` before invoking the official ScanNet downloader, then automate its expected confirmations.
- Download only configured `.sens` files; camera iteration needs color images and raw GT poses, not depth, meshes, labels, or PLY files.
- Existing non-empty checkpoints, `.sens` files, and complete processed scenes must be skipped.
- `run_camera_iteration.sh` must not create environments, install packages, download files, or extract `.sens` data.
- Unit tests must not require CUDA, conda, network access, model weights, or ScanNet.
- Prediction metrics use aligned values for conclusions; raw GT remains raw and evaluation-only.
- Delete `doc/2026-07-21_AutoDL_Three_Script_Setup_Design.md` after scripts, tests, README instructions, and local verification are complete. Keep `doc/2026-07-16_Camera_Iteration_Worktree_Design.md`.

## File Map

| Path | Responsibility |
|---|---|
| `scripts/autodl/setup_vggt_env.sh` | Create/reuse `vggt`, preserve base Torch/CUDA, install only project/light dependencies |
| `scripts/autodl/download_vggt_weights.sh` | Download or reuse a non-empty VGGT-1B checkpoint through a configurable HF endpoint |
| `scripts/autodl/prepare_scannet_camera_iteration.sh` | Officially download configured `.sens` scenes, auto-confirm after explicit ToS gate, and extract color/pose |
| `scripts/autodl/run_camera_iteration.sh` | Activate `vggt`, validate complete local inputs, run the study only |
| `scripts/autodl/camera_iteration/preflight.py` | Pure checkpoint, dependency, scene-list, raw-layout, and processed-layout contracts |
| `scripts/autodl/camera_iteration/extract_scannet_sens.py` | Extract and resume color/raw-pose output using shared completeness logic |
| `tests/camera_iteration/test_autodl_preflight.py` | Pure filesystem and extraction contracts |
| `tests/camera_iteration/test_autodl_scripts.py` | Static separation/default/license contracts for all Bash entrypoints |
| `README.md` | New-machine command sequence and branch-level quick start |
| `pre_experiments/camera_iteration/README.md` | Exact environment variables, acquisition, smoke run, and resume instructions |
| `log/2026-07-21_autodl_three_stage_setup.md` | Actual implementation and verification record; no fabricated remote result |

---

### Task 1: Strengthen Shared Preflight Contracts

**Files:**
- Modify: `scripts/autodl/camera_iteration/preflight.py`
- Modify: `scripts/autodl/camera_iteration/extract_scannet_sens.py`
- Modify: `tests/camera_iteration/test_autodl_preflight.py`

**Interfaces:**
- Produces: `processed_scene_is_complete(scene_dir: Path) -> bool`
- Produces: `missing_processed_scenes(scannet_root: Path, scenes: Sequence[str]) -> list[str]`
- Preserves: `find_checkpoint()`, `read_scene_list()`, `detect_scannet_layout()`, and extraction's public `scene_is_complete()` alias
- Contract: supported checkpoints and raw `.sens` files must be non-empty; `processed` requires every requested scene to contain at least one shared image/finite-4x4-pose frame ID

- [ ] **Step 1: Update tests to reject empty checkpoints and partial processed data**

Change checkpoint fixtures from `.touch()` to non-empty bytes and add explicit empty-file rejection:

```python
def test_checkpoint_prefers_nonempty_safetensors_and_accepts_model_pt(self):
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        model_pt = checkpoint_dir / "model.pt"
        model_pt.write_bytes(b"pt")
        self.assertEqual(find_checkpoint(checkpoint_dir), model_pt)

        safetensors = checkpoint_dir / "model.safetensors"
        safetensors.write_bytes(b"safe")
        self.assertEqual(find_checkpoint(checkpoint_dir), safetensors)

        safetensors.write_bytes(b"")
        self.assertEqual(find_checkpoint(checkpoint_dir), model_pt)

def test_empty_checkpoint_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        (checkpoint_dir / "model.safetensors").touch()
        with self.assertRaisesRegex(FileNotFoundError, "non-empty"):
            find_checkpoint(checkpoint_dir)
```

Create valid processed frames with a finite 4x4 pose and verify that one complete scene cannot satisfy a two-scene request:

```python
def write_processed_frame(root: Path, scene: str, frame_id: int = 0) -> None:
    scene_dir = root / "process_scannet" / scene
    (scene_dir / "color").mkdir(parents=True)
    (scene_dir / "pose").mkdir()
    (scene_dir / "color" / f"{frame_id}.jpg").write_bytes(b"image")
    pose = "\n".join(
        " ".join("1" if row == col else "0" for col in range(4))
        for row in range(4)
    )
    (scene_dir / "pose" / f"{frame_id}.txt").write_text(pose, encoding="utf-8")

def test_processed_layout_requires_every_configured_scene(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_processed_frame(root, "scene0000_00")
        with self.assertRaisesRegex(FileNotFoundError, "scene0013_02"):
            detect_scannet_layout(root, ["scene0000_00", "scene0013_02"])
```

Add a case where image and pose stems differ or the pose contains `nan`; both must be incomplete. Update `FakeSensorData.export_poses()` to write a finite identity matrix instead of the string `pose`.
Also replace raw `.sens` fixture `.touch()` calls with `write_bytes(b"sens")`,
and make `FakeSensorData.export_color_images()` write non-empty image bytes.

- [ ] **Step 2: Run focused tests and confirm the old permissive behavior fails**

Run:

```powershell
python -m unittest tests.camera_iteration.test_autodl_preflight -v
```

Expected: FAIL because empty checkpoints are currently accepted, processed directories do not inspect files, and one requested scene currently satisfies a multi-scene request.

- [ ] **Step 3: Implement non-empty and all-scenes validation**

Add `math` to standard-library imports and implement these helpers in `preflight.py`:

```python
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_stems(color_dir: Path) -> set[str]:
    if not color_dir.is_dir():
        return set()
    return {
        path.stem
        for path in color_dir.iterdir()
        if path.is_file() and path.stat().st_size > 0
        and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _finite_pose_stems(pose_dir: Path) -> set[str]:
    if not pose_dir.is_dir():
        return set()
    valid: set[str] = set()
    for path in pose_dir.glob("*.txt"):
        try:
            values = [float(value) for value in path.read_text(encoding="utf-8").split()]
        except (OSError, ValueError):
            continue
        if len(values) == 16 and all(math.isfinite(value) for value in values):
            valid.add(path.stem)
    return valid


def processed_scene_is_complete(scene_dir: Path) -> bool:
    """Return true when image and finite raw-pose IDs intersect."""
    return bool(
        _image_stems(scene_dir / "color")
        .intersection(_finite_pose_stems(scene_dir / "pose"))
    )


def missing_processed_scenes(
    scannet_root: Path,
    scenes: Sequence[str],
) -> list[str]:
    process_root = scannet_root / "process_scannet"
    return [
        scene
        for scene in scenes
        if not processed_scene_is_complete(process_root / scene)
    ]


def _raw_scene_is_complete(scans_root: Path, scene: str) -> bool:
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in scans_root.rglob(f"{scene}.sens")
    )
```

Change checkpoint resolution to:

```python
def find_checkpoint(checkpoint_dir: Path) -> Path:
    for filename in ("model.safetensors", "model.pt"):
        candidate = checkpoint_dir / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(
        "CKPT_DIR must contain non-empty model.safetensors or model.pt: "
        f"{checkpoint_dir.resolve()}"
    )
```

For configured scenes, make `detect_scannet_layout()` require all processed scenes or all raw scenes and name missing scene IDs in its error. Preserve the no-scene fallback for manual inspection:

```python
def detect_scannet_layout(
    scannet_root: Path,
    scenes: Sequence[str] | None = None,
) -> Literal["processed", "raw"]:
    if scenes:
        missing_processed = missing_processed_scenes(scannet_root, scenes)
        if not missing_processed:
            return "processed"
        scans_root = scannet_root / "raw_sens" / "scans"
        missing_raw = [
            scene for scene in scenes
            if not _raw_scene_is_complete(scans_root, scene)
        ]
        if not missing_raw:
            return "raw"
        raise FileNotFoundError(
            "SCANNET_ROOT is incomplete; missing processed scenes "
            f"{missing_processed} and raw scenes {missing_raw}: "
            f"{scannet_root.resolve()}"
        )

    process_root = scannet_root / "process_scannet"
    if process_root.is_dir() and any(
        processed_scene_is_complete(path)
        for path in process_root.iterdir()
        if path.is_dir()
    ):
        return "processed"
    scans_root = scannet_root / "raw_sens" / "scans"
    if scans_root.is_dir() and any(
        path.is_file() and path.stat().st_size > 0
        for path in scans_root.rglob("*.sens")
    ):
        return "raw"
    raise FileNotFoundError(
        "SCANNET_ROOT has neither complete process_scannet scenes nor "
        f"non-empty raw_sens/scans/**/*.sens: {scannet_root.resolve()}"
    )
```

In `extract_scannet_sens.py`, remove its private stem scanner and delegate while preserving imports used by existing tests:

```python
from scripts.autodl.camera_iteration.preflight import (
    processed_scene_is_complete,
    read_scene_list,
)


def scene_is_complete(scene_dir: Path) -> bool:
    return processed_scene_is_complete(scene_dir)
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
python -m unittest tests.camera_iteration.test_autodl_preflight -v
python -m unittest discover -s tests
```

Expected: all preflight tests and the complete CPU suite PASS.

- [ ] **Step 5: Commit the stricter input contract**

```powershell
git add scripts/autodl/camera_iteration/preflight.py scripts/autodl/camera_iteration/extract_scannet_sens.py tests/camera_iteration/test_autodl_preflight.py
git commit -m "Require complete camera experiment inputs"
```

---

### Task 2: Add the Shared `vggt` Environment Script

**Files:**
- Create: `scripts/autodl/setup_vggt_env.sh`
- Create: `tests/camera_iteration/test_autodl_scripts.py`

**Interfaces:**
- Consumes: `preflight.py --print-missing`, repository root, AutoDL Miniconda base environment
- Produces: reusable conda environment `vggt` with inherited Torch/CUDA and editable VGGT
- Environment overrides: `CONDA_ROOT`, `CONDA_ENV_NAME`, `CONDA_CLONE_FROM`

- [ ] **Step 1: Write the failing environment-script contract test**

Create `test_autodl_scripts.py`:

```python
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def script_text(name: str) -> str:
    return (ROOT / "scripts" / "autodl" / name).read_text(encoding="utf-8")


class AutoDLScriptsTest(unittest.TestCase):
    def test_environment_script_clones_base_without_installing_torch(self):
        content = script_text("setup_vggt_env.sh")
        for expected in (
            'CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"',
            'CONDA_CLONE_FROM="${CONDA_CLONE_FROM:-base}"',
            'conda create --name "$CONDA_ENV_NAME" --clone "$CONDA_CLONE_FROM"',
            "--no-deps --no-build-isolation -e",
            "preflight.py\" --print-missing",
            "torch.cuda.is_available()",
        ):
            self.assertIn(expected, content)
        self.assertIsNone(
            re.search(
                r"(?:pip3?|python\s+-m\s+pip)\s+install[^\n]*\btorch",
                content,
                re.I,
            )
        )

    def test_camera_requirements_never_install_torch(self):
        requirements = (ROOT / "requirements-camera-iteration.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(requirements, r"(?im)^\s*torch(?:vision|audio)?(?:[<=>]|\s|$)")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm the script is absent**

```powershell
python -m unittest tests.camera_iteration.test_autodl_scripts -v
```

Expected: ERROR with `FileNotFoundError` for `setup_vggt_env.sh`.

- [ ] **Step 3: Implement the environment script**

Create the script with this complete control flow:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_CLONE_FROM="${CONDA_CLONE_FROM:-base}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  printf '[env] missing conda initialization: %s\n' "$CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"

if ! conda run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  printf '[env] cloning %s -> %s\n' "$CONDA_CLONE_FROM" "$CONDA_ENV_NAME"
  conda create --name "$CONDA_ENV_NAME" --clone "$CONDA_CLONE_FROM" -y
else
  printf '[env] reuse conda environment: %s\n' "$CONDA_ENV_NAME"
fi
conda activate "$CONDA_ENV_NAME"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda) + ' device=' + torch.cuda.get_device_name(0))"
python -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"

mapfile -t missing_specs < <(
  python "$REPO_ROOT/scripts/autodl/camera_iteration/preflight.py" --print-missing
)
if (( ${#missing_specs[@]} > 0 )); then
  printf '[env] installing missing lightweight packages: %s\n' "${missing_specs[*]}"
  python -m pip install "${missing_specs[@]}"
fi

mapfile -t remaining_specs < <(
  python "$REPO_ROOT/scripts/autodl/camera_iteration/preflight.py" --print-missing
)
if (( ${#remaining_specs[@]} > 0 )); then
  printf '[env] unresolved packages: %s\n' "${remaining_specs[*]}" >&2
  exit 1
fi
printf '[env] ready: %s\n' "$CONDA_ENV_NAME"
```

- [ ] **Step 4: Verify the test and shell syntax**

```bash
python -m unittest tests.camera_iteration.test_autodl_scripts -v
bash -n scripts/autodl/setup_vggt_env.sh
```

Expected: tests PASS and `bash -n` exits 0. Do not execute conda creation in local CPU tests.

- [ ] **Step 5: Commit the environment entrypoint**

```bash
git add scripts/autodl/setup_vggt_env.sh tests/camera_iteration/test_autodl_scripts.py
git commit -m "Add shared VGGT environment setup"
```

---

### Task 3: Add Independent VGGT-1B Weight Acquisition

**Files:**
- Create: `scripts/autodl/download_vggt_weights.sh`
- Modify: `tests/camera_iteration/test_autodl_scripts.py`

**Interfaces:**
- Consumes: existing `vggt` environment and `huggingface_hub`
- Produces: non-empty `CKPT_DIR/model.safetensors` or `CKPT_DIR/model.pt`
- Overrides: `CONDA_ROOT`, `CONDA_ENV_NAME`, `CKPT_DIR`, `HF_REPO`, `HF_ENDPOINT`, `HF_HOME`, `HF_MAX_RETRIES`

- [ ] **Step 1: Add the failing weight-script contract test**

Append:

```python
def test_weight_script_is_independent_resumable_and_mirror_aware(self):
    content = script_text("download_vggt_weights.sh")
    for expected in (
        'CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"',
        'HF_REPO="${HF_REPO:-facebook/VGGT-1B}"',
        'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"',
        'CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"',
        "snapshot_download",
        "max_workers=1",
        "model.safetensors",
        "model.pt",
        "HF_MAX_RETRIES",
    ):
        self.assertIn(expected, content)
    self.assertNotIn("conda create", content)
    self.assertNotRegex(content, r"(?i)scannet|run_camera_iteration")
```

- [ ] **Step 2: Run the focused test and confirm the weight script is absent**

```powershell
python -m unittest tests.camera_iteration.test_autodl_scripts.AutoDLScriptsTest.test_weight_script_is_independent_resumable_and_mirror_aware -v
```

Expected: ERROR with `FileNotFoundError`.

- [ ] **Step 3: Implement bounded, non-empty checkpoint download**

Create `download_vggt_weights.sh` with this complete initialization and environment guard:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  printf '[weights] missing conda initialization: %s\n' "$CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
if ! conda run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  printf '[weights] missing conda environment %s. Run scripts/autodl/setup_vggt_env.sh first.\n' "$CONDA_ENV_NAME" >&2
  exit 1
fi
conda activate "$CONDA_ENV_NAME"
```

After activation, set:

```bash
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
HF_REPO="${HF_REPO:-facebook/VGGT-1B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-$AUTODL_TMP/hf_home}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-5}"
export HF_ENDPOINT HF_HOME HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$CKPT_DIR" "$HF_HOME"

if [[ -s "$CKPT_DIR/model.safetensors" || -s "$CKPT_DIR/model.pt" ]]; then
  printf '[weights] reuse checkpoint: %s\n' "$CKPT_DIR"
  exit 0
fi
```

Use the environment's Python for retry logic, passing all values as positional
arguments to a quoted heredoc:

```bash
python - "$HF_REPO" "$CKPT_DIR" "$HF_MAX_RETRIES" <<'PY'
from pathlib import Path
import sys
import time

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2])
max_retries = int(sys.argv[3])

for attempt in range(1, max_retries + 1):
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=["model.safetensors", "config.json"],
            max_workers=1,
            etag_timeout=60,
        )
        break
    except Exception as error:
        if attempt == max_retries:
            raise SystemExit(
                f"[weights] failed after {max_retries} attempts: {error}"
            )
        print(f"[weights] retry {attempt}/{max_retries}: {error}", flush=True)
        time.sleep(min(30, attempt * 5))
PY
```

After Python exits, require a supported non-empty result:

```bash
if [[ ! -s "$CKPT_DIR/model.safetensors" && ! -s "$CKPT_DIR/model.pt" ]]; then
  printf '[weights] no supported non-empty checkpoint in %s\n' "$CKPT_DIR" >&2
  exit 1
fi
printf '[weights] ready: %s\n' "$CKPT_DIR"
```

- [ ] **Step 4: Run static tests and syntax checks**

```bash
python -m unittest tests.camera_iteration.test_autodl_scripts -v
bash -n scripts/autodl/download_vggt_weights.sh
```

Expected: all script contract tests PASS and syntax exits 0. No network call occurs in tests.

- [ ] **Step 5: Commit the weight entrypoint**

```bash
git add scripts/autodl/download_vggt_weights.sh tests/camera_iteration/test_autodl_scripts.py
git commit -m "Add VGGT weight acquisition script"
```

---

### Task 4: Add Official ScanNet Download and Color/Pose Extraction

**Files:**
- Create: `scripts/autodl/prepare_scannet_camera_iteration.sh`
- Modify: `tests/camera_iteration/test_autodl_scripts.py`

**Interfaces:**
- Consumes: `SCANNET_TOS_ACCEPTED=1`, official `download-scannet.py`, configured scene list, branch-local extractor
- Produces: non-empty raw `.sens` plus complete `process_scannet/<scene>/{color,pose}` for every requested scene
- Overrides: `CONDA_ROOT`, `CONDA_ENV_NAME`, `SCANNET_ROOT`, `RAW_DOWNLOAD_ROOT`, `RAW_DIR`, `PROCESS_DIR`, `SCENE_LIST`, `SCENE_LIMIT`, `SCANNET_DOWNLOAD_SCRIPT`, `SCANNET_DOWNLOAD_URL`

- [ ] **Step 1: Add failing license, scope, and extraction tests**

Append:

```python
def test_scannet_script_requires_terms_and_only_prepares_camera_inputs(self):
    content = script_text("prepare_scannet_camera_iteration.sh")
    for expected in (
        'SCANNET_TOS_ACCEPTED',
        'SCANNET_DOWNLOAD_SCRIPT',
        'http://kaldir.vc.in.tum.de/scannet/download-scannet.py',
        'configs/camera_iteration_scannet.txt',
        '--type .sens',
        'extract_scannet_sens.py',
        'missing_processed_scenes',
    ):
        self.assertIn(expected, content)
    self.assertNotIn("--export-depth", content)
    self.assertNotIn("_vh_clean_2.ply", content)
    self.assertNotIn("download_vggt_weights", content)
    self.assertNotIn("conda create", content)

def test_scannet_terms_gate_precedes_download_commands(self):
    content = script_text("prepare_scannet_camera_iteration.sh")
    gate = content.index('SCANNET_TOS_ACCEPTED')
    official_url = content.index('SCANNET_DOWNLOAD_URL')
    self.assertLess(gate, official_url)
```

- [ ] **Step 2: Run the focused tests and confirm the data script is absent**

```powershell
python -m unittest tests.camera_iteration.test_autodl_scripts.AutoDLScriptsTest.test_scannet_script_requires_terms_and_only_prepares_camera_inputs tests.camera_iteration.test_autodl_scripts.AutoDLScriptsTest.test_scannet_terms_gate_precedes_download_commands -v
```

Expected: ERROR with `FileNotFoundError`.

- [ ] **Step 3: Implement official subset acquisition**

Create the script with this initialization, existing-environment check, and
license gate. The gate appears before any downloader URL or command:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

if [[ "${SCANNET_TOS_ACCEPTED:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
[scannet] Official ScanNet terms must be accepted before download.
[scannet] Review http://www.scan-net.org/ and rerun with SCANNET_TOS_ACCEPTED=1.
EOF
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  printf '[scannet] missing conda initialization: %s\n' "$CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
if ! conda run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  printf '[scannet] missing conda environment %s. Run scripts/autodl/setup_vggt_env.sh first.\n' "$CONDA_ENV_NAME" >&2
  exit 1
fi
conda activate "$CONDA_ENV_NAME"
```

Then define exact defaults:

```bash
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
SCANNET_ROOT="${SCANNET_ROOT:-$AUTODL_TMP/datasets/scannetv2}"
RAW_DOWNLOAD_ROOT="${RAW_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw_sens}"
RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/camera_iteration_scannet.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
SCANNET_DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-$SCANNET_ROOT/tools/download-scannet.py}"
SCANNET_DOWNLOAD_URL="${SCANNET_DOWNLOAD_URL:-http://kaldir.vc.in.tum.de/scannet/download-scannet.py}"
```

Retrieve the official script only when missing or empty, preferring `curl` then
`wget`, and fail with the override path when neither succeeds:

```bash
mkdir -p "$RAW_DIR" "$PROCESS_DIR" "$(dirname "$SCANNET_DOWNLOAD_SCRIPT")"
if [[ ! -s "$SCANNET_DOWNLOAD_SCRIPT" ]]; then
  printf '[scannet] retrieving official downloader: %s\n' "$SCANNET_DOWNLOAD_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$SCANNET_DOWNLOAD_URL" -o "$SCANNET_DOWNLOAD_SCRIPT"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$SCANNET_DOWNLOAD_SCRIPT" "$SCANNET_DOWNLOAD_URL"
  else
    printf '[scannet] curl or wget is required, or set SCANNET_DOWNLOAD_SCRIPT.\n' >&2
    exit 1
  fi
fi
if [[ ! -s "$SCANNET_DOWNLOAD_SCRIPT" ]]; then
  printf '[scannet] official downloader unavailable: %s\n' "$SCANNET_DOWNLOAD_SCRIPT" >&2
  exit 1
fi
```

Read scenes through the shared Python helper so comments and `SCENE_LIMIT=0`
retain existing semantics:

```bash
mapfile -t scenes < <(
  python - "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys
from scripts.autodl.camera_iteration.preflight import read_scene_list

for scene in read_scene_list(Path(sys.argv[1]), int(sys.argv[2])):
    print(scene)
PY
)
if (( ${#scenes[@]} == 0 )); then
  printf '[scannet] no scenes selected from %s\n' "$SCENE_LIST" >&2
  exit 1
fi
```

For each scene, skip a non-empty expected file, repair the known nested
official output if present, otherwise invoke the official script:

```bash
for scene in "${scenes[@]}"; do
  expected="$RAW_DIR/$scene/$scene.sens"
  nested="$RAW_DIR/scans/$scene/$scene.sens"
  if [[ ! -s "$expected" && -s "$nested" ]]; then
    mkdir -p "$(dirname "$expected")"
    mv "$nested" "$expected"
  fi
  if [[ -s "$expected" ]]; then
    printf '[scannet] reuse %s\n' "$expected"
    continue
  fi
  printf '\n\n\n\n' | python "$SCANNET_DOWNLOAD_SCRIPT" \
    -o "$RAW_DOWNLOAD_ROOT" --id "$scene" --type .sens
  if [[ ! -s "$expected" && -s "$nested" ]]; then
    mkdir -p "$(dirname "$expected")"
    mv "$nested" "$expected"
  fi
  if [[ ! -s "$expected" ]]; then
    printf '[scannet] missing downloaded scene: %s\n' "$expected" >&2
    exit 1
  fi
done
```

Once all raw files exist, invoke:

```bash
python "$REPO_ROOT/scripts/autodl/camera_iteration/extract_scannet_sens.py" \
  --raw-dir "$RAW_DIR" \
  --out-dir "$PROCESS_DIR" \
  --scene-list "$SCENE_LIST" \
  --scene-limit "$SCENE_LIMIT"
```

Finish with a quoted Python heredoc that validates every selected scene:

```bash
python - "$SCANNET_ROOT" "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys

from scripts.autodl.camera_iteration.preflight import (
    missing_processed_scenes,
    read_scene_list,
)

root = Path(sys.argv[1])
scenes = read_scene_list(Path(sys.argv[2]), int(sys.argv[3]))
missing = missing_processed_scenes(root, scenes)
if missing:
    raise SystemExit(f"[scannet] incomplete processed scenes: {missing}")
print(f"[scannet] ready scenes={len(scenes)} root={root / 'process_scannet'}")
PY
```

- [ ] **Step 4: Run static tests and shell syntax**

```bash
python -m unittest tests.camera_iteration.test_autodl_scripts -v
bash -n scripts/autodl/prepare_scannet_camera_iteration.sh
```

Expected: all static contracts PASS and shell syntax exits 0. The test suite must not invoke the official server.

- [ ] **Step 5: Commit the ScanNet entrypoint**

```bash
git add scripts/autodl/prepare_scannet_camera_iteration.sh tests/camera_iteration/test_autodl_scripts.py
git commit -m "Add camera ScanNet preparation script"
```

---

### Task 5: Make the Camera Runner Execution-Only

**Files:**
- Modify: `scripts/autodl/run_camera_iteration.sh`
- Modify: `tests/camera_iteration/test_autodl_preflight.py`
- Modify: `tests/camera_iteration/test_autodl_scripts.py`

**Interfaces:**
- Consumes: existing `vggt`, complete local checkpoint, complete processed scenes
- Produces: unchanged invocation of `pre_experiments.camera_iteration.run_study`
- Removes: conda creation, editable install, package installation, raw extraction, `RUN_EXTRACT`

- [ ] **Step 1: Replace the old runner test with strict separation assertions**

Move runner text assertions into `test_autodl_scripts.py` and use:

```python
def test_runner_only_activates_validates_and_runs(self):
    content = script_text("run_camera_iteration.sh")
    for expected in (
        'CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"',
        "preflight.py",
        "run_study",
        "setup_vggt_env.sh",
        "prepare_scannet_camera_iteration.sh",
        "25 50 100 200 500",
        "1 2 4 8 16",
    ):
        self.assertIn(expected, content)
    for forbidden in (
        "conda create",
        "pip install",
        "snapshot_download",
        "extract_scannet_sens.py",
        "RUN_EXTRACT",
        "wget",
        "curl",
    ):
        self.assertNotIn(forbidden, content)
```

Remove `test_runner_has_fixed_defaults_and_no_dataset_or_weight_downloads()` from `test_autodl_preflight.py`.

- [ ] **Step 2: Run the focused test and confirm the mixed runner fails**

```powershell
python -m unittest tests.camera_iteration.test_autodl_scripts.AutoDLScriptsTest.test_runner_only_activates_validates_and_runs -v
```

Expected: FAIL because the runner currently creates a conda environment, installs packages, exposes `RUN_EXTRACT`, and calls the extractor.

- [ ] **Step 3: Reduce the runner to activation, preflight, and execution**

Keep existing experiment variables, changing only `CONDA_ENV_NAME` default to `vggt` and removing `CONDA_CLONE_FROM` and `RUN_EXTRACT`. After sourcing conda:

```bash
if ! conda run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  printf '[run] missing conda environment %s. Run scripts/autodl/setup_vggt_env.sh first.\n' "$CONDA_ENV_NAME" >&2
  exit 1
fi
conda activate "$CONDA_ENV_NAME"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda) + ' device=' + torch.cuda.get_device_name(0))"
```

Retain the current preflight array. Capture layout without letting `set -e` hide the actionable instruction:

```bash
if ! layout="$("${preflight[@]}" --print-layout)"; then
  printf '[run] incomplete local inputs. Run weight and ScanNet preparation scripts first.\n' >&2
  exit 1
fi
if [[ "$layout" != "processed" ]]; then
  printf '[run] processed ScanNet data required. Run scripts/autodl/prepare_scannet_camera_iteration.sh.\n' >&2
  exit 1
fi
```

Preserve frame/iteration array parsing and the existing `run_study` invocation exactly. Do not call any preparation script automatically.

- [ ] **Step 4: Run runner and full regression checks**

```bash
python -m unittest tests.camera_iteration.test_autodl_scripts tests.camera_iteration.test_autodl_preflight -v
python -m unittest discover -s tests
bash -n scripts/autodl/run_camera_iteration.sh
```

Expected: all tests PASS and shell syntax exits 0.

- [ ] **Step 5: Commit execution-only runner behavior**

```bash
git add scripts/autodl/run_camera_iteration.sh tests/camera_iteration/test_autodl_preflight.py tests/camera_iteration/test_autodl_scripts.py
git commit -m "Separate camera runner from setup"
```

---

### Task 6: Publish Durable Instructions and Remove the Temporary Design

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `pre_experiments/camera_iteration/README.md`
- Create: `log/2026-07-21_autodl_three_stage_setup.md`
- Delete: `doc/2026-07-21_AutoDL_Three_Script_Setup_Design.md`
- Preserve: `doc/2026-07-16_Camera_Iteration_Worktree_Design.md`

**Interfaces:**
- Produces: exact new-machine command sequence and environment-variable reference
- Records: local verification only; remote download and experiment results remain unclaimed until Task 7

- [ ] **Step 1: Update root quick start**

Replace the old “weights and data already exist” section with:

```bash
git clone -b camera-iteration-preexperiment \
  https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination

bash scripts/autodl/setup_vggt_env.sh
bash scripts/autodl/download_vggt_weights.sh
SCANNET_TOS_ACCEPTED=1 \
  bash scripts/autodl/prepare_scannet_camera_iteration.sh
SCENE_LIMIT=1 FRAME_COUNTS="25" ITERATIONS="1 2 4 8 16" \
  bash scripts/autodl/run_camera_iteration.sh
```

State that Torch/CUDA come from the AutoDL base image, all experiments use `vggt`, and preparation scripts are independently resumable.

- [ ] **Step 2: Rewrite the detailed AutoDL section**

In `pre_experiments/camera_iteration/README.md`, document each script separately, including all defaults from the design, the explicit ScanNet ToS gate, official-script override, skip rules, absence of depth/mesh downloads, runner separation, smoke command, identical rerun resume check, and full protocol command.

Update `AGENTS.md` commands to include `bash -n` for all four shell entrypoints and state that only the environment script may install dependencies, only the weight script may contact Hugging Face, and only the ScanNet script may contact the official dataset server.

- [ ] **Step 3: Record actual local work and delete only the requested design file**

Create the dated log with sections for environment, weights, data, runner, and verification. State explicitly that local tests did not download data or weights and did not run CUDA inference. Then delete:

```powershell
git rm doc/2026-07-21_AutoDL_Three_Script_Setup_Design.md
```

Verify the preserved design still exists:

```powershell
Test-Path doc\2026-07-16_Camera_Iteration_Worktree_Design.md
Test-Path doc\2026-07-21_AutoDL_Three_Script_Setup_Design.md
```

Expected: `True`, then `False`.

- [ ] **Step 4: Run the complete local verification gate**

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/autodl/camera_iteration/preflight.py scripts/autodl/camera_iteration/extract_scannet_sens.py
bash -n scripts/autodl/setup_vggt_env.sh
bash -n scripts/autodl/download_vggt_weights.sh
bash -n scripts/autodl/prepare_scannet_camera_iteration.sh
bash -n scripts/autodl/run_camera_iteration.sh
git diff --check
git status --short
```

Expected: all CPU tests PASS, compilation and all four syntax checks exit 0, no whitespace errors, and only documented files are modified/deleted.

- [ ] **Step 5: Commit durable documentation and design cleanup**

```bash
git add README.md AGENTS.md pre_experiments/camera_iteration/README.md log/2026-07-21_autodl_three_stage_setup.md doc/2026-07-21_AutoDL_Three_Script_Setup_Design.md
git commit -m "Document three-stage AutoDL setup"
```

---

### Task 7: Push and Run the Remote Smoke Gate

**Files:**
- Modify after successful remote execution: `log/2026-07-21_autodl_three_stage_setup.md`

**Interfaces:**
- Consumes: pushed `camera-iteration-preexperiment`, new AutoDL machine, official ScanNet acceptance, working GitHub/HF/ScanNet connectivity
- Produces: one-scene 25-frame results and a verified resume event

- [ ] **Step 1: Push the implementation branch**

```bash
git push origin camera-iteration-preexperiment
```

Expected: remote branch advances to the local implementation commit.

- [ ] **Step 2: Prepare the new AutoDL machine in three explicit stages**

```bash
git clone -b camera-iteration-preexperiment \
  https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
bash scripts/autodl/setup_vggt_env.sh
bash scripts/autodl/download_vggt_weights.sh
SCANNET_TOS_ACCEPTED=1 SCENE_LIMIT=1 \
  bash scripts/autodl/prepare_scannet_camera_iteration.sh
```

Expected: `vggt` reports existing Torch/CUDA, a non-empty checkpoint exists, and `scene0000_00/color` plus `pose` are complete.

- [ ] **Step 3: Run and rerun the smoke experiment**

```bash
SCENE_LIMIT=1 FRAME_COUNTS="25" ITERATIONS="1 2 4 8 16" \
  bash scripts/autodl/run_camera_iteration.sh
SCENE_LIMIT=1 FRAME_COUNTS="25" ITERATIONS="1 2 4 8 16" \
  bash scripts/autodl/run_camera_iteration.sh
```

Expected: first run writes five requested metric rows plus a trace through iteration 16; second run prints `[resume]` and performs no model forward for the completed selection.

- [ ] **Step 4: Validate remote artifacts**

Under `/root/autodl-tmp/camera_iteration/results/<run_id>/`, require:

```text
run_metadata.json
summary.json
summary.csv
scene0000_00/frames_25/iteration_metrics.json
scene0000_00/frames_25/iteration_metrics.csv
scene0000_00/frames_25/camera_trace.npz
scene0000_00/frames_25/selected_frame_ids.json
scene0000_00/frames_25/complete.json
```

Use Python to assert five summary rows, finite metrics, `camera_trace.npz` iteration axis length 16, and matching run IDs in metadata/completion.

- [ ] **Step 5: Record only observed remote evidence**

Append the actual AutoDL GPU, Torch/CUDA versions, commands, resolved input paths, run ID, result path, elapsed time, peak memory, and resume output to the dated log. If any stage fails, record the exact failing stage and error without marking the smoke gate complete. Commit and push the log only after it reflects observed output.

## Acceptance Criteria

- A new AutoDL machine runs three explicit preparation scripts in order; no all-in-one setup exists.
- `vggt` inherits working Torch/CUDA from `base`; no repository command installs Torch.
- Weight acquisition defaults to `hf-mirror.com`, retries serially, and skips a non-empty supported checkpoint.
- ScanNet acquisition requires explicit terms acceptance, uses the official downloader, requests only configured `.sens`, and extracts only color/raw pose.
- Every requested scene must be complete before the runner starts.
- The runner performs no installation, download, or extraction.
- All CPU tests and shell syntax checks pass without external resources.
- The temporary three-script design file is deleted after local implementation; the existing camera worktree design remains.
- The remote smoke and resume claims are made only after actual AutoDL evidence exists.
