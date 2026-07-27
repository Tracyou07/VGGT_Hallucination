from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl"


class AutoDLScriptsTest(unittest.TestCase):
    def read(self, name: str) -> str:
        return (AUTODL / name).read_text(encoding="utf-8")

    def test_environment_setup_preserves_base_torch(self):
        content = self.read("setup_vggt_env.sh")
        for value in ("CONDA_ENV_NAME=\"${CONDA_ENV_NAME:-vggt}\"", "--clone", "--no-deps", "--no-build-isolation", "--print-missing", "torch.cuda.is_available"):
            self.assertIn(value, content)
        self.assertNotIn("pip install torch", content.lower())

    def test_fastvggt_scannet50_list(self):
        path = ROOT / "configs" / "fastvggt_scannet50.txt"
        expected = [
            "scene0000_00", "scene0013_02", "scene0029_01", "scene0042_02",
            "scene0056_00", "scene0071_00", "scene0084_01", "scene0096_00",
            "scene0109_00", "scene0121_01", "scene0136_01", "scene0150_00",
            "scene0164_01", "scene0177_01", "scene0194_00", "scene0207_01",
            "scene0221_01", "scene0238_00", "scene0254_01", "scene0267_00",
            "scene0280_00", "scene0294_02", "scene0309_00", "scene0325_01",
            "scene0340_01", "scene0353_02", "scene0367_01", "scene0380_02",
            "scene0395_00", "scene0409_01", "scene0421_02", "scene0435_03",
            "scene0451_01", "scene0466_01", "scene0477_00", "scene0493_01",
            "scene0509_01", "scene0525_00", "scene0540_02", "scene0555_00",
            "scene0571_00", "scene0582_02", "scene0593_00", "scene0606_01",
            "scene0619_00", "scene0631_01", "scene0648_00", "scene0663_01",
            "scene0675_00", "scene0691_00",
        ]
        entries = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(entries), 50)
        self.assertEqual(len(entries), len(set(entries)))
        self.assertTrue(all(re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", entry) for entry in entries))
        self.assertEqual(entries, expected)

    def test_weight_setup_is_independent_and_resumable(self):
        content = self.read("download_vggt_weights.sh")
        for value in ("facebook/VGGT-1B", "https://hf-mirror.com", "snapshot_download", "max_workers=1", "model.safetensors", "model.pt", "HF_MAX_RETRIES"):
            self.assertIn(value, content)
        for forbidden in ("conda create", "scannet", "run_camera_iteration"):
            self.assertNotIn(forbidden, content.lower())

    def test_scannet_setup_downloads_validated_assets_with_bounded_retries(self):
        content = self.read("prepare_scannet_camera_iteration.sh")
        for value in (
            "SCANNET_TOS_ACCEPTED",
            "http://kaldir.vc.in.tum.de/scannet/download-scannet.py",
            'SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/camera_iteration_scannet.txt}"',
            'SCENE_LIMIT="${SCENE_LIMIT:-10}"',
            'DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"',
            'DOWNLOAD_GT_PLY="${DOWNLOAD_GT_PLY:-0}"',
            'GT_DOWNLOAD_ROOT="${GT_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw}"',
            '[[ "$DOWNLOAD_RETRIES" =~ ^[1-9][0-9]*$ ]]',
            '[[ "$DOWNLOAD_GT_PLY" == "0" || "$DOWNLOAD_GT_PLY" == "1" ]]',
            "read_scene_list",
            "re.fullmatch(r\"scene[0-9]{4}_[0-9]{2}\", scene)",
            "len(scenes) != len(set(scenes))",
            "Selected scene list is empty",
            "download_asset() {",
            "for ((attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++)); do",
            'rm -f "${expected}.tmp"',
            "printf '\\n\\n\\n\\n' | python \"$SCANNET_DOWNLOAD_SCRIPT\"",
            "--type \"$file_type\"",
            'sens="$RAW_DOWNLOAD_ROOT/scans/$scene/$scene.sens"',
            'gt_ply="$GT_DOWNLOAD_ROOT/scans/$scene/${scene}_vh_clean_2.ply"',
            'download_asset "$scene" .sens "$RAW_DOWNLOAD_ROOT" "$sens"',
            'download_asset "$scene" _vh_clean_2.ply "$GT_DOWNLOAD_ROOT" "$gt_ply"',
            "extract_scannet_sens.py",
            "missing_processed_scenes",
        ):
            self.assertIn(value, content)
        for forbidden in ("export_depth", "download_vggt_weights", "conda create", "pip install", "snapshot_download", "find \"$raw_download_root\"", "cp \"$found\""):
            self.assertNotIn(forbidden, content.lower())
        self.assertLess(content.index("SCANNET_TOS_ACCEPTED"), content.index("http://kaldir"))

    def test_scannet_setup_preserves_consistent_raw_dir_override(self):
        content = self.read("prepare_scannet_camera_iteration.sh")
        for value in (
            'RAW_DOWNLOAD_ROOT_WAS_SET="${RAW_DOWNLOAD_ROOT+x}"',
            'RAW_DIR_WAS_SET="${RAW_DIR+x}"',
            'RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"',
            '[[ "$(basename "$RAW_DIR")" == "scans" ]]',
            'RAW_DIR_DOWNLOAD_ROOT="$(dirname "$RAW_DIR")"',
            'RAW_DOWNLOAD_ROOT="$RAW_DIR_DOWNLOAD_ROOT"',
            'RAW_DOWNLOAD_ROOT and RAW_DIR must identify the same scans root.',
        ):
            self.assertIn(value, content)

    def test_runner_only_validates_and_executes(self):
        content = self.read("run_camera_iteration.sh")
        for value in ("CONDA_ENV_NAME=\"${CONDA_ENV_NAME:-vggt}\"", "preflight.py", "run_study", "setup_vggt_env.sh", "prepare_scannet_camera_iteration.sh", "25 50 100 200 500", "1 2 4 8 16"):
            self.assertIn(value, content)
        for forbidden in ("conda create", "pip install", "snapshot_download", "extract_scannet", "RUN_EXTRACT", "wget", "curl"):
            self.assertNotIn(forbidden, content)

    def test_context_runner_fixes_round_1_5_protocol(self):
        content = self.read("run_camera_context.sh")
        for value in (
            "camera_context_scannet.txt",
            'FRAME_COUNTS="${FRAME_COUNTS:-25 50 100 200 500}"',
            'ITERATIONS="4"',
            'SAVE_CONTEXT_DIAGNOSTICS="1"',
            "run_camera_iteration.sh",
            "pre_experiments.camera_context.analyze",
        ):
            self.assertIn(value, content)
        self.assertNotIn("SAVE_CAMERA_TOKENS=1", content)

    def test_camera_head_amplification_runner_is_replay_only(self):
        content = self.read("run_camera_head_amplification.sh")
        for value in (
            "camera_head_amplification.run_replay",
            "results/camera_context/911b598_f4577f584448",
            'SHORT_FRAMES="${SHORT_FRAMES:-200}"',
            'LONG_FRAMES="${LONG_FRAMES:-500}"',
            'ITERATIONS="${ITERATIONS:-4}"',
            "context_diagnostics.npz",
            "model.safetensors",
        ):
            self.assertIn(value, content)
        for forbidden in (
            "run_camera_iteration.sh",
            "load_and_preprocess_images",
            "prepare_scannet",
            "pip install",
            "conda create",
        ):
            self.assertNotIn(forbidden, content)

    def test_local_global_runner_fixes_round2a_protocol(self):
        content = self.read("run_local_global_consistency.sh")
        for value in (
            "local_global_consistency.run_study",
            "local_global_consistency.analyze",
            "results/camera_context/911b598_f4577f584448",
            'WINDOW_LENGTH="${WINDOW_LENGTH:-100}"',
            'WINDOW_STRIDE="${WINDOW_STRIDE:-50}"',
            'CAMERA_ITERATIONS="4"',
            "--run-dir-file",
            "process_scannet",
            "model.safetensors",
        ):
            self.assertIn(value, content)
        for forbidden in (
            "pip install",
            "conda create",
            "snapshot_download",
            "prepare_scannet",
            "run_camera_iteration.sh",
            'find "$RESULT_DIR"',
        ):
            self.assertNotIn(forbidden, content)

    def test_shell_syntax(self):
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        for path in AUTODL.glob("*.sh"):
            subprocess.run(
                [bash, "-n"],
                input=path.read_text(encoding="utf-8").replace("\r", "").encode(),
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
