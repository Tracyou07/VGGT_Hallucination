from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


from pre_experiments.camera_refiner_data_construction.scannet_adaptation import (
    assign_scene_roles,
    build_candidate_manifest,
    build_final_manifest,
    processed_scene_frame_count,
)


ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = (
    ROOT
    / "scripts"
    / "autodl"
    / "camera_refiner_data_construction"
    / "prepare_scannet_adaptation200.sh"
)
WINDOWS_GIT_BASH = Path(r"C:\Program Files\Git\usr\bin\bash.exe")


def _resolve_bash() -> str | None:
    if WINDOWS_GIT_BASH.is_file():
        return str(WINDOWS_GIT_BASH)
    return shutil.which("bash")


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    windows_temp = Path(tempfile.gettempdir()).resolve()
    try:
        return (Path("/tmp") / resolved.relative_to(windows_temp)).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_pose(path: Path, *, finite: bool = True) -> None:
    values = [float(index) for index in range(16)]
    if not finite:
        values[-1] = math.inf
    path.write_text(" ".join(str(value) for value in values), encoding="utf-8")


class ScanNetAdaptationManifestTest(unittest.TestCase):
    def test_candidates_are_deterministic_unique_and_exclude_protected_scenes(self):
        train = ["scene0004_00", "scene0002_00", "scene0001_00", "scene0003_00"]
        first = build_candidate_manifest(
            train,
            excluded_scenes=["scene0002_00"],
            seed=33,
            role_counts={"refiner_train": 1, "validation": 1, "selector_train": 1},
            min_frames=500,
        )
        second = build_candidate_manifest(
            reversed(train),
            excluded_scenes=["scene0002_00"],
            seed=33,
            role_counts={"refiner_train": 1, "validation": 1, "selector_train": 1},
            min_frames=500,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["candidate_scenes"]), 3)
        self.assertNotIn("scene0002_00", first["candidate_scenes"])
        self.assertEqual(first["target_scene_count"], 3)
        self.assertEqual(len(first["manifest_digest"]), 64)

    def test_duplicate_or_invalid_official_scene_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_candidate_manifest(
                ["scene0001_00", "scene0001_00"],
                excluded_scenes=[],
                seed=33,
                role_counts={"refiner_train": 1},
                min_frames=500,
            )
        with self.assertRaisesRegex(ValueError, "invalid"):
            build_candidate_manifest(
                ["not-a-scene"],
                excluded_scenes=[],
                seed=33,
                role_counts={"refiner_train": 1},
                min_frames=500,
            )

    def test_roles_are_disjoint_and_follow_the_frozen_acceptance_order(self):
        roles = assign_scene_roles(
            ["scene0001_00", "scene0002_00", "scene0003_00", "scene0004_00"],
            {"refiner_train": 2, "validation": 1, "selector_train": 1},
        )
        self.assertEqual(roles["refiner_train"], ["scene0001_00", "scene0002_00"])
        self.assertEqual(roles["validation"], ["scene0003_00"])
        self.assertEqual(roles["selector_train"], ["scene0004_00"])


class ProcessedSceneValidationTest(unittest.TestCase):
    def test_only_nonempty_images_with_matching_finite_poses_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "scene0001_00"
            color = scene / "color"
            pose = scene / "pose"
            color.mkdir(parents=True)
            pose.mkdir()

            (color / "0.jpg").write_bytes(b"image")
            _write_pose(pose / "0.txt")
            (color / "1.jpg").write_bytes(b"image")
            _write_pose(pose / "1.txt", finite=False)
            (color / "2.jpg").write_bytes(b"")
            _write_pose(pose / "2.txt")
            (color / "3.jpg").write_bytes(b"image")
            (pose / "unmatched.txt").write_text("0 " * 16, encoding="utf-8")

            self.assertEqual(processed_scene_frame_count(scene), 1)

    def test_final_manifest_requires_exact_quota_and_validates_every_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process_root = root / "process_scannet"
            scenes = ["scene0001_00", "scene0002_00", "scene0003_00"]
            candidate = build_candidate_manifest(
                scenes,
                excluded_scenes=[],
                seed=33,
                role_counts={"refiner_train": 1, "validation": 1, "selector_train": 1},
                min_frames=2,
            )
            accepted = list(candidate["candidate_scenes"])
            for scene_name in accepted:
                scene = process_root / scene_name
                (scene / "color").mkdir(parents=True)
                (scene / "pose").mkdir()
                for frame in range(2):
                    (scene / "color" / f"{frame}.jpg").write_bytes(b"image")
                    _write_pose(scene / "pose" / f"{frame}.txt")

            final = build_final_manifest(candidate, accepted, process_root)
            self.assertEqual(final["scene_count"], 3)
            self.assertEqual(
                {role: len(values) for role, values in final["scene_roles"].items()},
                {"refiner_train": 1, "validation": 1, "selector_train": 1},
            )
            self.assertEqual(len(final["dataset_digest"]), 64)
            json.dumps(final)

            with self.assertRaisesRegex(ValueError, "exactly 3"):
                build_final_manifest(candidate, accepted[:2], process_root)


class ScanNetAdaptationRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bash = _resolve_bash()
        if self.bash is None:
            self.skipTest("bash is required for the AutoDL runner test")
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.official = self.root / "official.txt"
        self.excluded = self.root / "excluded.txt"
        self.official.write_text(
            "scene0001_00\nscene0002_00\nscene0003_00\nscene0004_00\n",
            encoding="utf-8",
        )
        self.excluded.write_text("scene0004_00\n", encoding="utf-8")
        self.curl_log = self.root / "curl.log"
        self.extract_log = self.root / "extract.log"
        self.fake_curl = self.root / "fake_curl.py"
        self.fake_curl.write_text(
            textwrap.dedent(
                """\
                import os
                from pathlib import Path
                import sys

                if os.environ.get("FAKE_CURL_MUST_NOT_RUN") == "1":
                    raise SystemExit("curl must not run while resuming a complete split")
                args = sys.argv[1:]
                output = Path(args[args.index("-o") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"synthetic sens")
                with Path(os.environ["FAKE_CURL_LOG"]).open("a", encoding="utf-8") as handle:
                    handle.write(f"{output}\\n")
                """
            ),
            encoding="utf-8",
        )
        conda_sh = self.root / "conda" / "etc" / "profile.d" / "conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text(
            textwrap.dedent(
                """\
                conda() {
                  return 0
                }

                flock() {
                  return 0
                }

                python() {
                  if [[ "${1:-}" == *extract_scannet_sens.py ]]; then
                    local out_dir=""
                    local scene_list=""
                    while (( $# )); do
                      case "$1" in
                        --out-dir)
                          out_dir="$2"
                          shift 2
                          ;;
                        --scene-list)
                          scene_list="$2"
                          shift 2
                          ;;
                        *)
                          shift
                          ;;
                      esac
                    done
                    local scene
                    scene="$(head -n 1 "$scene_list" | tr -d '\\r')"
                    mkdir -p "$out_dir/$scene/color" "$out_dir/$scene/pose"
                    for frame in 0 1; do
                      printf 'image\n' > "$out_dir/$scene/color/$frame.jpg"
                      printf '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n' \\
                        > "$out_dir/$scene/pose/$frame.txt"
                    done
                    printf '%s\n' "$scene" >> "$TEST_EXTRACT_LOG"
                    return 0
                  fi
                  command "$TEST_REAL_PYTHON" "$@" | tr -d '\\r'
                }
                """
            ),
            encoding="utf-8",
        )
        self.process_root = self.root / "scannet" / "process_scannet"
        self.state_root = self.root / "scannet" / "adaptation200_state"
        self.manifest = self.root / "results" / "adaptation200" / "manifest.json"
        self.env = os.environ.copy()
        self.env.update(
            {
                "SCANNET_TOS_ACCEPTED": "1",
                "AUTODL_TMP": _bash_path(self.root),
                "SCANNET_ROOT": _bash_path(self.root / "scannet"),
                "PROCESS_DIR": _bash_path(self.process_root),
                "STATE_DIR": _bash_path(self.state_root),
                "RESULT_DIR": _bash_path(self.manifest.parent),
                "FINAL_MANIFEST": _bash_path(self.manifest),
                "OFFICIAL_TRAIN": _bash_path(self.official),
                "EXCLUDED_SCENES": _bash_path(self.excluded),
                "TARGET_SCENES": "3",
                "REFINER_TRAIN_SCENES": "1",
                "VALIDATION_SCENES": "1",
                "SELECTOR_TRAIN_SCENES": "1",
                "MIN_MATCHING_FRAMES": "2",
                "MIN_FREE_GIB": "1",
                "DOWNLOAD_RETRIES": "1",
                "CONDA_ROOT": _bash_path(self.root / "conda"),
                "CONDA_ENV_NAME": "test",
                "SCANNET_CURL": _bash_path(Path(sys.executable)),
                "SCANNET_CURL_ARGS": _bash_path(self.fake_curl),
                "TEST_REAL_PYTHON": _bash_path(Path(sys.executable)),
                "TEST_EXTRACT_LOG": _bash_path(self.extract_log),
                "FAKE_CURL_LOG": str(self.curl_log),
            }
        )

    def run_prepare(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(PREPARE_SCRIPT)],
            cwd=ROOT,
            env=self.env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )

    def test_serial_pipeline_finalizes_and_resumes_without_redownloading(self):
        first = self.run_prepare()
        self.assertEqual(first.returncode, 0, f"{first.stdout}\n{first.stderr}")
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["scene_count"], 3)
        self.assertEqual(
            {role: len(scenes) for role, scenes in manifest["scene_roles"].items()},
            {"refiner_train": 1, "validation": 1, "selector_train": 1},
        )
        self.assertEqual(len(self.curl_log.read_text(encoding="utf-8").splitlines()), 3)
        self.assertEqual(len(self.extract_log.read_text(encoding="utf-8").splitlines()), 3)
        self.assertEqual(list(self.state_root.glob("raw_sens/scans/**/*.sens")), [])

        accepted = manifest["scenes"][0]["scene"]
        stale_sens = self.state_root / "raw_sens" / "scans" / accepted / f"{accepted}.sens"
        stale_sens.parent.mkdir(parents=True)
        stale_sens.write_bytes(b"validated before interruption")
        self.env["FAKE_CURL_MUST_NOT_RUN"] = "1"

        resumed = self.run_prepare()
        self.assertEqual(resumed.returncode, 0, f"{resumed.stdout}\n{resumed.stderr}")
        self.assertFalse(stale_sens.exists())
        self.assertEqual(len(self.curl_log.read_text(encoding="utf-8").splitlines()), 3)

        accepted_state = self.state_root / "accepted_scenes.txt"
        accepted_lines = accepted_state.read_text(encoding="utf-8").splitlines()
        reused_scene = accepted_lines[-1]
        accepted_state.write_bytes(
            ("\n".join(accepted_lines[:-1]) + "\n").encode("utf-8")
        )
        reused_sens = (
            self.state_root
            / "raw_sens"
            / "scans"
            / reused_scene
            / f"{reused_scene}.sens"
        )
        reused_sens.parent.mkdir(parents=True)
        reused_sens.write_bytes(b"extracted before interruption")

        reused = self.run_prepare()
        self.assertEqual(reused.returncode, 0, f"{reused.stdout}\n{reused.stderr}")
        self.assertFalse(reused_sens.exists())
        self.assertEqual(len(self.curl_log.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
