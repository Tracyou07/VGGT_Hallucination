import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl"
WINDOWS_GIT_BASH = Path(r"C:\Program Files\Git\usr\bin\bash.exe")
PREPARE_SCANNET = AUTODL / "prepare_scannet50.sh"


def resolve_bash(
    windows_git_bash: Path = WINDOWS_GIT_BASH,
    path_lookup=shutil.which,
) -> str | None:
    if windows_git_bash.is_file():
        return str(windows_git_bash)
    return path_lookup("bash")


BASH = resolve_bash()


def bash_path(path: Path) -> str:
    resolved = path.resolve()
    windows_temp = Path(tempfile.gettempdir()).resolve()
    try:
        return (Path("/tmp") / resolved.relative_to(windows_temp)).as_posix()
    except ValueError:
        return resolved.as_posix()


class ScanNetDownloadBehaviorTest(unittest.TestCase):
    scene = "scene0000_00"

    def setUp(self) -> None:
        if BASH is None:
            self.skipTest("bash is required for ScanNet download behavior tests")
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.scannet_root = self.temp_dir / "scannet"
        self.raw_download_root = self.scannet_root / "raw_sens"
        self.gt_download_root = self.scannet_root / "raw"
        self.scene_list = self.temp_dir / "scenes.txt"
        self.scene_list.write_text(f"{self.scene}\n", encoding="utf-8")
        self.downloader_state = self.temp_dir / "downloader-state.tsv"
        self.curl_state = self.temp_dir / "curl-state.tsv"
        self.extract_log = self.temp_dir / "extract-raw-dir.txt"
        self.fake_downloader = self.temp_dir / "fake-download-scannet.py"
        self.fake_downloader.write_text(
            textwrap.dedent(
                """\
                import os
                from pathlib import Path
                import sys
                import tempfile

                args = sys.argv[1:]

                def option(name):
                    return args[args.index(name) + 1]

                output_root = Path(option("-o"))
                scene = option("--id")
                file_type = option("--type")
                target = output_root / "scans" / scene / f"{scene}{file_type}"
                state = Path(os.environ["FAKE_DOWNLOADER_STATE"])
                previous = state.read_text(encoding="utf-8").splitlines() if state.exists() else []
                attempt = len(previous) + 1
                state.write_text(
                    "\\n".join([*previous, f"{attempt}\\t{output_root}"]) + "\\n",
                    encoding="utf-8",
                )

                mode = os.environ["FAKE_DOWNLOADER_MODE"]
                if mode == "must_not_run":
                    raise SystemExit("downloader must not be invoked")
                if mode == "skip_existing" and target.exists():
                    raise SystemExit(0)

                target.parent.mkdir(parents=True, exist_ok=True)
                if mode == "fail_once_random_then_succeed" and attempt == 1:
                    handle, partial = tempfile.mkstemp(
                        prefix="random-", suffix=".part", dir=target.parent
                    )
                    os.write(handle, b"partial")
                    os.close(handle)
                    raise SystemExit(23)

                target.write_bytes(b"complete")
                """
            ),
            encoding="utf-8",
        )
        self.fake_curl = self.temp_dir / "fake-curl.py"
        self.fake_curl.write_text(
            textwrap.dedent(
                """\
                from pathlib import Path
                import os
                import sys

                args = sys.argv[1:]
                output = Path(args[args.index("-o") + 1])
                url = args[-1]
                state = Path(os.environ["FAKE_CURL_STATE"])
                previous = state.read_text(encoding="utf-8").splitlines() if state.exists() else []
                attempt = len(previous) + 1
                state.write_text(
                    "\\n".join([*previous, f"{attempt}\\t{output}\\t{url}"]) + "\\n",
                    encoding="utf-8",
                )

                mode = os.environ["FAKE_CURL_MODE"]
                if mode == "must_not_run":
                    raise SystemExit("curl must not be invoked")
                output.parent.mkdir(parents=True, exist_ok=True)
                if mode == "resume_after_interrupt" and attempt == 1:
                    output.write_bytes(b"partial")
                    raise SystemExit(18)
                if mode == "resume_after_interrupt":
                    if "-C" not in args or args[args.index("-C") + 1] != "-":
                        raise SystemExit("resume flag missing")
                    if output.read_bytes() != b"partial":
                        raise SystemExit("partial download was not preserved")
                    with output.open("ab") as handle:
                        handle.write(b"-complete")
                else:
                    output.write_bytes(b"complete")
                """
            ),
            encoding="utf-8",
        )

        conda_sh = self.temp_dir / "conda" / "etc" / "profile.d" / "conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text(
            textwrap.dedent(
                """\
                conda() {
                  return 0
                }

                python() {
                  if [[ "${1:-}" == *extract_scannet_sens.py ]]; then
                    local raw_dir=""
                    local out_dir=""
                    while (( $# )); do
                      case "$1" in
                        --raw-dir)
                          raw_dir="$2"
                          shift 2
                          ;;
                        --out-dir)
                          out_dir="$2"
                          shift 2
                          ;;
                        *)
                          shift
                          ;;
                      esac
                    done
                    printf '%s\\n' "$raw_dir" > "$TEST_EXTRACT_LOG"
                    mkdir -p "$out_dir/$TEST_SCENE/color" "$out_dir/$TEST_SCENE/pose"
                    printf 'image\\n' > "$out_dir/$TEST_SCENE/color/000000.jpg"
                    printf 'pose\\n' > "$out_dir/$TEST_SCENE/pose/000000.txt"
                    return 0
                  fi
                  if [[ "${1:-}" == "-" ]]; then
                    if [[ "${2:-}" != "$TEST_SCENE_LIST" ]]; then
                      cat >/dev/null
                      return 0
                    fi
                    command "$TEST_REAL_PYTHON" "$@" | tr -d '\\r'
                    return
                  fi
                  command "$TEST_REAL_PYTHON" "$@"
                }
                """
            ),
            encoding="utf-8",
        )

        self.env = os.environ.copy()
        self.env.update(
            {
                "SCANNET_TOS_ACCEPTED": "1",
                "SCANNET_ROOT": bash_path(self.scannet_root),
                "RAW_DOWNLOAD_ROOT": bash_path(self.raw_download_root),
                "GT_DOWNLOAD_ROOT": bash_path(self.gt_download_root),
                "PROCESS_DIR": bash_path(self.scannet_root / "process_scannet"),
                "SCENE_LIST": bash_path(self.scene_list),
                "SCENE_LIMIT": "1",
                "DOWNLOAD_RETRIES": "1",
                "DOWNLOAD_GT_PLY": "0",
                "SCANNET_DOWNLOAD_SCRIPT": bash_path(self.fake_downloader),
                "CONDA_ROOT": bash_path(self.temp_dir / "conda"),
                "CONDA_ENV_NAME": "test",
                "TEST_REAL_PYTHON": bash_path(Path(sys.executable)),
                "TEST_SCENE": self.scene,
                "TEST_SCENE_LIST": bash_path(self.scene_list),
                "TEST_EXTRACT_LOG": bash_path(self.extract_log),
                "FAKE_DOWNLOADER_STATE": str(self.downloader_state),
                "FAKE_DOWNLOADER_MODE": "success",
                "FAKE_CURL_STATE": str(self.curl_state),
                "FAKE_CURL_MODE": "success",
                "SCANNET_CURL": bash_path(Path(sys.executable)),
                "SCANNET_CURL_ARGS": bash_path(self.fake_curl),
            }
        )

    def run_prepare(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BASH), str(PREPARE_SCANNET)],
            cwd=ROOT,
            env=self.env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def assert_success(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def downloader_invocations(self) -> list[tuple[int, Path]]:
        if not self.downloader_state.exists():
            return []
        return [
            (int(attempt), Path(output_root))
            for attempt, output_root in (
                line.split("\t")
                for line in self.downloader_state.read_text(encoding="utf-8").splitlines()
            )
        ]

    def curl_invocations(self) -> list[list[str]]:
        if not self.curl_state.exists():
            return []
        return [
            line.split("\t")
            for line in self.curl_state.read_text(encoding="utf-8").splitlines()
        ]

    def test_interrupted_sens_download_resumes_the_same_partial_file(self):
        destination = (
            self.raw_download_root / "scans" / self.scene / f"{self.scene}.sens"
        )
        partial = destination.with_name(f"{destination.name}.partial")
        self.env["DOWNLOAD_RETRIES"] = "1"
        self.env["FAKE_DOWNLOADER_MODE"] = "must_not_run"
        self.env["FAKE_CURL_MODE"] = "resume_after_interrupt"

        interrupted = self.run_prepare()

        self.assertNotEqual(interrupted.returncode, 0)
        self.assertEqual(partial.read_bytes(), b"partial")
        resumed = self.run_prepare()
        self.assert_success(resumed)
        self.assertEqual(destination.read_bytes(), b"partial-complete")
        self.assertFalse(partial.exists())
        invocations = self.curl_invocations()
        self.assertEqual([int(row[0]) for row in invocations], [1, 2])
        self.assertEqual(len({row[1] for row in invocations}), 1)
        self.assertTrue(
            invocations[0][2].endswith(
                f"/scannet/v1/scans/{self.scene}/{self.scene}.sens"
            )
        )

    def test_zero_byte_destination_gets_one_real_attempt(self):
        destination = (
            self.raw_download_root / "scans" / self.scene / f"{self.scene}.sens"
        )
        destination.parent.mkdir(parents=True)
        destination.touch()
        self.env["FAKE_DOWNLOADER_MODE"] = "skip_existing"

        result = self.run_prepare()

        self.assert_success(result)
        self.assertEqual(destination.read_bytes(), b"complete")
        self.assertEqual(len(self.curl_invocations()), 1)
        self.assertEqual(self.downloader_invocations(), [])

    def test_legacy_staging_is_removed_before_resumable_download(self):
        destination = (
            self.raw_download_root / "scans" / self.scene / f"{self.scene}.sens"
        )
        stale_staging = (
            destination.parent
            / f".scannet-download-{destination.name}.staging.stale"
        )
        stale_staging.mkdir(parents=True)
        (stale_staging / "random.part").write_bytes(b"obsolete")

        result = self.run_prepare()

        self.assert_success(result)
        self.assertEqual(destination.read_bytes(), b"complete")
        self.assertFalse(stale_staging.exists())
        self.assertEqual(len(self.curl_invocations()), 1)
        self.assertEqual(self.downloader_invocations(), [])

    def test_valid_destination_is_reused_without_downloader(self):
        destination = (
            self.raw_download_root / "scans" / self.scene / f"{self.scene}.sens"
        )
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing")
        stale_staging = (
            destination.parent
            / f".scannet-download-{destination.name}.staging.stale"
        )
        stale_staging.mkdir()
        (stale_staging / "random.part").write_bytes(b"partial")
        self.env["FAKE_DOWNLOADER_MODE"] = "must_not_run"

        result = self.run_prepare()

        self.assert_success(result)
        self.assertEqual(destination.read_bytes(), b"existing")
        self.assertFalse(stale_staging.exists())
        self.assertEqual(self.downloader_invocations(), [])

    def test_custom_raw_dir_wins_and_remains_extractor_input(self):
        custom_raw_dir = self.temp_dir / "arbitrary-sensor-cache"
        self.env["RAW_DIR"] = bash_path(custom_raw_dir)
        self.env["DOWNLOAD_GT_PLY"] = "1"

        result = self.run_prepare()

        self.assert_success(result)
        self.assertEqual(
            (custom_raw_dir / self.scene / f"{self.scene}.sens").read_bytes(),
            b"complete",
        )
        self.assertFalse(
            (self.raw_download_root / "scans" / self.scene / f"{self.scene}.sens").exists()
        )
        self.assertEqual(
            (
                self.gt_download_root
                / "scans"
                / self.scene
                / f"{self.scene}_vh_clean_2.ply"
            ).read_bytes(),
            b"complete",
        )
        self.assertEqual(
            self.extract_log.read_text(encoding="utf-8").strip(),
            bash_path(custom_raw_dir),
        )


class AutoDLScriptsTest(unittest.TestCase):
    def read(self, name: str) -> str:
        return (AUTODL / name).read_text(encoding="utf-8")

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

    def test_scannet_setup_downloads_validated_assets_with_bounded_retries(self):
        content = self.read("prepare_scannet50.sh")
        for value in (
            "SCANNET_TOS_ACCEPTED",
            "http://kaldir.vc.cit.tum.de/scannet/v1/scans",
            "http://kaldir.vc.cit.tum.de/scannet/v2/scans",
            'SCANNET_CURL="${SCANNET_CURL:-curl}"',
            'SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/fastvggt_scannet50.txt}"',
            'SCENE_LIMIT="${SCENE_LIMIT:-0}"',
            'DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"',
            'DOWNLOAD_GT_PLY="${DOWNLOAD_GT_PLY:-0}"',
            'GT_DOWNLOAD_ROOT="${GT_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw}"',
            '[[ "$DOWNLOAD_RETRIES" =~ ^[1-9][0-9]*$ ]]',
            '[[ "$DOWNLOAD_GT_PLY" == "0" || "$DOWNLOAD_GT_PLY" == "1" ]]',
            "read_scene_list",
            "re.fullmatch(r\"scene[0-9]{4}_[0-9]{2}\", scene)",
            "len(scenes) != len(set(scenes))",
            "Selected scene list is empty",
            'source "$SCRIPT_DIR/scannet_download.sh"',
            'sens="$RAW_DIR/$scene/$scene.sens"',
            'gt_ply="$GT_DOWNLOAD_ROOT/scans/$scene/${scene}_vh_clean_2.ply"',
            'download_asset "$scene" .sens "$sens"',
            'download_asset "$scene" _vh_clean_2.ply "$gt_ply"',
            "extract_scannet_sens.py",
            "missing_processed_scenes",
        ):
            self.assertIn(value, content)
        download_helper = self.read("scannet_download.sh")
        for value in (
            'partial="$expected.partial"',
            '"${curl_command[@]}" -fL -C -',
            "Partial file retained for the next run",
        ):
            self.assertIn(value, download_helper)
        for forbidden in ("export_depth", "download_vggt_weights", "conda create", "pip install", "snapshot_download", "find \"$raw_download_root\"", "cp \"$found\""):
            self.assertNotIn(forbidden, content.lower())
        self.assertLess(
            content.index("SCANNET_TOS_ACCEPTED"),
            content.index("http://kaldir"),
        )

    def test_scannet_setup_keeps_raw_dir_as_destination_and_extractor_input(self):
        content = self.read("prepare_scannet50.sh")
        for value in (
            'RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"',
            'sens="$RAW_DIR/$scene/$scene.sens"',
            '--raw-dir "$RAW_DIR"',
        ):
            self.assertIn(value, content)
        for forbidden in (
            '[[ "$(basename "$RAW_DIR")" == "scans" ]]',
            'RAW_DOWNLOAD_ROOT="$RAW_DIR_DOWNLOAD_ROOT"',
            "RAW_DOWNLOAD_ROOT and RAW_DIR must identify the same scans root.",
        ):
            self.assertNotIn(forbidden, content)

    def test_retired_setup_and_round_runners_are_absent(self):
        for path in (
            AUTODL / "setup_vggt_env.sh",
            AUTODL / "download_vggt_weights.sh",
            AUTODL / "run_camera_iteration.sh",
            AUTODL / "run_camera_context.sh",
            AUTODL / "run_camera_head_amplification.sh",
        ):
            self.assertFalse(path.exists(), path)

    def test_shell_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            fallback = "/usr/bin/bash"
            self.assertEqual(
                resolve_bash(Path(directory) / "missing-bash", lambda _: fallback),
                fallback,
            )
        bash = BASH
        self.assertIsNotNone(bash)
        for path in AUTODL.glob("*.sh"):
            subprocess.run(
                [bash, "-n"],
                input=path.read_text(encoding="utf-8").replace("\r", "").encode(),
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
