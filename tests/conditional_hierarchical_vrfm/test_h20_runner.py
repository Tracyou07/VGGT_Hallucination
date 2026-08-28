from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_privileged_conditional_hvrfm_teacher_lift.sh"
WINDOWS_GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(WINDOWS_GIT_BASH) if WINDOWS_GIT_BASH.is_file() else shutil.which("bash")


def bash_path(path: Path) -> str:
    resolved = path.resolve()
    windows_temp = Path(tempfile.gettempdir()).resolve()
    try:
        return (Path("/tmp") / resolved.relative_to(windows_temp)).as_posix()
    except ValueError:
        return resolved.as_posix()


class H20RunnerBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        if BASH is None:
            self.skipTest("bash is required for H20 runner behavior tests")
        self.temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)
        self.fake_bin = self.temporary / "bin"
        self.fake_bin.mkdir()
        self.repo = self.temporary / "worktree"
        self.repo.mkdir()
        (self.repo / ".git").write_text("gitdir: fixture\n", encoding="utf-8")
        self.source = self.temporary / "source"
        self.formal = self.temporary / "formal"
        self.prepared = self.temporary / "prepared"
        self.checkpoint = self.temporary / "checkpoint"
        self.scannet = self.temporary / "scannet"
        for directory in (
            self.source,
            self.formal,
            self.prepared,
            self.checkpoint,
            self.scannet,
        ):
            directory.mkdir()
        (self.source / "verified_completion.json").write_text("{}\n", encoding="utf-8")
        (self.formal / "verified_completion.json").write_text("{}\n", encoding="utf-8")
        (self.checkpoint / "model.safetensors").write_bytes(b"checkpoint")
        (self.scannet / "verified_completion.json").write_text("{}\n", encoding="utf-8")
        self.python_log = self.temporary / "python.tsv"
        self.du_count = self.temporary / "du-count.txt"
        self.fake_python = self.temporary / "python"
        self._install_fakes()
        self.base_env = os.environ.copy()
        self.base_env.update(
            {
                "PATH": f"{bash_path(self.fake_bin)}:/usr/bin:/bin",
                "REPO_ROOT": bash_path(self.repo),
                "SOURCE_RUN": bash_path(self.source),
                "FORMAL_LABEL_ROOT": bash_path(self.formal),
                "PREPARED_ROOT": bash_path(self.prepared),
                "CHECKPOINT_DIR": bash_path(self.checkpoint),
                "SCANNET_MARKER": bash_path(self.scannet / "verified_completion.json"),
                "PYTHON": bash_path(self.fake_python),
                "RUN_ID": "fixture_run",
                "FIXTURE_PYTHON_LOG": bash_path(self.python_log),
                "FIXTURE_DU_COUNT": bash_path(self.du_count),
                "FIXTURE_GIT_COMMIT": "1" * 40,
                "FIXTURE_GPU_ROWS": "2, NVIDIA H20\n",
                "FIXTURE_CREATE_COMPLETION": "1",
            }
        )

    def _write_executable(self, name: str, content: str) -> Path:
        path = self.fake_bin / name
        path.write_text(
            textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n"
        )
        path.chmod(0o755)
        return path

    def _install_fakes(self) -> None:
        self._write_executable(
            "hostname",
            """\
            #!/usr/bin/env bash
            printf '%s\n' "${FIXTURE_HOSTNAME:?}"
            """,
        )
        self._write_executable(
            "id",
            """\
            #!/usr/bin/env bash
            [[ "${1:-}" == "-un" ]] || exit 90
            printf '%s\n' "${FIXTURE_USER:?}"
            """,
        )
        self._write_executable(
            "df",
            """\
            #!/usr/bin/env bash
            printf 'Avail\n%sG\n' "${FIXTURE_FREE_GIB:?}"
            """,
        )
        self._write_executable(
            "git",
            """\
            #!/usr/bin/env bash
            case "$*" in
              *"branch --show-current"*) printf '%s\n' "${FIXTURE_BRANCH:?}" ;;
              *"status --short"*) printf '%b' "${FIXTURE_DIRTY_STATUS:-}" ;;
              *"rev-parse HEAD"*) printf '%s\n' "${FIXTURE_GIT_COMMIT:?}" ;;
              *) printf 'unexpected git command: %s\n' "$*" >&2; exit 91 ;;
            esac
            """,
        )
        self._write_executable(
            "nvidia-smi",
            """\
            #!/usr/bin/env bash
            if [[ "$*" == *"--query-gpu=index,name"* ]]; then
              printf '%b' "${FIXTURE_GPU_ROWS:?}"
              exit 0
            fi
            if [[ "$*" == *"--query-compute-apps=pid"* ]]; then
              gpu=""
              while (( $# )); do
                case "$1" in
                  -i|--id) gpu="$2"; shift 2 ;;
                  *) shift ;;
                esac
              done
              variable="FIXTURE_GPU_${gpu}_PIDS"
              printf '%b' "${!variable:-}"
              exit 0
            fi
            printf 'unexpected nvidia-smi command\n' >&2
            exit 92
            """,
        )
        self._write_executable(
            "du",
            """\
            #!/usr/bin/env bash
            count=0
            [[ ! -f "$FIXTURE_DU_COUNT" ]] || count="$(cat "$FIXTURE_DU_COUNT")"
            printf '%s\n' "$((count + 1))" > "$FIXTURE_DU_COUNT"
            IFS=',' read -r -a values <<< "${FIXTURE_DU_KIB_SEQUENCE:-0}"
            index="$count"
            (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
            printf '%s\t%s\n' "${values[index]}" "${@: -1}"
            """,
        )
        self._write_executable(
            "flock",
            """\
            #!/usr/bin/env bash
            exit "${FIXTURE_FLOCK_RC:-0}"
            """,
        )
        self.fake_python.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                stage=""
                run_root=""
                previous=""
                for argument in "$@"; do
                  case "$argument" in
                    preflight|prepare|smoke|calibration|report|verify) stage="$argument" ;;
                  esac
                  if [[ "$previous" == "--run-root" ]]; then run_root="$argument"; fi
                  previous="$argument"
                done
                printf '%s\t%s\t%s\n' "${CUDA_VISIBLE_DEVICES:-}" "$stage" "$*" >> "$FIXTURE_PYTHON_LOG"
                if [[ "${FIXTURE_STDERR_STAGE:-}" == "$stage" ]]; then
                  printf 'fixture stderr for %s\n' "$stage" >&2
                fi
                if [[ "${FIXTURE_FAIL_STAGE:-}" == "$stage" ]]; then exit 47; fi
                if [[ "$stage" == "verify" && "${FIXTURE_CREATE_COMPLETION:-0}" == "1" ]]; then
                  printf '{}\n' > "$run_root/verified_completion.json"
                fi
                """
            ).lstrip(),
            encoding="utf-8",
            newline="\n",
        )
        self.fake_python.chmod(0o755)

    def run_runner_fixture(
        self,
        *,
        hostname: str = "VM-0-11-ubuntu",
        user: str = "ubuntu",
        free_gib: int = 150,
        gpu_name: str = "NVIDIA H20",
        gpu_compute_pids: str = "",
        branch: str = "codex/privileged-conditional-hvrfm",
        dirty_status: str = "",
        arguments: tuple[str, ...] = (),
        gpu_rows: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.base_env.copy()
        env.update(
            {
                "FIXTURE_HOSTNAME": hostname,
                "FIXTURE_USER": user,
                "FIXTURE_FREE_GIB": str(free_gib),
                "FIXTURE_BRANCH": branch,
                "FIXTURE_DIRTY_STATUS": dirty_status,
                "FIXTURE_GPU_ROWS": gpu_rows or f"2, {gpu_name}\n",
                "FIXTURE_GPU_2_PIDS": gpu_compute_pids,
            }
        )
        if environment:
            env.update(environment)
        return subprocess.run(
            [
                str(BASH),
                "-c",
                'fake_bin="$1"; runner="$2"; shift 2; export PATH="$fake_bin:$PATH"; source "$runner" "$@"',
                "runner-fixture",
                bash_path(self.fake_bin),
                bash_path(RUNNER),
                *arguments,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=60,
        )

    def python_stages(self) -> list[tuple[str, str]]:
        if not self.python_log.exists():
            return []
        return [
            (line.split("\t", 2)[0], line.split("\t", 2)[1])
            for line in self.python_log.read_text(encoding="utf-8").splitlines()
        ]

    def test_preflight_only_succeeds_with_controlled_h20_facts(self) -> None:
        result = self.run_runner_fixture(arguments=("--preflight-only",))

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["result_root"],
            "/data/yjh/output/vggt/privileged_conditional_hvrfm",
        )
        self.assertEqual(
            payload["planned_stages"],
            ["preflight", "prepare", "smoke", "calibration", "report", "verify"],
        )
        self.assertEqual(payload["gpu_index"], "2")
        self.assertEqual(self.python_stages(), [])

    def test_preflight_only_creates_no_result_or_run_directory(self) -> None:
        result_root = self.temporary / "results"

        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"RESULT_ROOT": bash_path(result_root)},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result_root.exists())
        self.assertEqual(self.python_stages(), [])

    def test_preflight_only_fails_before_compute_for_busy_gpu(self) -> None:
        result = self.run_runner_fixture(
            gpu_compute_pids="8123\n",
            arguments=("--preflight-only",),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("active compute process", result.stderr)
        self.assertEqual(self.python_stages(), [])

    def test_complete_run_selects_an_idle_h20_and_runs_six_ordered_stages(self) -> None:
        result_root = self.temporary / "results"
        result = self.run_runner_fixture(
            gpu_rows="0, NVIDIA A100\n2, NVIDIA H20\n3, NVIDIA H20\n",
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_GPU_2_PIDS": "7002\n",
                "FIXTURE_GPU_3_PIDS": "",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.python_stages(),
            [
                ("3", "preflight"),
                ("3", "prepare"),
                ("3", "smoke"),
                ("3", "calibration"),
                ("3", "report"),
                ("3", "verify"),
            ],
        )
        run_root = result_root / "fixture_run"
        self.assertTrue((run_root / "verified_completion.json").is_file())
        log_root = result_root / ".runner_control" / "fixture_run" / "logs"
        for stage in ("preflight", "prepare", "smoke", "calibration", "report", "verify"):
            self.assertTrue((log_root / f"{stage}.out.log").is_file())
            self.assertTrue((log_root / f"{stage}.err.log").is_file())

    def test_identity_git_and_space_gates_fail_before_python(self) -> None:
        cases = (
            ({"hostname": "other-host"}, "host identity"),
            ({"user": "root"}, "user identity"),
            ({"branch": "main"}, "wrong branch"),
            ({"dirty_status": " M changed.py\n"}, "dirty worktree"),
            ({"free_gib": 99}, "100 GiB"),
        )
        for kwargs, message in cases:
            with self.subTest(message=message):
                result = self.run_runner_fixture(arguments=("--preflight-only",), **kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message.lower(), result.stderr.lower())
                self.assertEqual(self.python_stages(), [])

    def test_nonempty_stage_stderr_stops_and_preserves_the_run(self) -> None:
        result_root = self.temporary / "results"
        result = self.run_runner_fixture(
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_STDERR_STAGE": "smoke",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("smoke wrote stderr", result.stderr)
        self.assertEqual(
            [stage for _, stage in self.python_stages()],
            ["preflight", "prepare", "smoke"],
        )
        self.assertTrue(
            (
                result_root
                / ".runner_control"
                / "fixture_run"
                / "logs"
                / "smoke.err.log"
            ).is_file()
        )

    def test_run_size_is_checked_after_each_completed_stage(self) -> None:
        result_root = self.temporary / "results"
        result = self.run_runner_fixture(
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_DU_KIB_SEQUENCE": "0,20971520",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run root reached 20 GiB", result.stderr)
        self.assertEqual(
            [stage for _, stage in self.python_stages()],
            ["preflight", "prepare"],
        )

    def test_rejects_any_argument_other_than_preflight_only(self) -> None:
        result = self.run_runner_fixture(arguments=("--unknown",))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only optional argument", result.stderr)
        self.assertEqual(self.python_stages(), [])

    def test_runner_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            [str(BASH), "-n", str(RUNNER)],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
