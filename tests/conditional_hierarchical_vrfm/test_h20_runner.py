from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_privileged_conditional_hvrfm_teacher_lift.sh"
WINDOWS_GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(WINDOWS_GIT_BASH) if WINDOWS_GIT_BASH.is_file() else shutil.which("bash")
RUNNER_FIXTURE_TIMEOUT_SECONDS = 120

FORMAL_GIT = "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e"
VRFM_COMPLETION_DIGEST = (
    "3fdc97395eef8261ad7eaa055aec0bd441cf8d43fee9847464f190e269ab474e"
)


def bash_path(path: Path) -> str:
    resolved = path.resolve()
    windows_temp = Path(tempfile.gettempdir()).resolve()
    try:
        return (Path("/tmp") / resolved.relative_to(windows_temp)).as_posix()
    except ValueError:
        return resolved.as_posix()


def bash_lexical_path(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    windows_temp = Path(tempfile.gettempdir()).resolve()
    try:
        return (Path("/tmp") / absolute.relative_to(windows_temp)).as_posix()
    except ValueError:
        return absolute.as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        (self.checkpoint / "model.safetensors").write_bytes(b"checkpoint")
        (self.source / "manifests").mkdir()
        (self.formal / "manifests").mkdir()
        self._write_authenticated_inputs()
        self.python_log = self.temporary / "python.tsv"
        self.du_count = self.temporary / "du-count.txt"
        self.gpu_count = self.temporary / "gpu-count.txt"
        self.git_count_root = self.temporary / "git-count"
        self.gpu_lock_state = self.temporary / "gpu-lock-state"
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
                "FIXTURE_GPU_COUNT": bash_path(self.gpu_count),
                "FIXTURE_GIT_COUNT_ROOT": bash_path(self.git_count_root),
                "FIXTURE_GPU_LOCK_STATE": bash_path(self.gpu_lock_state),
                "FIXTURE_GIT_COMMIT": "1" * 40,
                "FIXTURE_GPU_ROWS": "2, NVIDIA H20\n",
                "FIXTURE_CREATE_COMPLETION": "1",
                "FIXTURE_REAL_PYTHON": Path(sys.executable).as_posix(),
                "FIXTURE_JSON_HELPER": (self.temporary / "json-helper.py").resolve().as_posix(),
                "FIXTURE_WINDOWS_TEMP": str(Path(tempfile.gettempdir()).resolve()),
                "HVRFM_RUNNER_TEST_MODE": "1",
                "HVRFM_TEST_SCANNET_MARKER_SHA256": sha256_file(
                    self.scannet / "verified_completion.json"
                ),
                "HVRFM_TEST_VRFM_MARKER_SHA256": sha256_file(
                    self.source / "verified_completion.json"
                ),
                "HVRFM_TEST_SOURCE_MANIFEST_SHA256": sha256_file(
                    self.source / "manifests" / "source_manifest.json"
                ),
                "HVRFM_TEST_FORMAL_MARKER_SHA256": sha256_file(
                    self.formal / "verified_completion.json"
                ),
                "HVRFM_TEST_FORMAL_DATA_MANIFEST_SHA256": sha256_file(
                    self.formal / "manifests" / "data_manifest.json"
                ),
                "HVRFM_TEST_CHECKPOINT_SHA256": sha256_file(
                    self.checkpoint / "model.safetensors"
                ),
            }
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_authenticated_inputs(self) -> None:
        source_manifest_path = self.source / "manifests" / "source_manifest.json"
        self._write_json(
            source_manifest_path,
            {
                "schema": "variational_camera_latent.source.v1",
                "dataset_root": str(self.source / "prediction_only" / "source"),
                "source_run_digest": "a" * 64,
                "records": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        source_manifest_sha = sha256_file(source_manifest_path)
        self._write_json(
            self.source / "verified_completion.json",
            {
                "schema": "variational_camera_latent.verified_completion.v1",
                "scene_count": 10,
                "overlap_count": 80,
                "candidate_count": 2560,
                "signal": "PROMISING",
                "prediction_manifest_sha256": "b" * 64,
                "privileged_manifest_sha256": "c" * 64,
                "report_sha256": "d" * 64,
                "completion_digest": VRFM_COMPLETION_DIGEST,
            },
        )
        checkpoint_sha = sha256_file(self.checkpoint / "model.safetensors")
        formal_manifest_path = self.formal / "manifests" / "data_manifest.json"
        self._write_json(
            formal_manifest_path,
            {
                "schema": "long_short_camera_head.data_manifest.v1",
                "git_revision": FORMAL_GIT,
                "source_run": bash_path(self.source),
                "source_manifest_sha256": source_manifest_sha,
                "prepared_root": bash_path(self.prepared),
                "checkpoint_dir": bash_path(self.checkpoint),
                "base_checkpoint_sha256": checkpoint_sha,
                "records": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        formal_manifest_sha = sha256_file(formal_manifest_path)
        self._write_json(
            self.formal / "verified_completion.json",
            {
                "schema": "long_short_camera_head.verified_completion.v1",
                "git_revision": FORMAL_GIT,
                "verifier_git_revision": FORMAL_GIT,
                "source_manifest_sha256": source_manifest_sha,
                "base_checkpoint_sha256": checkpoint_sha,
                "config_sha256": "e" * 64,
                "data_manifest_sha256": formal_manifest_sha,
                "test_evidence_sha256": "f" * 64,
                "stage_completion_sha256": {},
                "scene_count": 10,
                "train_scene_count": 8,
                "locked_replay_scene_count": 2,
                "classification": "NO_SOURCE_HEAD_SIGNAL",
                "report_sha256": "1" * 64,
                "artifacts": [],
                "inference_leakage_audit": True,
                "formal_protocol_sha256": "2" * 64,
            },
        )
        self._write_json(
            self.scannet / "verified_completion.json",
            {
                "schema": "camera_solution_space_01.scannet50_verified_completion.v1",
                "scene_count": 50,
                "asset_count": 100,
                "total_bytes": 37_587_327_416,
            },
        )

    def _write_executable(self, name: str, content: str) -> Path:
        path = self.fake_bin / name
        path.write_text(
            textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n"
        )
        path.chmod(0o755)
        return path

    def _install_fakes(self) -> None:
        (self.temporary / "json-helper.py").write_text(
            textwrap.dedent(
                """\
                import hashlib
                import json
                import os
                from pathlib import Path
                import sys

                def native_path(name):
                    if name == "/tmp":
                        return Path(os.environ["FIXTURE_WINDOWS_TEMP"])
                    if name.startswith("/tmp/"):
                        return Path(os.environ["FIXTURE_WINDOWS_TEMP"]) / name[5:]
                    return Path(name)

                arguments = sys.argv[1:]
                if arguments and arguments[0] == "--sha256sum":
                    for name in (item for item in arguments[1:] if item != "--"):
                        path = native_path(name)
                        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
                    raise SystemExit(0)
                if "-n" in "".join(arguments[:1]) or "-cn" in arguments:
                    values = {}
                    index = 0
                    while index < len(arguments):
                        if arguments[index] == "--arg":
                            values[arguments[index + 1]] = arguments[index + 2]
                            index += 3
                        else:
                            index += 1
                    print(json.dumps({
                        "result_root": values["result_root"],
                        "planned_stages": [
                            "preflight", "prepare", "smoke", "calibration",
                            "report", "verify",
                        ],
                        "gpu_index": values["gpu_index"],
                        "git_commit": values["git_commit"],
                    }, separators=(",", ":")))
                    raise SystemExit(0)
                if any(item.startswith("-") and "e" in item for item in arguments):
                    expected = {}
                    index = 0
                    while index < len(arguments):
                        if arguments[index] == "--arg":
                            expected[arguments[index + 1]] = arguments[index + 2]
                            index += 3
                        elif arguments[index] == "--argjson":
                            expected[arguments[index + 1]] = json.loads(arguments[index + 2])
                            index += 3
                        else:
                            index += 1
                    try:
                        payload = json.loads(native_path(arguments[-1]).read_text(encoding="utf-8"))
                        for field, value in expected.items():
                            if field.endswith("_length"):
                                observed = len(payload[field.removesuffix("_length")])
                            else:
                                observed = payload[field]
                            if observed != value:
                                raise KeyError(field)
                    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
                        raise SystemExit(1)
                    raise SystemExit(0)
                expression = next(
                    (item for item in arguments if item.startswith(".")), None
                )
                if expression is None:
                    raise SystemExit(91)
                path = native_path(arguments[-1])
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    selector, *operations = [part.strip() for part in expression.split("|")]
                    for field in selector.removeprefix(".").split("."):
                        value = value[field]
                    if operations == ["length"]:
                        value = len(value)
                    elif operations:
                        raise KeyError(expression)
                except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
                    raise SystemExit(1)
                if value is None or value is False:
                    raise SystemExit(1)
                if isinstance(value, bool):
                    print("true" if value else "false")
                elif isinstance(value, (dict, list)):
                    print(json.dumps(value, separators=(",", ":")))
                else:
                    print(value)
                """
            ).lstrip(),
            encoding="utf-8",
            newline="\n",
        )
        self._write_executable(
            "jq",
            """\
            #!/usr/bin/env bash
            export MSYS2_ARG_CONV_EXCL='*'
            exec "$FIXTURE_REAL_PYTHON" "$FIXTURE_JSON_HELPER" "$@"
            """,
        )
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
            if [[ -n "${FIXTURE_FREE_BYTES:-}" ]]; then
              printf 'Avail\n%s\n' "$FIXTURE_FREE_BYTES"
            elif [[ "$*" == *"-B1"* ]]; then
              printf 'Avail\n%s\n' "$((FIXTURE_FREE_GIB * 1024 * 1024 * 1024))"
            else
              printf 'Avail\n%sG\n' "${FIXTURE_FREE_GIB:?}"
            fi
            """,
        )
        self._write_executable(
            "git",
            """\
            #!/usr/bin/env bash
            next_value() {
              local kind="$1"
              local fallback="$2"
              local sequence_variable="FIXTURE_GIT_${kind}_SEQUENCE"
              local sequence="${!sequence_variable:-}"
              if [[ -z "$sequence" ]]; then
                printf '%b' "$fallback"
                return
              fi
              local count_file="${FIXTURE_GIT_COUNT_ROOT}-${kind}"
              local count=0
              [[ ! -f "$count_file" ]] || count="$(cat "$count_file")"
              printf '%s\n' "$((count + 1))" > "$count_file"
              local values
              IFS='|' read -r -a values <<< "$sequence"
              local index="$count"
              (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
              [[ "${values[index]}" != "__FAIL__" ]] || exit 91
              [[ "${values[index]}" == "__EMPTY__" ]] || printf '%b' "${values[index]}"
            }
            case "$*" in
              *"branch --show-current"*) next_value BRANCH "${FIXTURE_BRANCH:?}\\n" ;;
              *"status --short"*) next_value STATUS "${FIXTURE_DIRTY_STATUS:-}" ;;
              *"rev-parse HEAD"*) next_value HEAD "${FIXTURE_GIT_COMMIT:?}\\n" ;;
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
              sequence_variable="FIXTURE_GPU_${gpu}_PID_SEQUENCE"
              sequence="${!sequence_variable:-}"
              if [[ -n "$sequence" ]]; then
                count=0
                [[ ! -f "$FIXTURE_GPU_COUNT" ]] || count="$(cat "$FIXTURE_GPU_COUNT")"
                printf '%s\n' "$((count + 1))" > "$FIXTURE_GPU_COUNT"
                IFS='|' read -r -a values <<< "$sequence"
                index="$count"
                (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
                if [[ "${values[index]}" == "__FAIL__" ]]; then exit 93; fi
                [[ "${values[index]}" == "__EMPTY__" ]] || printf '%b' "${values[index]}"
              else
                printf '%b' "${!variable:-}"
              fi
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
            if [[ "${1:-}" == "-n" && "${2:-}" == "8" ]]; then
              mkdir "$FIXTURE_GPU_LOCK_STATE" 2>/dev/null || exit 1
              exit 0
            fi
            if [[ "${1:-}" == "-u" && "${2:-}" == "8" ]]; then
              rmdir "$FIXTURE_GPU_LOCK_STATE" 2>/dev/null || true
              exit 0
            fi
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
                if [[ "${FIXTURE_BLOCK_STAGE:-}" == "$stage" ]]; then
                  : > "$FIXTURE_BLOCK_STARTED"
                  while [[ ! -f "$FIXTURE_BLOCK_RELEASE" ]]; do sleep 0.05; done
                fi
                if [[ "${FIXTURE_STDERR_STAGE:-}" == "$stage" ]]; then
                  printf 'fixture stderr for %s\n' "$stage" >&2
                fi
                if [[ "${FIXTURE_FAIL_STAGE:-}" == "$stage" ]]; then exit 47; fi
                if [[ "${FIXTURE_TAMPER_AFTER_STAGE:-}" == "$stage" ]]; then
                  printf 'tampered\n' > "$FIXTURE_TAMPER_PATH"
                fi
                if [[ "$stage" == "verify" && "${FIXTURE_CREATE_COMPLETION:-0}" == "1" ]]; then
                  printf '{}\n' > "$run_root/verified_completion.json"
                fi
                """
            ).lstrip(),
            encoding="utf-8",
            newline="\n",
        )
        self.fake_python.chmod(0o755)

    def fixture_environment(
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
    ) -> dict[str, str]:
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
        return env

    def runner_command(self, arguments: tuple[str, ...] = ()) -> list[str]:
        return [
            str(BASH),
            "-c",
            'fake_bin="$1"; runner="$2"; shift 2; export PATH="$fake_bin:$PATH"; source "$runner" "$@"',
            "runner-fixture",
            bash_path(self.fake_bin),
            bash_path(RUNNER),
            *arguments,
        ]

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
        env = self.fixture_environment(
            hostname=hostname,
            user=user,
            free_gib=free_gib,
            gpu_name=gpu_name,
            gpu_compute_pids=gpu_compute_pids,
            branch=branch,
            dirty_status=dirty_status,
            gpu_rows=gpu_rows,
            environment=environment,
        )
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
            timeout=RUNNER_FIXTURE_TIMEOUT_SECONDS,
        )

    def python_stages(self) -> list[tuple[str, str]]:
        if not self.python_log.exists():
            return []
        return [
            (line.split("\t", 2)[0], line.split("\t", 2)[1])
            for line in self.python_log.read_text(encoding="utf-8").splitlines()
        ]

    def make_directory_link(self, link: Path, target: Path) -> None:
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError:
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                self.skipTest(f"directory symlink/junction unavailable: {result.stderr}")

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
        self._check_disk_threshold_accepts_exactly_100_gib()
        self._check_preflight_json_escapes_quoted_and_newline_result_root()

    def test_preflight_only_creates_no_result_or_run_directory(self) -> None:
        result_root = self.temporary / "results"

        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"RESULT_ROOT": bash_lexical_path(result_root)},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result_root.exists())
        self.assertEqual(self.python_stages(), [])
        self._check_result_root_symlink_is_rejected_without_outside_writes()
        self._check_control_parent_symlink_is_rejected_without_outside_writes()

    def test_preflight_only_fails_before_compute_for_busy_gpu(self) -> None:
        result = self.run_runner_fixture(
            gpu_compute_pids="8123\n",
            arguments=("--preflight-only",),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("active compute process", result.stderr)
        self.assertEqual(self.python_stages(), [])
        self._check_preflight_rejects_na_gpu_process_output()
        self._check_preflight_rejects_whitespace_malformed_and_failed_gpu_queries()

    def _check_preflight_rejects_na_gpu_process_output(self) -> None:
        result = self.run_runner_fixture(
            gpu_compute_pids="N/A\n",
            arguments=("--preflight-only",),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gpu", result.stderr.lower())
        self.assertEqual(self.python_stages(), [])

    def _check_preflight_rejects_whitespace_malformed_and_failed_gpu_queries(self) -> None:
        cases = (
            ("whitespace", {"FIXTURE_GPU_2_PIDS": " \\n"}),
            ("malformed", {"FIXTURE_GPU_2_PIDS": "not-a-pid\\n"}),
            ("query-failure", {"FIXTURE_GPU_2_PID_SEQUENCE": "__FAIL__"}),
        )
        for label, change in cases:
            with self.subTest(label=label):
                self.gpu_count.unlink(missing_ok=True)
                result = self.run_runner_fixture(
                    arguments=("--preflight-only",), environment=change
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("gpu", result.stderr.lower())
                self.assertEqual(self.python_stages(), [])

    def _check_gpu_is_rechecked_immediately_before_first_stage(self) -> None:
        self.python_log.unlink(missing_ok=True)
        self.gpu_count.unlink(missing_ok=True)
        result_root = self.temporary / "gpu-first-results"
        result = self.run_runner_fixture(
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": "gpu_first",
                "FIXTURE_GPU_2_PID_SEQUENCE": "__EMPTY__|8123\\n",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.python_stages(), [])

    def _check_gpu_and_git_are_rechecked_before_the_second_stage(self) -> None:
        cases = (
            (
                "gpu",
                {"FIXTURE_GPU_2_PID_SEQUENCE": "__EMPTY__|__EMPTY__|8123\\n"},
            ),
            (
                "branch",
                {
                    "FIXTURE_GIT_BRANCH_SEQUENCE": (
                        "codex/privileged-conditional-hvrfm|"
                        "codex/privileged-conditional-hvrfm|main"
                    )
                },
            ),
            (
                "status-failure",
                {
                    "FIXTURE_GIT_STATUS_SEQUENCE": (
                        "__EMPTY__|__EMPTY__|__FAIL__"
                    )
                },
            ),
        )
        for label, change in cases:
            with self.subTest(label=label):
                for counter in self.temporary.glob("git-count-*"):
                    counter.unlink()
                self.gpu_count.unlink(missing_ok=True)
                self.python_log.unlink(missing_ok=True)
                result = self.run_runner_fixture(
                    environment={
                        "RESULT_ROOT": bash_path(self.temporary / f"stage-results-{label}"),
                        "RUN_ID": f"stage_{label}",
                        **change,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(
                    [stage for _, stage in self.python_stages()], ["preflight"]
                )
                if label == "status-failure":
                    self.assertIn(
                        "could not recheck worktree cleanliness",
                        result.stderr.lower(),
                    )

    def _check_disk_threshold_is_exact_in_bytes(self) -> None:
        threshold = 100 * 1024**3
        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"FIXTURE_FREE_BYTES": str(threshold - 1)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("100 gib", result.stderr.lower())
        self.assertEqual(self.python_stages(), [])

    def _check_initial_git_status_failure_is_not_treated_as_clean(self) -> None:
        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"FIXTURE_GIT_STATUS_SEQUENCE": "__FAIL__"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not inspect worktree cleanliness", result.stderr.lower())
        self.assertEqual(self.python_stages(), [])

    def _check_disk_threshold_accepts_exactly_100_gib(self) -> None:
        threshold = 100 * 1024**3
        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"FIXTURE_FREE_BYTES": str(threshold)},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.python_stages(), [])

    def _check_empty_scannet_marker_is_not_authenticated(self) -> None:
        (self.scannet / "verified_completion.json").write_bytes(b"")

        result = self.run_runner_fixture(arguments=("--preflight-only",))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scannet", result.stderr.lower())
        self.assertEqual(self.python_stages(), [])

    def _check_marker_schema_and_manifest_bindings_are_authenticated(self) -> None:
        cases = ("vrfm-schema", "source-binding", "formal-data-binding")
        for label in cases:
            with self.subTest(label=label):
                self._write_authenticated_inputs()
                self.python_log.unlink(missing_ok=True)
                overrides: dict[str, str] = {}
                if label == "vrfm-schema":
                    marker = self.source / "verified_completion.json"
                    self._write_json(marker, {})
                    overrides["HVRFM_TEST_VRFM_MARKER_SHA256"] = sha256_file(marker)
                elif label == "source-binding":
                    manifest = self.source / "manifests" / "source_manifest.json"
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    payload["records"][0]["scene"] = "changed-scene"
                    self._write_json(manifest, payload)
                    overrides["HVRFM_TEST_SOURCE_MANIFEST_SHA256"] = sha256_file(
                        manifest
                    )
                else:
                    manifest = self.formal / "manifests" / "data_manifest.json"
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    payload["records"][0]["scene"] = "changed-scene"
                    self._write_json(manifest, payload)
                    overrides["HVRFM_TEST_FORMAL_DATA_MANIFEST_SHA256"] = sha256_file(
                        manifest
                    )
                result = self.run_runner_fixture(
                    arguments=("--preflight-only",), environment=overrides
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.python_stages(), [])

    def _check_authenticated_inputs_are_rechecked_before_the_second_stage(self) -> None:
        self.python_log.unlink(missing_ok=True)
        result_root = self.temporary / "marker-stage-results"
        result = self.run_runner_fixture(
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_TAMPER_AFTER_STAGE": "preflight",
                "FIXTURE_TAMPER_PATH": bash_path(
                    self.source / "verified_completion.json"
                ),
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([stage for _, stage in self.python_stages()], ["preflight"])

    def _check_preflight_json_escapes_quoted_and_newline_result_root(self) -> None:
        result_root = self.temporary / 'result "quoted"\nsecond-line'
        expected = bash_lexical_path(result_root)
        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"RESULT_ROOT": expected},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result_root"], expected)
        self.assertFalse(result_root.exists())
        self.assertEqual(self.python_stages(), [])

    def _check_formal_mode_ignores_fixture_path_overrides(self) -> None:
        ignored_overrides = {
            "REPO_ROOT": bash_path(self.temporary / "ignored-repository"),
            "PYTHON": bash_path(self.temporary / "ignored-python"),
            "RESULT_ROOT": bash_path(self.temporary / "ignored-results"),
            "SOURCE_RUN": bash_path(self.temporary / "ignored-source"),
            "FORMAL_LABEL_ROOT": bash_path(self.temporary / "ignored-formal"),
            "PREPARED_ROOT": bash_path(self.temporary / "ignored-prepared"),
            "CHECKPOINT_DIR": bash_path(self.temporary / "ignored-checkpoint"),
            "SCANNET_MARKER": bash_path(self.temporary / "ignored-scannet-marker"),
        }
        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={
                "HVRFM_RUNNER_TEST_MODE": "0",
                **ignored_overrides,
            },
        )

        if result.returncode == 0:
            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["result_root"],
                "/data/yjh/output/vggt/privileged_conditional_hvrfm",
            )
            self.assertNotEqual(
                payload["result_root"], ignored_overrides["RESULT_ROOT"]
            )
        else:
            self.assertIn(
                "/home/ubuntu/yjh/vggt/.worktrees/privileged_conditional_hvrfm",
                result.stderr,
            )
        for ignored_path in ignored_overrides.values():
            self.assertNotIn(ignored_path, result.stdout + result.stderr)
        self.assertEqual(self.python_stages(), [])

    def _check_result_root_symlink_is_rejected_without_outside_writes(self) -> None:
        outside = self.temporary / "outside-result"
        outside.mkdir()
        result_root = self.temporary / "results-link"
        self.make_directory_link(result_root, outside)

        result = self.run_runner_fixture(
            arguments=("--preflight-only",),
            environment={"RESULT_ROOT": bash_lexical_path(result_root)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.python_stages(), [])

    def _check_control_parent_symlink_is_rejected_without_outside_writes(self) -> None:
        outside = self.temporary / "outside-control"
        outside.mkdir()
        result_root = self.temporary / "results"
        result_root.mkdir()
        self.make_directory_link(result_root / ".runner_control", outside)

        result = self.run_runner_fixture(
            environment={"RESULT_ROOT": bash_path(result_root)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.python_stages(), [])

    def _check_git_identity_is_rechecked_before_the_first_stage(self) -> None:
        cases = (
            (
                "branch",
                {"FIXTURE_GIT_BRANCH_SEQUENCE": "codex/privileged-conditional-hvrfm|main"},
            ),
            (
                "head",
                {"FIXTURE_GIT_HEAD_SEQUENCE": f"{'1' * 40}|{'2' * 40}"},
            ),
            (
                "dirty",
                {"FIXTURE_GIT_STATUS_SEQUENCE": "__EMPTY__| M changed.py\\n"},
            ),
            (
                "status-failure",
                {"FIXTURE_GIT_STATUS_SEQUENCE": "__EMPTY__|__FAIL__"},
            ),
        )
        for label, change in cases:
            with self.subTest(label=label):
                for counter in self.temporary.glob("git-count-*"):
                    counter.unlink()
                self.python_log.unlink(missing_ok=True)
                result_root = self.temporary / f"results-{label}"
                result = self.run_runner_fixture(
                    environment={
                        "RESULT_ROOT": bash_path(result_root),
                        "RUN_ID": f"fixture_{label}",
                        **change,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.python_stages(), [])

    def _check_two_runs_cannot_hold_the_same_gpu_lock(self) -> None:
        self.python_log.unlink(missing_ok=True)
        self.du_count.unlink(missing_ok=True)
        if self.gpu_lock_state.exists():
            self.gpu_lock_state.rmdir()
        result_root = self.temporary / "shared-results"
        started = self.temporary / "block-started"
        release = self.temporary / "block-release"
        first_env = self.fixture_environment(
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": "first_run",
                "FIXTURE_BLOCK_STAGE": "preflight",
                "FIXTURE_BLOCK_STARTED": bash_path(started),
                "FIXTURE_BLOCK_RELEASE": bash_path(release),
            }
        )
        first = subprocess.Popen(
            self.runner_command(),
            cwd=ROOT,
            env=first_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(600):
                if started.exists():
                    break
                if first.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(
                started.exists(),
                f"first runner exited before blocking: returncode={first.poll()}",
            )
            second = self.run_runner_fixture(
                environment={
                    "RESULT_ROOT": bash_path(result_root),
                    "RUN_ID": "second_run",
                }
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("gpu", second.stderr.lower())
        finally:
            release.touch()
            first_stdout, first_stderr = first.communicate(
                timeout=RUNNER_FIXTURE_TIMEOUT_SECONDS
            )
        self.assertEqual(first.returncode, 0, first_stderr + first_stdout)

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
        self._check_gpu_is_rechecked_immediately_before_first_stage()
        self._check_gpu_and_git_are_rechecked_before_the_second_stage()

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
        self._check_disk_threshold_is_exact_in_bytes()
        self._check_initial_git_status_failure_is_not_treated_as_clean()
        self._check_empty_scannet_marker_is_not_authenticated()
        self._check_marker_schema_and_manifest_bindings_are_authenticated()
        self._check_formal_mode_ignores_fixture_path_overrides()
        self._write_authenticated_inputs()
        self._check_git_identity_is_rechecked_before_the_first_stage()

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
        self._write_authenticated_inputs()
        self._check_authenticated_inputs_are_rechecked_before_the_second_stage()

    def test_run_size_is_checked_after_each_completed_stage(self) -> None:
        result_root = self.temporary / "results"
        result = self.run_runner_fixture(
            environment={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_DU_KIB_SEQUENCE": "0,0,0,20971520",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run root reached 20 GiB", result.stderr)
        self.assertEqual(
            [stage for _, stage in self.python_stages()],
            ["preflight", "prepare"],
        )
        self._write_authenticated_inputs()
        self._check_two_runs_cannot_hold_the_same_gpu_lock()

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
