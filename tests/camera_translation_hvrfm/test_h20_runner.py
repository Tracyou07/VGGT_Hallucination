from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_camera_translation_hvrfm_targets.sh"
WINDOWS_GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(WINDOWS_GIT_BASH) if WINDOWS_GIT_BASH.is_file() else shutil.which("bash")
TIMEOUT_SECONDS = 120

REFERENCE_GIT = "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c"
FORMAL_GIT = "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e"
FORMAL_ASSIGNMENTS = {
    "REPO_ROOT": "/home/ubuntu/yjh/vggt/.worktrees/privileged_conditional_hvrfm",
    "PYTHON": "/home/ubuntu/anaconda3/envs/vggt-gx/bin/python",
    "EXPECTED_PYTHON_REALPATH": (
        "/home/ubuntu/anaconda3/envs/vggt-gx/bin/python3.10"
    ),
    "RESULT_ROOT": "/data/yjh/output/vggt/camera_translation_hvrfm",
    "SOURCE_RUN": (
        "/data/yjh/output/vggt/variational_camera_latent/"
        "vrfm_camera_20260827T044926Z"
    ),
    "REFERENCE_RUN": (
        "/data/yjh/output/vggt/privileged_conditional_hvrfm/"
        "privileged_teacher_lift_20260829T012716Z_tolfix"
    ),
    "FORMAL_RUN": (
        "/data/yjh/output/vggt/long_short_camera_head/"
        "long_short_head_formal_20260828T072407Z"
    ),
    "CHECKPOINT_DIR": "/data/yjh/share/pretrained/VGGT-1B",
    "EXPECTED_SOURCE_COMPLETION_SHA256": (
        "fd1b93caa16f45f0dbdc55fd7000aba9ab8bf166a7240f5ac2a716a0b3de9a32"
    ),
    "EXPECTED_SOURCE_MANIFEST_SHA256": (
        "be5aaa1b61be5e25709e40b3912e48aab38b6bbfac4be3b7ed183140219d6054"
    ),
    "EXPECTED_REFERENCE_COMPLETION_SHA256": (
        "7e63ca36e6fc4c08772e3356255f84c2853c9d46310ae546cc5e53dc1792048c"
    ),
    "EXPECTED_REFERENCE_INVENTORY_SHA256": (
        "046cf50cc7c7610a24d9f02571f7f0c438c79e43e89becf972e5d8594c465309"
    ),
    "EXPECTED_REFERENCE_CONFIG_SHA256": (
        "525333c71cc6e94300591def1191c9c02294380ecf77055e6cf44ea2028c6b5f"
    ),
    "EXPECTED_REFERENCE_REPORT_SHA256": (
        "5e0aedb1411c94ab839a7287750fa947731dbd4f10bfd9b4c89f8571a2474efc"
    ),
    "EXPECTED_REFERENCE_LONG_MANIFEST_SHA256": (
        "6b6ab434bb4cd8bd4afbeaf8a2d11354f321d8791501ada3ef2f9376eb064166"
    ),
    "EXPECTED_REFERENCE_TEACHER_MANIFEST_SHA256": (
        "d4c113515a72a2d79cd5e2f5139e290a787dc4d438c7caa22d9725d8fd99691e"
    ),
    "EXPECTED_FORMAL_COMPLETION_SHA256": (
        "4d24b944792f348ccc8c180a99f3e0ee11397ce472900eb6abe38f6924732667"
    ),
    "EXPECTED_FORMAL_MANIFEST_SHA256": (
        "944ee57a75a68af45fc0ea6037070267552ea3f042bd2346638cdc65f2dd4a6e"
    ),
    "EXPECTED_CHECKPOINT_SHA256": (
        "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
    ),
}
DEFAULT_GPU_ROWS = "".join(f"{index}, NVIDIA H20\n" for index in range(8))
STAGES = ("preflight", "prepare", "smoke", "calibration", "report", "verify")


def bash_path(path: Path) -> str:
    resolved = path.resolve()
    windows_temp = Path(tempfile.gettempdir()).resolve()
    try:
        return (Path("/tmp") / resolved.relative_to(windows_temp)).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_assignments(text: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for name, expected in FORMAL_ASSIGNMENTS.items():
        matches = re.findall(
            rf'^  {re.escape(name)}="{re.escape(expected)}"$', text, re.MULTILINE
        )
        if len(matches) != 1:
            raise AssertionError(f"formal assignment for {name} must be one literal")
        assignments[name] = expected
    return assignments


class H20RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        if BASH is None:
            self.skipTest("bash is required")
        self.temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)
        self.fake_bin = self.temporary / "bin"
        self.fake_bin.mkdir()
        self.repo = self.temporary / "worktree"
        self.repo.mkdir()
        (self.repo / ".git").write_text("gitdir: fixture\n", encoding="utf-8")
        self.source = self.temporary / "source"
        self.reference = self.temporary / "reference"
        self.formal = self.temporary / "formal"
        self.checkpoint = self.temporary / "checkpoint"
        for directory in (self.source, self.reference, self.formal, self.checkpoint):
            directory.mkdir()
        (self.source / "manifests").mkdir()
        (self.reference / "manifests").mkdir()
        (self.reference / "reports").mkdir()
        (self.formal / "manifests").mkdir()
        (self.checkpoint / "model.safetensors").write_bytes(b"checkpoint")
        self._write_authenticated_inputs()
        self.python_log = self.temporary / "python.tsv"
        self.git_count_root = self.temporary / "git-count"
        self.gpu_count_root = self.temporary / "gpu-count"
        self.du_count = self.temporary / "du-count"
        self.df_count = self.temporary / "df-count"
        self.jq_cache = self.temporary / "jq-cache"
        self.run_lock_state = self.temporary / "run-lock"
        self.gpu_lock_state = self.temporary / "gpu-lock"
        self.fake_python = self.temporary / "python"
        self.completion_helper = self.temporary / "completion-helper.py"
        self._install_fakes()
        self.expected_python_sha256 = sha256_file(self.fake_python)
        self.python_env_log = self.temporary / "python-env.tsv"
        self.base_env = os.environ.copy()
        for name in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
        ):
            self.base_env.pop(name, None)
        self.base_env.update(
            {
                "PATH": f"{bash_path(self.fake_bin)}:/usr/bin:/bin",
                "CAMERA_TRANSLATION_HVRFM_RUNNER_TEST_MODE": "1",
                "REPO_ROOT": bash_path(self.repo),
                "PYTHON": bash_path(self.fake_python),
                "CTHVRFM_TEST_PYTHON_REALPATH": bash_path(self.fake_python),
                "CTHVRFM_TEST_PYTHON_SHA256": self.expected_python_sha256,
                "RESULT_ROOT": bash_path(self.temporary / "results-default"),
                "SOURCE_RUN": bash_path(self.source),
                "REFERENCE_RUN": bash_path(self.reference),
                "FORMAL_RUN": bash_path(self.formal),
                "CHECKPOINT_DIR": bash_path(self.checkpoint),
                "RUN_ID": "fixture_run",
                "FIXTURE_HOSTNAME": "VM-0-11-ubuntu",
                "FIXTURE_USER": "ubuntu",
                "FIXTURE_BRANCH": "codex/privileged-conditional-hvrfm",
                "FIXTURE_GIT_COMMIT": "1" * 40,
                "FIXTURE_GIT_COUNT_ROOT": bash_path(self.git_count_root),
                "FIXTURE_GPU_COUNT_ROOT": bash_path(self.gpu_count_root),
                "FIXTURE_GPU_ROWS": DEFAULT_GPU_ROWS,
                "FIXTURE_FREE_BYTES": str(150 * 1024**3),
                "FIXTURE_DF_COUNT": bash_path(self.df_count),
                "FIXTURE_JQ_CACHE_ROOT": bash_path(self.jq_cache),
                "FIXTURE_DU_COUNT": bash_path(self.du_count),
                "FIXTURE_RUN_LOCK_STATE": bash_path(self.run_lock_state),
                "FIXTURE_GPU_LOCK_STATE": bash_path(self.gpu_lock_state),
                "FIXTURE_PYTHON_LOG": bash_path(self.python_log),
                "FIXTURE_PYTHON_ENV_LOG": bash_path(self.python_env_log),
                "FIXTURE_REAL_PYTHON": Path(sys.executable).as_posix(),
                "FIXTURE_JSON_HELPER": (
                    self.temporary / "json-helper.py"
                ).resolve().as_posix(),
                "FIXTURE_COMPLETION_HELPER": self.completion_helper.resolve().as_posix(),
                "FIXTURE_WINDOWS_TEMP": str(Path(tempfile.gettempdir()).resolve()),
                "FIXTURE_CREATE_COMPLETION": "1",
                **self._digest_environment(),
            }
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def _write_authenticated_inputs(self) -> None:
        checkpoint_sha = sha256_file(self.checkpoint / "model.safetensors")
        source_manifest = self.source / "manifests" / "source_manifest.json"
        self._write_json(
            source_manifest,
            {
                "schema": "variational_camera_latent.source.v1",
                "records": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        source_manifest_sha = sha256_file(source_manifest)
        self._write_json(
            self.source / "verified_completion.json",
            {
                "schema": "variational_camera_latent.verified_completion.v1",
                "signal": "WEAK_SIGNAL",
                "scene_count": 10,
                "overlap_count": 80,
                "candidate_count": 2560,
            },
        )
        long_manifest = self.reference / "manifests" / "long_context.json"
        teacher_manifest = self.reference / "manifests" / "teacher.json"
        self._write_json(
            long_manifest,
            {
                "schema": "conditional_hierarchical_vrfm.long_context_manifest.v1",
                "records": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        self._write_json(
            teacher_manifest,
            {
                "schema": "conditional_hierarchical_vrfm.teacher_manifest.v1",
                "git_commit": REFERENCE_GIT,
                "checkpoint_sha256": checkpoint_sha,
                "formal_completion_sha256": "pending",
                "formal_data_manifest_sha256": "pending",
                "records": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        formal_manifest = self.formal / "manifests" / "data_manifest.json"
        self._write_json(
            formal_manifest,
            {
                "schema": "long_short_camera_head.data_manifest.v1",
                "git_revision": FORMAL_GIT,
                "source_run": bash_path(self.source),
                "source_manifest_sha256": source_manifest_sha,
                "checkpoint_dir": bash_path(self.checkpoint),
                "base_checkpoint_sha256": checkpoint_sha,
                "records": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        formal_manifest_sha = sha256_file(formal_manifest)
        self._write_json(
            self.formal / "verified_completion.json",
            {
                "schema": "long_short_camera_head.verified_completion.v1",
                "git_revision": FORMAL_GIT,
                "source_manifest_sha256": source_manifest_sha,
                "base_checkpoint_sha256": checkpoint_sha,
                "data_manifest_sha256": formal_manifest_sha,
                "classification": "NO_SOURCE_HEAD_SIGNAL",
                "scene_count": 10,
                "train_scene_count": 8,
                "locked_replay_scene_count": 2,
                "inference_leakage_audit": True,
            },
        )
        formal_completion_sha = sha256_file(self.formal / "verified_completion.json")
        teacher_payload = json.loads(teacher_manifest.read_text(encoding="utf-8"))
        teacher_payload["formal_completion_sha256"] = formal_completion_sha
        teacher_payload["formal_data_manifest_sha256"] = formal_manifest_sha
        self._write_json(teacher_manifest, teacher_payload)
        self._write_json(
            self.reference / "config.json",
            {
                "schema": "conditional_hierarchical_vrfm.run_config.v1",
                "git_commit": REFERENCE_GIT,
                "checkpoint_sha256": checkpoint_sha,
                "source_manifest_sha256": source_manifest_sha,
                "formal_completion_sha256": formal_completion_sha,
                "formal_data_manifest_sha256": formal_manifest_sha,
                "long_manifest_sha256": sha256_file(long_manifest),
                "teacher_manifest_sha256": sha256_file(teacher_manifest),
                "source_run": bash_path(self.source),
                "formal_run_root": bash_path(self.formal),
                "scene_count": 10,
                "variant_count": 4,
            },
        )
        self._write_json(
            self.reference / "reports" / "stage_a.json",
            {
                "schema": "conditional_hierarchical_vrfm.stage_a_report.v1",
                "git_commit": REFERENCE_GIT,
                "classification": "LATENT_LIFT_FAILED",
                "scene_metrics": [{"scene": f"scene{index:04d}_00"} for index in range(10)],
            },
        )
        inventory = self.reference / "manifests" / "verification_inventory.json"
        self._write_json(
            inventory,
            {
                "schema": "conditional_hierarchical_vrfm.verification_inventory.v1",
                "git_commit": REFERENCE_GIT,
                "classification": "LATENT_LIFT_FAILED",
                "files": {f"artifact/{index:02d}": "a" * 64 for index in range(87)},
            },
        )
        self._write_json(
            self.reference / "verified_completion.json",
            {
                "schema": "conditional_hierarchical_vrfm.verified_completion.v1",
                "git_commit": REFERENCE_GIT,
                "classification": "LATENT_LIFT_FAILED",
                "file_count": 87,
                "inventory_sha256": sha256_file(inventory),
            },
        )

    def _digest_environment(self) -> dict[str, str]:
        return {
            "CTHVRFM_TEST_SOURCE_COMPLETION_SHA256": sha256_file(
                self.source / "verified_completion.json"
            ),
            "CTHVRFM_TEST_SOURCE_MANIFEST_SHA256": sha256_file(
                self.source / "manifests" / "source_manifest.json"
            ),
            "CTHVRFM_TEST_REFERENCE_COMPLETION_SHA256": sha256_file(
                self.reference / "verified_completion.json"
            ),
            "CTHVRFM_TEST_REFERENCE_INVENTORY_SHA256": sha256_file(
                self.reference / "manifests" / "verification_inventory.json"
            ),
            "CTHVRFM_TEST_REFERENCE_CONFIG_SHA256": sha256_file(
                self.reference / "config.json"
            ),
            "CTHVRFM_TEST_REFERENCE_REPORT_SHA256": sha256_file(
                self.reference / "reports" / "stage_a.json"
            ),
            "CTHVRFM_TEST_REFERENCE_LONG_MANIFEST_SHA256": sha256_file(
                self.reference / "manifests" / "long_context.json"
            ),
            "CTHVRFM_TEST_REFERENCE_TEACHER_MANIFEST_SHA256": sha256_file(
                self.reference / "manifests" / "teacher.json"
            ),
            "CTHVRFM_TEST_FORMAL_COMPLETION_SHA256": sha256_file(
                self.formal / "verified_completion.json"
            ),
            "CTHVRFM_TEST_FORMAL_MANIFEST_SHA256": sha256_file(
                self.formal / "manifests" / "data_manifest.json"
            ),
            "CTHVRFM_TEST_CHECKPOINT_SHA256": sha256_file(
                self.checkpoint / "model.safetensors"
            ),
        }

    def _write_executable(self, name: str, content: str) -> None:
        path = self.fake_bin / name
        path.write_text(
            textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n"
        )
        path.chmod(0o755)

    def _install_fakes(self) -> None:
        (self.temporary / "json-helper.py").write_text(
            textwrap.dedent(
                """\
                import hashlib
                import json
                import os
                from pathlib import Path
                import os
                import sys

                def native(name):
                    if name == "/tmp":
                        return Path(os.environ["FIXTURE_WINDOWS_TEMP"])
                    if name.startswith("/tmp/"):
                        return Path(os.environ["FIXTURE_WINDOWS_TEMP"]) / name[5:]
                    return Path(name)

                arguments = sys.argv[1:]
                if arguments and arguments[0] == "--sha256sum":
                    for name in (item for item in arguments[1:] if item != "--"):
                        print(f"{hashlib.sha256(native(name).read_bytes()).hexdigest()}  {name}")
                    raise SystemExit(0)
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
                if "-cn" in arguments:
                    if "result_root" not in expected:
                        print(json.dumps(expected, separators=(",", ":")))
                        raise SystemExit(0)
                    print(json.dumps({
                        "result_root": expected["result_root"],
                        "planned_stages": [
                            "preflight", "prepare", "smoke", "calibration", "report", "verify"
                        ],
                        "gpu_index": expected["gpu_index"],
                        "git_commit": expected["git_commit"],
                        "python_path": expected["python_path"],
                        "python_realpath": expected["python_realpath"],
                        "python_sha256": expected["python_sha256"],
                    }, separators=(",", ":")))
                    raise SystemExit(0)
                try:
                    payload = json.loads(native(arguments[-1]).read_text(encoding="utf-8"))
                    for field, value in expected.items():
                        if field.endswith("_length"):
                            observed = len(payload[field.removesuffix("_length")])
                        else:
                            observed = payload[field]
                        if observed != value:
                            raise KeyError(field)
                except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
                    raise SystemExit(1)
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
            [[ "${FIXTURE_JQ_RC:-0}" == "0" ]] || exit "$FIXTURE_JQ_RC"
            exec "$FIXTURE_REAL_PYTHON" "$FIXTURE_JSON_HELPER" "$@"
            """,
        )
        self._write_executable(
            "sha256sum",
            """\
            #!/usr/bin/env bash
            export MSYS2_ARG_CONV_EXCL='*'
            exec "$FIXTURE_REAL_PYTHON" "$FIXTURE_JSON_HELPER" --sha256sum "$@"
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
            count=0; [[ ! -f "$FIXTURE_DF_COUNT" ]] || count="$(cat "$FIXTURE_DF_COUNT")"
            printf '%s\n' "$((count + 1))" > "$FIXTURE_DF_COUNT"
            IFS=',' read -r -a values <<< "${FIXTURE_FREE_BYTES_SEQUENCE:-${FIXTURE_FREE_BYTES:?}}"
            index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
            printf 'Avail\n%s\n' "${values[index]}"
            """,
        )
        self._write_executable(
            "git",
            """\
            #!/usr/bin/env bash
            next_value() {
              local kind="$1" fallback="$2" variable="FIXTURE_GIT_${1}_SEQUENCE"
              local sequence="${!variable:-}" count_file="${FIXTURE_GIT_COUNT_ROOT}-${kind}" count=0
              if [[ -z "$sequence" ]]; then printf '%b' "$fallback"; return; fi
              [[ ! -f "$count_file" ]] || count="$(cat "$count_file")"
              printf '%s\n' "$((count + 1))" > "$count_file"
              IFS='|' read -r -a values <<< "$sequence"
              local index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
              [[ "${values[index]}" != "__FAIL__" ]] || exit 91
              [[ "${values[index]}" == "__EMPTY__" ]] || printf '%b' "${values[index]}"
            }
            case "$*" in
              *"branch --show-current"*) next_value BRANCH "${FIXTURE_BRANCH:?}\\n" ;;
              *"status --short"*) next_value STATUS "${FIXTURE_DIRTY_STATUS:-}" ;;
              *"rev-parse HEAD"*) next_value HEAD "${FIXTURE_GIT_COMMIT:?}\\n" ;;
              *) exit 92 ;;
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
                case "$1" in -i|--id) gpu="$2"; shift 2 ;; *) shift ;; esac
              done
              variable="FIXTURE_GPU_${gpu}_PIDS"
              sequence_variable="FIXTURE_GPU_${gpu}_PID_SEQUENCE"
              sequence="${!sequence_variable:-}"
              if [[ -n "$sequence" ]]; then
                count_file="${FIXTURE_GPU_COUNT_ROOT}-${gpu}"; count=0
                [[ ! -f "$count_file" ]] || count="$(cat "$count_file")"
                printf '%s\n' "$((count + 1))" > "$count_file"
                IFS='|' read -r -a values <<< "$sequence"
                index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
                [[ "${values[index]}" != "__FAIL__" ]] || exit 93
                [[ "${values[index]}" == "__EMPTY__" ]] || printf '%b' "${values[index]}"
              else
                printf '%b' "${!variable:-}"
              fi
              exit 0
            fi
            exit 94
            """,
        )
        self._write_executable(
            "du",
            """\
            #!/usr/bin/env bash
            count=0; [[ ! -f "$FIXTURE_DU_COUNT" ]] || count="$(cat "$FIXTURE_DU_COUNT")"
            printf '%s\n' "$((count + 1))" > "$FIXTURE_DU_COUNT"
            IFS=',' read -r -a values <<< "${FIXTURE_DU_KIB_SEQUENCE:-0}"
            index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
            printf '%s\t%s\n' "${values[index]}" "${@: -1}"
            """,
        )
        self._write_executable(
            "flock",
            """\
            #!/usr/bin/env bash
            if [[ "${1:-}" == "-n" ]]; then
              fd="$2"; [[ "$fd" == "9" ]] && state="$FIXTURE_RUN_LOCK_STATE" || state="$FIXTURE_GPU_LOCK_STATE"
              mkdir "$state" 2>/dev/null || exit 1
              if [[ "${FIXTURE_BLOCK_LOCK_FD:-}" == "$fd" ]]; then
                : > "$FIXTURE_BLOCK_STARTED"
                while [[ ! -f "$FIXTURE_BLOCK_RELEASE" ]]; do sleep 0.05; done
              fi
              exit 0
            fi
            if [[ "${1:-}" == "-u" ]]; then
              fd="$2"; [[ "$fd" == "9" ]] && state="$FIXTURE_RUN_LOCK_STATE" || state="$FIXTURE_GPU_LOCK_STATE"
              rmdir "$state" 2>/dev/null || true
              exit 0
            fi
            exit 95
            """,
        )
        (self.fake_bin / "functions.sh").write_text(
            textwrap.dedent(
                """\
                if [[ -n "${FIXTURE_FINAL_PRINTF_TAMPER_PATH:-}" ]]; then
                  printf() {
                    if [[ "${1:-}" == '[camera-translation-hvrfm] verified %s\\n' ]]; then
                      builtin printf 'tampered after terminal verification\n' > \
                        "$FIXTURE_FINAL_PRINTF_TAMPER_PATH"
                      : > "${FIXTURE_FINAL_PRINTF_TAMPER_PATH}.fixture-final-printf"
                    fi
                    builtin printf "$@"
                  }
                  export -f printf
                fi
                if [[ -n "${FIXTURE_CLEANUP_ORDER_HOOK:-}" ]]; then
                  rm() {
                    builtin printf 'cleanup\n'
                    command rm "$@"
                  }
                  export -f rm
                fi
                hostname() { printf '%s\n' "${FIXTURE_HOSTNAME:?}"; }
                id() {
                  [[ "${1:-}" == "-un" ]] || return 90
                  printf '%s\n' "${FIXTURE_USER:?}"
                }
                df() {
                  local count=0 index
                  local -a values
                  [[ ! -f "$FIXTURE_DF_COUNT" ]] || count="$(<"$FIXTURE_DF_COUNT")"
                  printf '%s\n' "$((count + 1))" > "$FIXTURE_DF_COUNT"
                  IFS=',' read -r -a values <<< "${FIXTURE_FREE_BYTES_SEQUENCE:-${FIXTURE_FREE_BYTES:?}}"
                  index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
                  printf 'Avail\n%s\n' "${values[index]}"
                }
                realpath() {
                  local argument path lexical=0 after_separator=0
                  for argument in "$@"; do [[ "$argument" != "-s" ]] || lexical=1; done
                  for path in "$@"; do
                    if (( after_separator == 0 )); then
                      [[ "$path" != "--" ]] || after_separator=1
                      continue
                    fi
                    if [[ "$path" == "${FIXTURE_CONTROL_SYMLINK_PATH:-}" \
                      && -e "$path.fixture-symlinked" && "$lexical" == "0" ]]; then
                      printf '%s-resolved\n' "$path"
                    elif [[ "$path" == "${FIXTURE_SYMLINK_PATH:-}" && "$lexical" == "0" ]]; then
                      printf '%s-resolved\n' "$path"
                    else
                      printf '%s\n' "$path"
                    fi
                  done
                }
                stat() {
                  local path after_separator=0 format='%d:%i'
                  local -a stat_paths=()
                  for path in "$@"; do
                    [[ "$path" != "%s" ]] || format='%s'
                    [[ "$path" != "%d:%i:%s" ]] || format='%d:%i:%s'
                  done
                  for path in "$@"; do
                    if (( after_separator == 0 )); then
                      [[ "$path" != "--" ]] || after_separator=1
                      continue
                    fi
                    stat_paths+=("$path")
                  done
                  if [[ "$format" == '%d:%i:%s' ]]; then
                    /usr/bin/stat -Lc "$format" -- "${stat_paths[@]}"
                    return
                  fi
                  for path in "${stat_paths[@]}"; do
                    if [[ "$path" == "${FIXTURE_SWAP_ON_STAT_PATH:-}" \
                      && ! -e "$path.fixture-fd-swapped" ]]; then
                      mv -- "$path" "$path.fixture-opened"
                      printf 'replacement\n' > "$path"
                      : > "$path.fixture-fd-swapped"
                    fi
                    if [[ "$path" == "${FIXTURE_REPLACE_PATH:-}" && -e "$path.fixture-replaced" ]]; then
                      printf 'fixture-replaced:%s\n' "$path"
                    elif [[ "$path" == /proc/*/fd/* \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*.lock \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/*.lock \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/identity.json \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/logs/*.log \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/final_validation_snapshot.json \
                      || "$path" == "$RESULT_ROOT"/*/verified_completion.json \
                      || "$path" == "$RESULT_ROOT"/*/manifests/verification_inventory.json ]]; then
                      /usr/bin/stat -Lc "$format" -- "$path"
                    else
                      printf 'fixture:%s\n' "$path"
                    fi
                  done
                }
                sha256sum() {
                  local path digest all_control=1
                  local -a hash_paths=()
                  for path in "$@"; do
                    [[ "$path" != "--" ]] || continue
                    hash_paths+=("$path")
                    [[ "$path" == "$RESULT_ROOT"/.runner_control/*/identity.json \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/logs/*.log \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/final_validation_snapshot.json ]] \
                      || all_control=0
                  done
                  if (( all_control == 1 && ${#hash_paths[@]} > 1 )); then
                    local swap_path="${FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH:-}"
                    local digest_output=""
                    if [[ -n "$swap_path" && -f "$swap_path" \
                      && ! -e "$swap_path.fixture-ledger-swapped" ]]; then
                      mv -- "$swap_path" "$swap_path.fixture-ledger-opened"
                      cp -- "$swap_path.fixture-ledger-opened" "$swap_path"
                      digest_output="$(/usr/bin/sha256sum -- "${hash_paths[@]}")"
                      mv -- "$swap_path" "$swap_path.fixture-ledger-replacement"
                      mv -- "$swap_path.fixture-ledger-opened" "$swap_path"
                      : > "$swap_path.fixture-ledger-swapped"
                      builtin printf '%s\n' "$digest_output"
                    else
                      /usr/bin/sha256sum -- "${hash_paths[@]}"
                    fi
                    return
                  fi
                  for path in "$@"; do
                    [[ "$path" != "--" ]] || continue
                    if [[ "$path" == /proc/*/fd/* ]]; then
                      /usr/bin/sha256sum -- "$path"
                      continue
                    elif [[ "$path" == "$RESULT_ROOT"/.runner_control/*/identity.json \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/logs/*.log \
                      || "$path" == "$RESULT_ROOT"/.runner_control/*/final_validation_snapshot.json ]]; then
                      /usr/bin/sha256sum -- "$path"
                      continue
                    elif [[ "$path" == "${FIXTURE_BAD_HASH_PATH:-}" || -f "$path.fixture-tampered" ]]; then
                      digest="$(printf '0%.0s' {1..64})"
                    else
                      case "$path" in
                        "$SOURCE_RUN/verified_completion.json") digest="$CTHVRFM_TEST_SOURCE_COMPLETION_SHA256" ;;
                        "$SOURCE_RUN/manifests/source_manifest.json") digest="$CTHVRFM_TEST_SOURCE_MANIFEST_SHA256" ;;
                        "$REFERENCE_RUN/verified_completion.json") digest="$CTHVRFM_TEST_REFERENCE_COMPLETION_SHA256" ;;
                        "$REFERENCE_RUN/manifests/verification_inventory.json") digest="$CTHVRFM_TEST_REFERENCE_INVENTORY_SHA256" ;;
                        "$REFERENCE_RUN/config.json") digest="$CTHVRFM_TEST_REFERENCE_CONFIG_SHA256" ;;
                        "$REFERENCE_RUN/reports/stage_a.json") digest="$CTHVRFM_TEST_REFERENCE_REPORT_SHA256" ;;
                        "$REFERENCE_RUN/manifests/long_context.json") digest="$CTHVRFM_TEST_REFERENCE_LONG_MANIFEST_SHA256" ;;
                        "$REFERENCE_RUN/manifests/teacher.json") digest="$CTHVRFM_TEST_REFERENCE_TEACHER_MANIFEST_SHA256" ;;
                        "$FORMAL_RUN/verified_completion.json") digest="$CTHVRFM_TEST_FORMAL_COMPLETION_SHA256" ;;
                        "$FORMAL_RUN/manifests/data_manifest.json") digest="$CTHVRFM_TEST_FORMAL_MANIFEST_SHA256" ;;
                        "$CHECKPOINT_DIR/model.safetensors") digest="$CTHVRFM_TEST_CHECKPOINT_SHA256" ;;
                        "$PYTHON") digest="$CTHVRFM_TEST_PYTHON_SHA256" ;;
                        *) return 97 ;;
                      esac
                    fi
                    printf '%s  %s\n' "$digest" "$path"
                  done
                }
                jq() {
                  if [[ " $* " == *" -cn "* ]]; then
                    "$fake_bin/jq" "$@"
                    return
                  fi
                  local path="${@: -1}" key marker status
                  key="${path//[^A-Za-z0-9]/_}"
                  marker="$FIXTURE_JQ_CACHE_ROOT/$key"
                  [[ ! -f "$marker" ]] || return 0
                  "$fake_bin/jq" "$@" || { status=$?; return "$status"; }
                  mkdir -p "$FIXTURE_JQ_CACHE_ROOT"
                  : > "$marker"
                }
                git() {
                  next_fixture_git_value() {
                    local kind="$1" fallback="$2" variable="FIXTURE_GIT_${1}_SEQUENCE"
                    local sequence="${!variable:-}" count_file="${FIXTURE_GIT_COUNT_ROOT}-${kind}" count=0
                    if [[ -z "$sequence" ]]; then printf '%b' "$fallback"; return; fi
                    [[ ! -f "$count_file" ]] || count="$(<"$count_file")"
                    printf '%s\n' "$((count + 1))" > "$count_file"
                    IFS='|' read -r -a values <<< "$sequence"
                    local index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
                    [[ "${values[index]}" != "__FAIL__" ]] || return 91
                    [[ "${values[index]}" == "__EMPTY__" ]] || printf '%b' "${values[index]}"
                  }
                  case "$*" in
                    *"branch --show-current"*) next_fixture_git_value BRANCH "${FIXTURE_BRANCH:?}\\n" ;;
                    *"status --short"*) next_fixture_git_value STATUS "${FIXTURE_DIRTY_STATUS:-}" ;;
                    *"rev-parse HEAD"*) next_fixture_git_value HEAD "${FIXTURE_GIT_COMMIT:?}\\n" ;;
                    *) return 92 ;;
                  esac
                }
                nvidia-smi() {
                  if [[ "$*" == *"--query-gpu=index,name"* ]]; then
                    printf '%b' "${FIXTURE_GPU_ROWS:?}"
                    return 0
                  fi
                  if [[ "$*" == *"--query-compute-apps=pid"* ]]; then
                    local gpu="" variable sequence_variable sequence count_file count index
                    local -a values
                    while (( $# )); do
                      case "$1" in -i|--id) gpu="$2"; shift 2 ;; *) shift ;; esac
                    done
                    variable="FIXTURE_GPU_${gpu}_PIDS"
                    sequence_variable="FIXTURE_GPU_${gpu}_PID_SEQUENCE"
                    sequence="${!sequence_variable:-}"
                    if [[ -n "$sequence" ]]; then
                      count_file="${FIXTURE_GPU_COUNT_ROOT}-${gpu}"; count=0
                      [[ ! -f "$count_file" ]] || count="$(<"$count_file")"
                      printf '%s\n' "$((count + 1))" > "$count_file"
                      IFS='|' read -r -a values <<< "$sequence"
                      index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
                      [[ "${values[index]}" != "__FAIL__" ]] || return 93
                      [[ "${values[index]}" == "__EMPTY__" ]] || printf '%b' "${values[index]}"
                    else
                      printf '%b' "${!variable:-}"
                    fi
                    return 0
                  fi
                  return 94
                }
                du() {
                  local count=0 index
                  local -a values
                  [[ ! -f "$FIXTURE_DU_COUNT" ]] || count="$(<"$FIXTURE_DU_COUNT")"
                  printf '%s\n' "$((count + 1))" > "$FIXTURE_DU_COUNT"
                  IFS=',' read -r -a values <<< "${FIXTURE_DU_KIB_SEQUENCE:-0}"
                  index="$count"; (( index < ${#values[@]} )) || index="$((${#values[@]} - 1))"
                  printf '%s\t%s\n' "${values[index]}" "${@: -1}"
                }
                flock() {
                  local fd state opened_identity replacement_identity
                  if [[ "${1:-}" == "-n" ]]; then
                    fd="$2"; [[ "$fd" == "9" ]] && state="$FIXTURE_RUN_LOCK_STATE" || state="$FIXTURE_GPU_LOCK_STATE"
                    mkdir "$state" 2>/dev/null || return 1
                    if [[ "${FIXTURE_BLOCK_LOCK_FD:-}" == "$fd" ]]; then
                      : > "$FIXTURE_BLOCK_STARTED"
                      while [[ ! -f "$FIXTURE_BLOCK_RELEASE" ]]; do sleep 0.05; done
                    fi
                    return 0
                  fi
                  if [[ "${1:-}" == "-u" ]]; then
                    fd="$2"; [[ "$fd" == "9" ]] && state="$FIXTURE_RUN_LOCK_STATE" || state="$FIXTURE_GPU_LOCK_STATE"
                    [[ "${FIXTURE_FAIL_UNLOCK_FD:-}" != "$fd" ]] || return 96
                    if [[ "${FIXTURE_HANDOFF_LOCK_FD:-}" == "$fd" ]]; then
                      opened_identity="$(/usr/bin/stat -Lc '%d:%i' -- "/proc/$$/fd/$fd")" || return 98
                      [[ ! -e "${FIXTURE_HANDOFF_LOCK_PATH:?}" \
                        && ! -L "$FIXTURE_HANDOFF_LOCK_PATH" ]] || return 99
                      printf 'next-owner\n' > "${FIXTURE_HANDOFF_LOCK_PATH:?}"
                      replacement_identity="$(/usr/bin/stat -Lc '%d:%i' -- \
                        "$FIXTURE_HANDOFF_LOCK_PATH")" || return 98
                      [[ "$replacement_identity" != "$opened_identity" ]] || return 99
                    fi
                    rmdir "$state" 2>/dev/null || true
                    return 0
                  fi
                  return 95
                }
                """
            ).lstrip(),
            encoding="utf-8",
            newline="\n",
        )
        self.completion_helper.write_text(
            textwrap.dedent(
                """\
                import hashlib
                import json
                import os
                from pathlib import Path
                import sys

                def digest(payload):
                    encoded = json.dumps(
                        payload, sort_keys=True, separators=(",", ":"),
                    ).encode("utf-8")
                    return hashlib.sha256(encoded).hexdigest()

                def native(name):
                    if name == "/tmp":
                        return Path(os.environ["FIXTURE_WINDOWS_TEMP"])
                    if name.startswith("/tmp/"):
                        return Path(os.environ["FIXTURE_WINDOWS_TEMP"]) / name[5:]
                    return Path(name)

                root = native(sys.argv[1])
                run_id = sys.argv[2]
                git_commit = sys.argv[3]
                report_bytes = b"report completion\\n"
                calibration_bytes = b"calibration completion\\n"
                report_path = root / "reports" / "completed.json"
                calibration_path = root / "calibration" / "completed.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_bytes(report_bytes)
                calibration_path.write_bytes(calibration_bytes)
                report_sha256 = hashlib.sha256(report_bytes).hexdigest()
                calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
                files = {
                    "reports/completed.json": {
                        "sha256": report_sha256,
                        "bytes": len(report_bytes),
                    },
                    "calibration/completed.json": {
                        "sha256": calibration_sha256,
                        "bytes": len(calibration_bytes),
                    },
                }
                if run_id == "missing_calibration_record":
                    del files["calibration/completed.json"]
                report_binding = report_sha256
                if run_id == "forged_report_binding":
                    report_binding = "c" * 64
                inventory_unsigned = {
                    "schema": "camera_translation_hvrfm.verification_inventory.v1",
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "classification": "TRANSLATION_ENDPOINTS_READY",
                    "report_completion_sha256": report_binding,
                    "calibration_completion_sha256": calibration_sha256,
                    "files": files,
                    "file_count": len(files),
                    "total_bytes": sum(record["bytes"] for record in files.values()),
                }
                inventory = {
                    **inventory_unsigned,
                    "completion_digest": digest(inventory_unsigned),
                }
                inventory_bytes = (
                    json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\\n"
                ).encode("utf-8")
                inventory_path = root / "manifests" / "verification_inventory.json"
                inventory_path.parent.mkdir(parents=True, exist_ok=True)
                inventory_path.write_bytes(inventory_bytes)
                completion_unsigned = {
                    "schema": "camera_translation_hvrfm.verified_completion.v1",
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "classification": "TRANSLATION_ENDPOINTS_READY",
                    "inventory_path": "manifests/verification_inventory.json",
                    "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                    "report_completion_sha256": report_binding,
                    "file_count": inventory["file_count"],
                    "total_bytes": inventory["total_bytes"],
                }
                completion = {
                    **completion_unsigned,
                    "completion_digest": digest(completion_unsigned),
                }
                (root / "verified_completion.json").write_text(
                    json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\\n",
                    encoding="utf-8",
                )
                if run_id == "missing_inventory_file":
                    report_path.unlink()
                elif run_id == "bad_inventory_hash":
                    report_path.write_bytes(b"changed report completion\\n")
                elif run_id == "fifo_inventory_file" and hasattr(os, "mkfifo"):
                    report_path.unlink()
                    os.mkfifo(report_path)
                elif run_id == "fifo_completion_marker" and hasattr(os, "mkfifo"):
                    completion_path = root / "verified_completion.json"
                    completion_path.unlink()
                    os.mkfifo(completion_path)
                """
            ).lstrip(),
            encoding="utf-8",
            newline="\n",
        )
        self.fake_python.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                if [[ "${1:-}" == "-" ]]; then
                  if [[ "${2:-}" == "control-ledger" \
                    && -n "${FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH:-}" ]]; then
                    checker_swap_path="$FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH"
                    if [[ -f "$checker_swap_path" \
                      && ! -e "$checker_swap_path.fixture-ledger-swapped" ]]; then
                      mv -- "$checker_swap_path" "$checker_swap_path.fixture-ledger-opened"
                      cp -- "$checker_swap_path.fixture-ledger-opened" "$checker_swap_path"
                      : > "$checker_swap_path.fixture-ledger-swapped"
                    fi
                  fi
                  validator_mode="legacy"
                  validator_run_root="${2%/verified_completion.json}"
                  validator_run_id="${4:-}"
                  if [[ "${2:-}" == "capture" || "${2:-}" == "verify" ]]; then
                    validator_mode="$2"
                    validator_run_root="$3"
                    validator_run_id="$4"
                  fi
                  alternating_control=""
                  if [[ "$validator_run_id" == "terminal_control_alternating" ]]; then
                    alternating_control="${validator_run_root%/*}/.runner_control/${validator_run_id}/logs/preflight.out.log"
                    if [[ "$validator_mode" == "capture" ]]; then
                      mv -- "$alternating_control" "$alternating_control.fixture-ledger-original"
                      cp -- "$alternating_control.fixture-ledger-original" "$alternating_control"
                    elif [[ "$validator_mode" == "verify" ]]; then
                      mv -- "$alternating_control" "$alternating_control.fixture-ledger-original"
                      mv -- "$alternating_control.fixture-ledger-replacement" "$alternating_control"
                    fi
                  fi
                  "$FIXTURE_REAL_PYTHON" "$@"
                  status=$?
                  if [[ -n "$alternating_control" ]]; then
                    if [[ "$validator_mode" == "capture" ]]; then
                      mv -- "$alternating_control" "$alternating_control.fixture-ledger-replacement"
                      mv -- "$alternating_control.fixture-ledger-original" "$alternating_control"
                    elif [[ "$validator_mode" == "verify" ]]; then
                      mv -- "$alternating_control" "$alternating_control.fixture-ledger-replacement"
                      mv -- "$alternating_control.fixture-ledger-original" "$alternating_control"
                    fi
                  fi
                  if (( status == 0 )) && [[ "$validator_mode" != "verify" ]]; then
                    if [[ "$validator_run_id" == "terminal_marker_tamper" ]]; then
                      mv -- "$validator_run_root/verified_completion.json" \
                        "$validator_run_root/verified_completion.json.fixture-validated"
                      cp -- "$validator_run_root/verified_completion.json.fixture-validated" \
                        "$validator_run_root/verified_completion.json"
                    elif [[ "$validator_run_id" == "terminal_leaf_in_place" ]]; then
                      printf 'changed report completion\n' > \
                        "$validator_run_root/reports/completed.json"
                    elif [[ "$validator_run_id" == "terminal_leaf_replacement" ]]; then
                      mv -- "$validator_run_root/reports/completed.json" \
                        "$validator_run_root/reports/completed.json.fixture-validated"
                      cp -- "$validator_run_root/reports/completed.json.fixture-validated" \
                        "$validator_run_root/reports/completed.json"
                    fi
                  fi
                  exit "$status"
                fi
                stage=""; run_root=""; git_commit=""; previous=""
                for argument in "$@"; do
                  case "$argument" in preflight|prepare|smoke|calibration|report|verify) stage="$argument" ;; esac
                  [[ "$previous" != "--run-root" ]] || run_root="$argument"
                  [[ "$previous" != "--git-commit" ]] || git_commit="$argument"
                  previous="$argument"
                done
                printf '%s\t%s\t%s\n' "${CUDA_VISIBLE_DEVICES:-}" "$stage" "$*" >> "$FIXTURE_PYTHON_LOG"
                {
                  printf '%s' "$stage"
                  env | LC_ALL=C sort | sed 's/=.*//' | while IFS= read -r name; do printf '\t%s' "$name"; done
                  printf '\tHF_HUB_OFFLINE=%s\tTRANSFORMERS_OFFLINE=%s\n' \
                    "${HF_HUB_OFFLINE:-}" "${TRANSFORMERS_OFFLINE:-}"
                } >> "$FIXTURE_PYTHON_ENV_LOG"
                if [[ "$stage" == "preflight" ]]; then mkdir "$run_root" || exit 96; fi
                if [[ "${FIXTURE_BLOCK_STAGE:-}" == "$stage" ]]; then
                  : > "$FIXTURE_BLOCK_STARTED"
                  while [[ ! -f "$FIXTURE_BLOCK_RELEASE" ]]; do sleep 0.05; done
                fi
                [[ "${FIXTURE_STDERR_STAGE:-}" != "$stage" ]] || printf 'fixture stderr\n' >&2
                [[ "${FIXTURE_FAIL_STAGE:-}" != "$stage" ]] || exit 47
                if [[ "${FIXTURE_TAMPER_AFTER_STAGE:-}" == "$stage" ]]; then
                  printf 'changed\n' > "$FIXTURE_TAMPER_PATH"
                  : > "$FIXTURE_TAMPER_PATH.fixture-tampered"
                fi
                if [[ "${FIXTURE_TAMPER_PYTHON_AFTER_STAGE:-}" == "$stage" ]]; then
                  : > "$0.fixture-tampered"
                fi
                if [[ "${FIXTURE_REPLACE_PATH_AFTER_STAGE:-}" == "$stage" ]]; then
                  : > "$FIXTURE_REPLACE_PATH.fixture-replaced"
                fi
                if [[ "${FIXTURE_CONTROL_MUTATE_AFTER_STAGE:-}" == "$stage" ]]; then
                  case "${FIXTURE_CONTROL_MUTATION_MODE:-replace}" in
                    replace)
                      mv -- "$FIXTURE_CONTROL_MUTATE_PATH" \
                        "$FIXTURE_CONTROL_MUTATE_PATH.fixture-validated"
                      cp -- "$FIXTURE_CONTROL_MUTATE_PATH.fixture-validated" \
                        "$FIXTURE_CONTROL_MUTATE_PATH"
                      ;;
                    symlink)
                      : > "$FIXTURE_CONTROL_MUTATE_PATH.fixture-symlinked"
                      ;;
                    *) exit 99 ;;
                  esac
                fi
                if [[ "$stage" == "verify" && "${FIXTURE_CREATE_COMPLETION:-0}" == "1" ]]; then
                  if [[ "${FIXTURE_COMPLETION_MODE:-valid}" == "empty" ]]; then
                    printf '{}\n' > "$run_root/verified_completion.json"
                  else
                    "$FIXTURE_REAL_PYTHON" "$FIXTURE_COMPLETION_HELPER" \
                      "$run_root" "$(basename "$run_root")" "$git_commit"
                  fi
                fi
                """
            ).lstrip(),
            encoding="utf-8",
            newline="\n",
        )
        self.fake_python.chmod(0o755)

    def environment(self, changes: dict[str, str] | None = None) -> dict[str, str]:
        env = self.base_env.copy()
        if changes:
            env.update(changes)
        return env

    def required_arguments(self) -> tuple[str, ...]:
        return (
            "--expected-git-commit",
            self.base_env["FIXTURE_GIT_COMMIT"],
            "--expected-python-sha256",
            self.expected_python_sha256,
        )

    def command(
        self,
        arguments: tuple[str, ...] = (),
        *,
        include_required: bool = True,
    ) -> list[str]:
        effective_arguments = (
            (*self.required_arguments(), *arguments) if include_required else arguments
        )
        return [
            str(BASH),
            "-c",
            (
                'fake_bin="$1"; runner="$2"; shift 2; '
                'export PATH="$fake_bin:$PATH"; '
                'source "$fake_bin/functions.sh"; '
                'source "$runner" "$@"'
            ),
            "fixture",
            bash_path(self.fake_bin),
            bash_path(RUNNER),
            *effective_arguments,
        ]

    def run_runner(
        self,
        *,
        arguments: tuple[str, ...] = (),
        changes: dict[str, str] | None = None,
        include_required: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(arguments, include_required=include_required),
            cwd=ROOT,
            env=self.environment(changes),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )

    def calls(self) -> list[tuple[str, str, str]]:
        if not self.python_log.exists():
            return []
        return [
            tuple(line.split("\t", 2))  # type: ignore[misc]
            for line in self.python_log.read_text(encoding="utf-8").splitlines()
        ]

    def reset_runtime(self) -> None:
        self.python_log.unlink(missing_ok=True)
        self.python_env_log.unlink(missing_ok=True)
        Path(f"{self.fake_python}.fixture-tampered").unlink(missing_ok=True)
        self.du_count.unlink(missing_ok=True)
        self.df_count.unlink(missing_ok=True)
        for path in self.temporary.glob("git-count-*"):
            path.unlink()
        for path in self.temporary.glob("gpu-count-*"):
            path.unlink()
        for path in (self.run_lock_state, self.gpu_lock_state):
            if path.exists():
                path.rmdir()

    def test_preflight_only_is_read_only_and_reports_exact_plan(self) -> None:
        result_root = self.temporary / "preflight-results"
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={"RESULT_ROOT": bash_path(result_root)},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "result_root": bash_path(result_root),
                "planned_stages": list(STAGES),
                "gpu_index": "0",
                "git_commit": "1" * 40,
                "python_path": bash_path(self.fake_python),
                "python_realpath": bash_path(self.fake_python),
                "python_sha256": self.expected_python_sha256,
            },
        )
        self.assertFalse(result_root.exists())
        self.assertEqual(self.calls(), [])

    def test_requires_preverified_git_commit_and_python_digest(self) -> None:
        missing = self.run_runner(
            arguments=("--preflight-only",), include_required=False
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("expected-git-commit", missing.stderr.lower())
        self.assertEqual(self.calls(), [])

        wrong_git = self.run_runner(
            arguments=(
                "--expected-git-commit",
                "2" * 40,
                "--expected-python-sha256",
                self.expected_python_sha256,
                "--preflight-only",
            ),
            include_required=False,
        )
        self.assertNotEqual(wrong_git.returncode, 0)
        self.assertIn("expected git commit", wrong_git.stderr.lower())
        self.assertEqual(self.calls(), [])

        wrong_python = self.run_runner(
            arguments=(
                "--expected-git-commit",
                "1" * 40,
                "--expected-python-sha256",
                "0" * 64,
                "--preflight-only",
            ),
            include_required=False,
        )
        self.assertNotEqual(wrong_python.returncode, 0)
        self.assertIn("python sha-256", wrong_python.stderr.lower())
        self.assertEqual(self.calls(), [])

    def test_shell_gates_fail_without_output_or_python(self) -> None:
        cases = (
            ("host", {"FIXTURE_HOSTNAME": "wrong"}, "host identity"),
            ("user", {"FIXTURE_USER": "root"}, "user identity"),
            ("branch", {"FIXTURE_BRANCH": "main"}, "wrong branch"),
            ("dirty", {"FIXTURE_DIRTY_STATUS": " M file\n"}, "dirty worktree"),
            ("short-head", {"FIXTURE_GIT_COMMIT": "1" * 39}, "full git commit"),
            ("disk", {"FIXTURE_FREE_BYTES": str(100 * 1024**3 - 1)}, "100 gib"),
            ("inventory", {"FIXTURE_GPU_ROWS": DEFAULT_GPU_ROWS[:-15]}, "eight"),
            (
                "busy",
                {**{f"FIXTURE_GPU_{index}_PIDS": f"{8000 + index}\\n" for index in range(8)}},
                "active compute",
            ),
        )
        for label, changes, message in cases:
            with self.subTest(label=label):
                self.reset_runtime()
                result_root = self.temporary / f"failed-{label}"
                result = self.run_runner(
                    arguments=("--preflight-only",),
                    changes={"RESULT_ROOT": bash_path(result_root), **changes},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr.lower())
                self.assertFalse(result_root.exists())
                self.assertEqual(self.calls(), [])

    def test_hash_and_provenance_are_authenticated_before_writes(self) -> None:
        result_root = self.temporary / "hash-failure"
        report = self.reference / "reports" / "stage_a.json"
        report.write_text("{}\n", encoding="utf-8")
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_BAD_HASH_PATH": bash_path(report),
            },
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reference report sha-256 mismatch", result.stderr.lower())
        self.assertFalse(result_root.exists())
        self.assertEqual(self.calls(), [])

    def test_every_critical_tree_is_passed_to_the_symlink_gate(self) -> None:
        cases = {
            "repository": self.repo,
            "source": self.source,
            "reference": self.reference,
            "formal": self.formal,
            "checkpoint": self.checkpoint,
            "result": self.temporary / "linked-result",
        }
        for label, path in cases.items():
            with self.subTest(label=label):
                self.reset_runtime()
                result_root = (
                    path if label == "result" else self.temporary / f"symlink-{label}"
                )
                result = self.run_runner(
                    arguments=("--preflight-only",),
                    changes={
                        "RESULT_ROOT": bash_path(result_root),
                        "FIXTURE_SYMLINK_PATH": bash_path(path),
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symlink", result.stderr.lower())
                self.assertFalse(result_root.exists())
                self.assertEqual(self.calls(), [])

    def test_python_and_every_nested_frozen_leaf_use_the_symlink_gate(self) -> None:
        paths = (
            self.fake_python,
            self.source / "manifests",
            self.source / "verified_completion.json",
            self.source / "manifests" / "source_manifest.json",
            self.reference / "manifests",
            self.reference / "reports",
            self.reference / "verified_completion.json",
            self.reference / "manifests" / "verification_inventory.json",
            self.reference / "config.json",
            self.reference / "reports" / "stage_a.json",
            self.reference / "manifests" / "long_context.json",
            self.reference / "manifests" / "teacher.json",
            self.formal / "manifests",
            self.formal / "verified_completion.json",
            self.formal / "manifests" / "data_manifest.json",
        )
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                self.reset_runtime()
                result_root = self.temporary / f"nested-symlink-{index}"
                result = self.run_runner(
                    arguments=("--preflight-only",),
                    changes={
                        "RESULT_ROOT": bash_path(result_root),
                        "FIXTURE_SYMLINK_PATH": bash_path(path),
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symlink", result.stderr.lower())
                self.assertFalse(result_root.exists())
                self.assertEqual(self.calls(), [])

    def test_overlong_run_id_is_rejected_without_writes(self) -> None:
        result_root = self.temporary / "overlong"
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={"RESULT_ROOT": bash_path(result_root), "RUN_ID": "a" * 129},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run_id", result.stderr.lower())
        self.assertFalse(result_root.exists())
        self.assertEqual(self.calls(), [])

    def test_secret_environment_is_rejected_and_script_has_no_credentials(self) -> None:
        result_root = self.temporary / "secret-failure"
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={"RESULT_ROOT": bash_path(result_root), "HF_TOKEN": "not-allowed"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("credential", result.stderr.lower())
        self.assertFalse(result_root.exists())
        self.assertEqual(self.calls(), [])

        text = RUNNER.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{8,}", text))
        self.assertNotIn("BEGIN OPENSSH PRIVATE KEY", text)
        self.assertIsNone(re.search(r"Bearer\s+[A-Za-z0-9._-]{12,}", text, re.I))

    def test_formal_mode_uses_only_frozen_literal_paths_and_digests(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(formal_assignments(text), FORMAL_ASSIGNMENTS)

    def test_formal_mode_ignores_environment_path_overrides(self) -> None:
        pinned_cleanup_commands = subprocess.run(
            [
                str(BASH),
                "-c",
                (
                    "test -x /usr/bin/stat && test -x /usr/bin/rm "
                    "&& test -x /usr/bin/flock"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if pinned_cleanup_commands.returncode != 0:
            self.skipTest("formal runtime probe requires pinned cleanup commands")
        absent_formal_repo = subprocess.run(
            [
                str(BASH),
                "-c",
                'test ! -e "$1" && test ! -L "$1"',
                "fixture",
                FORMAL_ASSIGNMENTS["REPO_ROOT"],
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if absent_formal_repo.returncode != 0:
            self.skipTest("formal runtime probe requires an absent frozen worktree")
        ignored = self.temporary / "ignored"
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={
                "CAMERA_TRANSLATION_HVRFM_RUNNER_TEST_MODE": "0",
                "REPO_ROOT": bash_path(ignored / "repo"),
                "PYTHON": bash_path(ignored / "python"),
                "RESULT_ROOT": bash_path(ignored / "results"),
                "SOURCE_RUN": bash_path(ignored / "source"),
                "REFERENCE_RUN": bash_path(ignored / "reference"),
                "FORMAL_RUN": bash_path(ignored / "formal"),
                "CHECKPOINT_DIR": bash_path(ignored / "checkpoint"),
                "RUN_ID": "ignored_run_id",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(FORMAL_ASSIGNMENTS["REPO_ROOT"], result.stderr)
        self.assertNotIn(bash_path(ignored), result.stdout + result.stderr)

    def test_complete_run_uses_one_idle_h20_full_args_and_external_logs(self) -> None:
        result_root = self.temporary / "complete-results"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_GPU_0_PIDS": "7000\n",
                "FIXTURE_GPU_1_PIDS": "",
            }
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([(gpu, stage) for gpu, stage, _ in calls], [("1", s) for s in STAGES])
        run_root = result_root / "fixture_run"
        self.assertTrue((run_root / "verified_completion.json").is_file())
        log_root = result_root / ".runner_control" / "fixture_run" / "logs"
        for stage in STAGES:
            self.assertTrue((log_root / f"{stage}.out.log").is_file())
            self.assertTrue((log_root / f"{stage}.err.log").is_file())
            command = next(raw for _, observed, raw in calls if observed == stage)
            self.assertIn("-m pre_experiments.camera_translation_hvrfm.stages", command)
            expected_pairs = {
                "--run-root": bash_path(run_root),
                "--git-commit": "1" * 40,
                "--source-run": bash_path(self.source),
                "--reference-run": bash_path(self.reference),
                "--formal-run": bash_path(self.formal),
                "--checkpoint-dir": bash_path(self.checkpoint),
                "--expected-source-completion-sha256": self.base_env[
                    "CTHVRFM_TEST_SOURCE_COMPLETION_SHA256"
                ],
                "--expected-reference-completion-sha256": self.base_env[
                    "CTHVRFM_TEST_REFERENCE_COMPLETION_SHA256"
                ],
                "--expected-formal-completion-sha256": self.base_env[
                    "CTHVRFM_TEST_FORMAL_COMPLETION_SHA256"
                ],
                "--expected-checkpoint-sha256": self.base_env[
                    "CTHVRFM_TEST_CHECKPOINT_SHA256"
                ],
                "--device": "cuda",
            }
            for flag, value in expected_pairs.items():
                self.assertIn(f"{flag} {value}", command)
        self.assertFalse((run_root / "logs").exists())

        snapshot = (
            result_root
            / ".runner_control"
            / "fixture_run"
            / "final_validation_snapshot.json"
        ).read_bytes()
        canonical_snapshot = (
            json.dumps(
                json.loads(snapshot), sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(snapshot, canonical_snapshot)
        self.assertNotIn(b"\r\n", snapshot)

        identity = json.loads(
            (result_root / ".runner_control" / "fixture_run" / "identity.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(identity["git_commit"], "1" * 40)
        self.assertEqual(identity["python_path"], bash_path(self.fake_python))
        self.assertEqual(identity["python_realpath"], bash_path(self.fake_python))
        self.assertEqual(identity["python_sha256"], self.expected_python_sha256)

    def test_control_root_and_stale_lock_must_not_preexist(self) -> None:
        result_root = self.temporary / "existing-control-results"
        control_root = result_root / ".runner_control" / "fixture_run"
        control_root.mkdir(parents=True)
        result = self.run_runner(changes={"RESULT_ROOT": bash_path(result_root)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("control root", result.stderr.lower())
        self.assertEqual(self.calls(), [])

        self.reset_runtime()
        result_root = self.temporary / "stale-lock-results"
        control_parent = result_root / ".runner_control"
        control_parent.mkdir(parents=True)
        stale = control_parent / "gpu_0.lock"
        stale.write_text("stale-lock-sentinel\n", encoding="utf-8")
        result = self.run_runner(
            changes={"RESULT_ROOT": bash_path(result_root), "RUN_ID": "stale_lock"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale", result.stderr.lower())
        self.assertEqual(stale.read_text(encoding="utf-8"), "stale-lock-sentinel\n")
        self.assertEqual(self.calls(), [])

    def test_interstage_python_disk_and_control_identity_are_rechecked(self) -> None:
        enough = str(150 * 1024**3)
        low = str(100 * 1024**3 - 1)
        cases = (
            (
                "python",
                {"FIXTURE_TAMPER_PYTHON_AFTER_STAGE": "preflight"},
                "python sha-256",
            ),
            (
                "disk",
                {"FIXTURE_FREE_BYTES_SEQUENCE": f"{enough},{enough},{low}"},
                "100 gib",
            ),
        )
        for label, changes, message in cases:
            with self.subTest(label=label):
                self.reset_runtime()
                result = self.run_runner(
                    changes={
                        "RESULT_ROOT": bash_path(self.temporary / f"interstage-{label}"),
                        "RUN_ID": f"interstage_{label}",
                        **changes,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr.lower())
                self.assertEqual(
                    [stage for _, stage, _ in self.calls()], ["preflight"]
                )

        self.reset_runtime()
        result_root = self.temporary / "interstage-control"
        control_root = result_root / ".runner_control" / "interstage_control"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": "interstage_control",
                "FIXTURE_REPLACE_PATH_AFTER_STAGE": "preflight",
                "FIXTURE_REPLACE_PATH": bash_path(control_root),
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity", result.stderr.lower())
        self.assertEqual([stage for _, stage, _ in self.calls()], ["preflight"])

    def test_identity_and_prior_stage_logs_stay_authenticated(self) -> None:
        cases = (
            ("identity_replace", "identity.json", "replace", "identity"),
            ("log_symlink", "logs/preflight.out.log", "symlink", "symlink"),
        )
        for label, relative, mode, expected in cases:
            with self.subTest(label=label):
                self.reset_runtime()
                result_root = self.temporary / label
                run_id = label
                target = result_root / ".runner_control" / run_id / relative
                changes = {
                    "RESULT_ROOT": bash_path(result_root),
                    "RUN_ID": run_id,
                    "FIXTURE_CONTROL_MUTATE_AFTER_STAGE": "prepare",
                    "FIXTURE_CONTROL_MUTATION_MODE": mode,
                    "FIXTURE_CONTROL_MUTATE_PATH": bash_path(target),
                }
                if mode == "symlink":
                    changes["FIXTURE_CONTROL_SYMLINK_PATH"] = bash_path(target)
                result = self.run_runner(changes=changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr.lower())
                self.assertEqual(
                    [stage for _, stage, _ in self.calls()],
                    ["preflight", "prepare"],
                )
                self.assertFalse(
                    (result_root / run_id / "verified_completion.json").exists()
                )

    def test_final_marker_is_strict_and_stage_environment_is_minimal(self) -> None:
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(self.temporary / "empty-completion"),
                "RUN_ID": "empty_completion",
                "FIXTURE_COMPLETION_MODE": "empty",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verified_completion", result.stderr.lower())

        self.reset_runtime()
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(self.temporary / "minimal-env"),
                "RUN_ID": "minimal_env",
                "AWS_SECRET_ACCESS_KEY": "must-not-leak",
                "GITHUB_TOKEN": "must-not-leak",
                "WANDB_API_KEY": "must-not-leak",
                "CUSTOM_CREDENTIAL": "must-not-leak",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        environment_log = self.python_env_log.read_text(encoding="utf-8")
        for secret in (
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "WANDB_API_KEY",
            "CUSTOM_CREDENTIAL",
        ):
            self.assertNotIn(secret, environment_log)
        self.assertIn("HF_HUB_OFFLINE=1", environment_log)
        self.assertIn("TRANSFORMERS_OFFLINE=1", environment_log)

    def test_final_inventory_rejects_missing_and_bad_hash_files(self) -> None:
        for run_id in ("missing_inventory_file", "bad_inventory_hash"):
            with self.subTest(run_id=run_id):
                self.reset_runtime()
                result = self.run_runner(
                    changes={
                        "RESULT_ROOT": bash_path(self.temporary / run_id),
                        "RUN_ID": run_id,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("verification inventory", result.stderr.lower())

    def test_final_inventory_binds_required_completion_records(self) -> None:
        for run_id in ("forged_report_binding", "missing_calibration_record"):
            with self.subTest(run_id=run_id):
                self.reset_runtime()
                result = self.run_runner(
                    changes={
                        "RESULT_ROOT": bash_path(self.temporary / run_id),
                        "RUN_ID": run_id,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("verification inventory", result.stderr.lower())

    def test_inventory_leaf_open_is_nonblocking_before_type_check(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('getattr(os, "O_NONBLOCK", 0)', text)

    def test_top_level_markers_are_not_opened_by_blocking_shell_redirection(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn('exec 4<"$completion"', text)
        self.assertNotIn('exec 3<"$inventory"', text)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO integration requires POSIX")
    def test_final_inventory_fifo_fails_without_blocking(self) -> None:
        started = time.monotonic()
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(self.temporary / "fifo-inventory"),
                "RUN_ID": "fifo_inventory_file",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verification inventory", result.stderr.lower())
        self.assertLess(time.monotonic() - started, TIMEOUT_SECONDS / 2)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO integration requires POSIX")
    def test_top_level_marker_fifo_fails_without_blocking(self) -> None:
        started = time.monotonic()
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(self.temporary / "fifo-completion"),
                "RUN_ID": "fifo_completion_marker",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, TIMEOUT_SECONDS / 2)

    def test_terminal_marker_change_after_validator_is_rejected(self) -> None:
        result_root = self.temporary / "terminal-marker"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": "terminal_marker_tamper",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("[camera-translation-hvrfm] verified", result.stdout)
        completion = result_root / "terminal_marker_tamper" / "verified_completion.json"
        validated = Path(f"{completion}.fixture-validated")
        self.assertEqual(completion.read_bytes(), validated.read_bytes())
        self.assertNotEqual(
            (completion.stat().st_dev, completion.stat().st_ino),
            (validated.stat().st_dev, validated.stat().st_ino),
        )

    def test_terminal_barrier_reauthenticates_every_inventory_member(self) -> None:
        for run_id in ("terminal_leaf_in_place", "terminal_leaf_replacement"):
            with self.subTest(run_id=run_id):
                self.reset_runtime()
                result_root = self.temporary / run_id
                result = self.run_runner(
                    changes={
                        "RESULT_ROOT": bash_path(result_root),
                        "RUN_ID": run_id,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("[camera-translation-hvrfm] verified", result.stdout)
                if run_id == "terminal_leaf_replacement":
                    report = result_root / run_id / "reports" / "completed.json"
                    validated = Path(f"{report}.fixture-validated")
                    self.assertEqual(report.read_bytes(), validated.read_bytes())
                    self.assertNotEqual(
                        (report.stat().st_dev, report.stat().st_ino),
                        (validated.stat().st_dev, validated.stat().st_ino),
                    )

    def test_terminal_snapshot_reuses_the_authenticated_control_root(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("snapshot_directories = {}", text)
        self.assertRegex(
            text,
            r"create_snapshot\(\s*control_root,\s*snapshot_relative,\s*"
            r"snapshot_bytes,\s*control_directories\[\"\.\"\]",
        )

    def test_control_ledger_rejects_split_hash_race_and_alternating_inode(self) -> None:
        cases = (
            ("control_ledger_window", "FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH"),
            ("terminal_control_alternating", None),
        )
        for run_id, hook_name in cases:
            with self.subTest(run_id=run_id):
                self.reset_runtime()
                result_root = self.temporary / run_id
                changes = {
                    "RESULT_ROOT": bash_path(result_root),
                    "RUN_ID": run_id,
                }
                if hook_name is not None:
                    prior_log = (
                        result_root
                        / ".runner_control"
                        / run_id
                        / "logs"
                        / "preflight.out.log"
                    )
                    changes[hook_name] = bash_path(prior_log)
                result = self.run_runner(changes=changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("[camera-translation-hvrfm] verified", result.stdout)

        body = RUNNER.read_text(encoding="utf-8").split(
            "require_control_artifacts() {", 1
        )[1].split("\n}", 1)[0]
        self.assertIn("control-ledger", body)
        self.assertNotIn('sha256sum -- "${CONTROL_ARTIFACT_PATHS[@]}"', body)

    def test_final_success_bypasses_printf_hook_and_follows_cleanup(self) -> None:
        result_root = self.temporary / "final-success-boundary"
        run_id = "final_success_boundary"
        report = result_root / run_id / "reports" / "completed.json"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": run_id,
                "FIXTURE_FINAL_PRINTF_TAMPER_PATH": bash_path(report),
                "FIXTURE_CLEANUP_ORDER_HOOK": "1",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(Path(f"{report}.fixture-final-printf").exists())
        inventory = json.loads(
            (result_root / run_id / "manifests" / "verification_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            sha256_file(report), inventory["files"]["reports/completed.json"]["sha256"]
        )
        stdout_lines = [line for line in result.stdout.splitlines() if line]
        self.assertTrue(stdout_lines[-1].startswith("[camera-translation-hvrfm] verified "))

        text = RUNNER.read_text(encoding="utf-8")
        verify = text.rfind("validate_final_completion verify")
        strict_cleanup = text.rfind("release_locks_strict")
        disarm = text.rfind("trap - EXIT")
        success = text.rfind("builtin printf '[camera-translation-hvrfm] verified")
        self.assertGreater(verify, 0)
        self.assertGreater(strict_cleanup, verify)
        self.assertGreater(disarm, strict_cleanup)
        self.assertGreater(success, disarm)

        self.reset_runtime()
        cleanup_failure = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(self.temporary / "cleanup-failure"),
                "RUN_ID": "cleanup_failure",
                "FIXTURE_FAIL_UNLOCK_FD": "8",
            }
        )
        self.assertNotEqual(cleanup_failure.returncode, 0)
        self.assertNotIn(
            "[camera-translation-hvrfm] verified", cleanup_failure.stdout
        )

    def test_jq_fixture_enforces_provenance_instead_of_masking_it(self) -> None:
        completion = self.source / "verified_completion.json"
        completion.write_text("{}\n", encoding="utf-8")
        observed = sha256_file(completion)
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={
                "RESULT_ROOT": bash_path(self.temporary / "semantic-jq"),
                "CTHVRFM_TEST_SOURCE_COMPLETION_SHA256": observed,
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source completion provenance", result.stderr.lower())
        self.assertEqual(self.calls(), [])

    def test_every_stage_rechecks_git_gpu_disk_hashes_and_isolation(self) -> None:
        cases = (
            (
                "git",
                {"FIXTURE_GIT_BRANCH_SEQUENCE": "codex/privileged-conditional-hvrfm|codex/privileged-conditional-hvrfm|main"},
            ),
            (
                "gpu",
                {
                    "FIXTURE_GPU_0_PID_SEQUENCE": (
                        "__EMPTY__|__EMPTY__|8123\\n"
                    )
                },
            ),
            (
                "hash",
                {
                    "FIXTURE_TAMPER_AFTER_STAGE": "preflight",
                    "FIXTURE_TAMPER_PATH": bash_path(self.source / "verified_completion.json"),
                },
            ),
        )
        for label, changes in cases:
            with self.subTest(label=label):
                self._write_authenticated_inputs()
                self.base_env.update(self._digest_environment())
                self.reset_runtime()
                result = self.run_runner(
                    changes={
                        "RESULT_ROOT": bash_path(self.temporary / f"recheck-{label}"),
                        "RUN_ID": f"recheck_{label}",
                        **changes,
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual([stage for _, stage, _ in self.calls()], ["preflight"])

        self._write_authenticated_inputs()
        self.base_env.update(self._digest_environment())
        self.reset_runtime()
        result = self.run_runner(
            arguments=("--preflight-only",),
            changes={"RESULT_ROOT": bash_path(self.source / "nested-output")},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overlap", result.stderr.lower())

    def test_nonempty_stderr_and_size_limit_stop_without_deleting_run(self) -> None:
        result_root = self.temporary / "stderr-results"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "FIXTURE_STDERR_STAGE": "smoke",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("smoke wrote stderr", result.stderr.lower())
        self.assertTrue((result_root / "fixture_run").is_dir())
        self.assertEqual([stage for _, stage, _ in self.calls()], ["preflight", "prepare", "smoke"])

        self.reset_runtime()
        result_root = self.temporary / "size-results"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": "size_run",
                "FIXTURE_DU_KIB_SEQUENCE": "0,0,20971520",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("20 gib", result.stderr.lower())
        self.assertTrue((result_root / "size_run").is_dir())

    def test_run_and_gpu_locks_are_nonblocking(self) -> None:
        for fd, expected in (("9", "control root"), ("8", "gpu")):
            with self.subTest(fd=fd):
                self.reset_runtime()
                result_root = self.temporary / f"locks-{fd}"
                started = self.temporary / f"started-{fd}"
                release = self.temporary / f"release-{fd}"
                first_env = self.environment(
                    {
                        "RESULT_ROOT": bash_path(result_root),
                        "RUN_ID": f"first_{fd}",
                        "FIXTURE_BLOCK_LOCK_FD": fd,
                        "FIXTURE_BLOCK_STARTED": bash_path(started),
                        "FIXTURE_BLOCK_RELEASE": bash_path(release),
                        "FIXTURE_FAIL_STAGE": "preflight",
                    }
                )
                first = subprocess.Popen(
                    self.command(),
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
                        if started.exists() or first.poll() is not None:
                            break
                        time.sleep(0.05)
                    self.assertTrue(started.exists(), f"first runner exited: {first.poll()}")
                    second_changes = {
                        "RESULT_ROOT": bash_path(result_root),
                        "RUN_ID": f"second_{fd}" if fd == "8" else f"first_{fd}",
                        "FIXTURE_RUN_LOCK_STATE": (
                            bash_path(self.temporary / "second-run-lock")
                            if fd == "8"
                            else bash_path(self.run_lock_state)
                        ),
                    }
                    second = self.run_runner(changes=second_changes)
                    self.assertNotEqual(second.returncode, 0)
                    self.assertIn(expected, second.stderr.lower())
                finally:
                    release.touch()
                    stdout, stderr = first.communicate(timeout=TIMEOUT_SECONDS)
                self.assertNotEqual(first.returncode, 0, stderr + stdout)
                self.assertIn("preflight stage failed", stderr.lower())

    def test_create_new_lock_and_log_bind_open_fd_to_path(self) -> None:
        cases = (
            ("run_lock", "run.lock"),
            ("stdout_log", "logs/preflight.out.log"),
        )
        for label, relative in cases:
            with self.subTest(label=label):
                self.reset_runtime()
                result_root = self.temporary / f"fd-bind-{label}"
                run_id = f"fd_bind_{label}"
                target = result_root / ".runner_control" / run_id / relative
                result = self.run_runner(
                    changes={
                        "RESULT_ROOT": bash_path(result_root),
                        "RUN_ID": run_id,
                        "FIXTURE_SWAP_ON_STAT_PATH": bash_path(target),
                    }
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("opened fd", result.stderr.lower())

    def test_lock_handoff_after_unlink_is_not_deleted(self) -> None:
        result_root = self.temporary / "lock-handoff"
        run_id = "lock_handoff"
        run_lock = result_root / ".runner_control" / run_id / "run.lock"
        result = self.run_runner(
            changes={
                "RESULT_ROOT": bash_path(result_root),
                "RUN_ID": run_id,
                "FIXTURE_HANDOFF_LOCK_FD": "9",
                "FIXTURE_HANDOFF_LOCK_PATH": bash_path(run_lock),
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(run_lock.read_text(encoding="utf-8"), "next-owner\n")

    def test_invalid_argument_and_bash_syntax(self) -> None:
        result = self.run_runner(arguments=("--unknown",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stderr.lower())
        self.assertIn("stale lock", result.stderr.lower())
        syntax = subprocess.run(
            [str(BASH), "-n", str(RUNNER)],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
