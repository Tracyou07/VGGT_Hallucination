from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from pre_experiments.camera_translation_hvrfm.data import PublishedTranslationSample

try:
    from pre_experiments.camera_translation_hvrfm import pipeline
except (ImportError, ModuleNotFoundError):
    pipeline = None  # type: ignore[assignment]


SCENES = (
    "scene0000_00",
    "scene0013_02",
    "scene0029_01",
    "scene0084_01",
    "scene0121_01",
    "scene0207_01",
    "scene0280_00",
    "scene0325_01",
    "scene0675_00",
    "scene0691_00",
)
VALIDATION = frozenset({"scene0325_01", "scene0675_00"})
FAILED_GIT = "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c"
FORMAL_GIT = "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _current_git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class UpstreamFixture:
    def __init__(self, root: Path) -> None:
        self.workspace = root
        self.run_root = root / "new_stage_a_prime_run"
        self.source_run = root / "vrfm_camera_20260827T044926Z"
        self.reference_run = (
            root / "privileged_teacher_lift_20260829T012716Z_tolfix"
        )
        self.formal_run = root / "long_short_head_formal_20260828T072407Z"
        self.checkpoint_dir = root / "VGGT-1B"
        self.checkpoint_file = self.checkpoint_dir / "model.safetensors"
        self.checkpoint_file.parent.mkdir(parents=True)
        self.checkpoint_file.write_bytes(b"frozen camera head checkpoint")
        self.checkpoint_sha256 = _sha(self.checkpoint_file)
        self.source_paths: dict[str, Path] = {}
        self.source_sha256: dict[str, str] = {}
        self.long_paths: dict[str, Path] = {}
        self.long_sha256: dict[str, str] = {}
        self.teacher_paths: dict[str, Path] = {}
        self.teacher_sha256: dict[str, str] = {}
        self.formal_paths: dict[str, Path] = {}
        self.formal_sha256: dict[str, str] = {}

        self._write_source()
        self._write_formal()
        self._write_reference()
        self.inputs = self.make_inputs()

    @staticmethod
    def role(scene: str) -> str:
        return "validation" if scene in VALIDATION else "train"

    def _write_source(self) -> None:
        rows = []
        stale_root = Path("/stale/migrated/vrfm/prediction_only/source")
        for scene in SCENES:
            path = self.source_run / "prediction_only" / "source" / f"{scene}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"source:{scene}\n".encode())
            digest = _sha(path)
            self.source_paths[scene] = path
            self.source_sha256[scene] = digest
            rows.append(
                {
                    "scene": scene,
                    "role": self.role(scene),
                    "path": str(stale_root / f"{scene}.npz"),
                    "overlap_count": 8,
                    "sha256": digest,
                }
            )
        self.source_manifest = self.source_run / "manifests" / "source_manifest.json"
        _write_json(
            self.source_manifest,
            {
                "schema": "variational_camera_latent.source.v1",
                "dataset_root": str(stale_root),
                "source_run_digest": "6" * 64,
                "records": rows,
            },
        )
        unsigned = {
            "schema": "variational_camera_latent.verified_completion.v1",
            "signal": "WEAK_SIGNAL",
            "scene_count": 10,
            "overlap_count": 80,
            "candidate_count": 2560,
            "prediction_manifest_sha256": "1" * 64,
            "privileged_manifest_sha256": "2" * 64,
            "report_sha256": "3" * 64,
        }
        self.source_completion = self.source_run / "verified_completion.json"
        _write_json(
            self.source_completion,
            {**unsigned, "completion_digest": _canonical_digest(unsigned)},
        )
        self.source_completion_sha256 = _sha(self.source_completion)
        self.source_manifest_sha256 = _sha(self.source_manifest)

    def _write_formal(self) -> None:
        rows = []
        for index, scene in enumerate(SCENES):
            path = (
                self.formal_run
                / "data"
                / "privileged_labels"
                / f"{scene}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"formal:{scene}\n".encode())
            digest = _sha(path)
            self.formal_paths[scene] = path
            self.formal_sha256[scene] = digest
            rows.append(
                {
                    "scene": scene,
                    "role": self.role(scene),
                    "source_path": str(self.source_paths[scene].resolve()),
                    "source_sha256": self.source_sha256[scene],
                    "long_context_path": str(
                        (
                            self.formal_run
                            / "data"
                            / "long_context"
                            / f"{scene}.npz"
                        ).resolve()
                    ),
                    "long_context_sha256": "0" * 64,
                    "privileged_path": str(path.resolve()),
                    "privileged_sha256": digest,
                    "teacher_frame_count": 300 + 50 * (index % 4),
                }
            )
        self.formal_manifest = self.formal_run / "manifests" / "data_manifest.json"
        _write_json(
            self.formal_manifest,
            {
                "schema": "long_short_camera_head.data_manifest.v1",
                "git_revision": FORMAL_GIT,
                "source_run": str(self.source_run.resolve()),
                "source_manifest_sha256": self.source_manifest_sha256,
                "prepared_root": str((self.workspace / "prepared").resolve()),
                "checkpoint_dir": str(self.checkpoint_dir.resolve()),
                "base_checkpoint_sha256": self.checkpoint_sha256,
                "records": rows,
            },
        )
        marker = {
            "schema": "long_short_camera_head.verified_completion.v1",
            "git_revision": FORMAL_GIT,
            "verifier_git_revision": FORMAL_GIT,
            "source_manifest_sha256": self.source_manifest_sha256,
            "base_checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": "4" * 64,
            "data_manifest_sha256": _sha(self.formal_manifest),
            "test_evidence_sha256": "5" * 64,
            "stage_completion_sha256": {
                "evaluation_gt_only": "6" * 64,
                "evaluation_long_short": "7" * 64,
                "smoke": "8" * 64,
                "training_gt_only": "9" * 64,
                "training_long_short": "a" * 64,
            },
            "scene_count": 10,
            "train_scene_count": 8,
            "locked_replay_scene_count": 2,
            "classification": "NO_SOURCE_HEAD_SIGNAL",
            "report_sha256": "b" * 64,
            "artifacts": [
                {
                    "scene": scene,
                    "variant": variant,
                    "checkpoint_sha256": "c" * 64,
                    "prediction_sha256": "d" * 64,
                    "evaluation_sha256": "e" * 64,
                }
                for scene in SCENES
                for variant in ("gt_only", "long_short")
            ],
            "inference_leakage_audit": True,
            "formal_protocol_sha256": "f" * 64,
        }
        self.formal_completion = self.formal_run / "verified_completion.json"
        _write_json(self.formal_completion, marker)
        self.formal_completion_sha256 = _sha(self.formal_completion)
        self.formal_manifest_sha256 = _sha(self.formal_manifest)

    def _write_reference(self) -> None:
        long_rows = []
        teacher_rows = []
        inventory_files: dict[str, str] = {}
        formal_rows = json.loads(self.formal_manifest.read_text())["records"]
        formal_by_scene = {row["scene"]: row for row in formal_rows}
        for scene in SCENES:
            long_path = (
                self.reference_run
                / "prediction_only"
                / "long_context"
                / f"{scene}.npz"
            )
            long_path.parent.mkdir(parents=True, exist_ok=True)
            long_path.write_bytes(f"long:{scene}\n".encode())
            long_digest = _sha(long_path)
            self.long_paths[scene] = long_path
            self.long_sha256[scene] = long_digest
            formal_by_scene[scene]["long_context_sha256"] = long_digest
            long_rows.append(
                {
                    "scene": scene,
                    "role": self.role(scene),
                    "file": f"{scene}.npz",
                    "sha256": long_digest,
                    "source_sha256": self.source_sha256[scene],
                }
            )
            teacher_path = (
                self.reference_run
                / "privileged_labels"
                / "teacher"
                / f"{scene}.npz"
            )
            teacher_path.parent.mkdir(parents=True, exist_ok=True)
            teacher_path.write_bytes(f"teacher:{scene}\n".encode())
            teacher_digest = _sha(teacher_path)
            self.teacher_paths[scene] = teacher_path
            self.teacher_sha256[scene] = teacher_digest
            teacher_rows.append(
                {
                    "scene": scene,
                    "role": self.role(scene),
                    "file": f"privileged_labels/teacher/{scene}.npz",
                    "sha256": teacher_digest,
                    "formal_label_sha256": self.formal_sha256[scene],
                }
            )
            inventory_files[
                f"prediction_only/long_context/{scene}.npz"
            ] = long_digest
            inventory_files[
                f"privileged_labels/teacher/{scene}.npz"
            ] = teacher_digest

        # The formal manifest is finalized only after the reference long digests exist.
        formal_payload = json.loads(self.formal_manifest.read_text())
        formal_payload["records"] = [formal_by_scene[scene] for scene in SCENES]
        _write_json(self.formal_manifest, formal_payload)
        self.formal_manifest_sha256 = _sha(self.formal_manifest)
        formal_marker = json.loads(self.formal_completion.read_text())
        formal_marker["data_manifest_sha256"] = self.formal_manifest_sha256
        _write_json(self.formal_completion, formal_marker)
        self.formal_completion_sha256 = _sha(self.formal_completion)

        self.long_manifest = self.reference_run / "manifests" / "long_context.json"
        _write_json(
            self.long_manifest,
            {
                "schema": "conditional_hierarchical_vrfm.long_context_manifest.v1",
                "records": long_rows,
            },
        )
        self.teacher_manifest = self.reference_run / "manifests" / "teacher.json"
        _write_json(
            self.teacher_manifest,
            {
                "schema": "conditional_hierarchical_vrfm.teacher_manifest.v1",
                "git_commit": FAILED_GIT,
                "checkpoint_sha256": self.checkpoint_sha256,
                "formal_completion_sha256": self.formal_completion_sha256,
                "formal_data_manifest_sha256": self.formal_manifest_sha256,
                "teacher_upper_bound": {
                    "scene_count": 10,
                    "positive_scene_count": 10,
                    "mean_coverage": 0.89,
                    "mean_utility": 0.1293578270771188,
                },
                "records": teacher_rows,
            },
        )
        self.config = self.reference_run / "config.json"
        _write_json(
            self.config,
            {
                "schema": "conditional_hierarchical_vrfm.run_config.v1",
                "git_commit": FAILED_GIT,
                "checkpoint_sha256": self.checkpoint_sha256,
                "basis_sha256": "89fecc83d51be8a1923a0c177c20b45dec8d8aa611fae0b615ef9293511dd213",
                "long_manifest_sha256": _sha(self.long_manifest),
                "teacher_manifest_sha256": _sha(self.teacher_manifest),
                "source_run": str(self.source_run.resolve()),
                "source_manifest_sha256": self.source_manifest_sha256,
                "formal_run_root": str(self.formal_run.resolve()),
                "formal_completion_sha256": self.formal_completion_sha256,
                "formal_data_manifest_sha256": self.formal_manifest_sha256,
                "smoke_scene": "scene0000_00",
                "smoke_steps": 20,
                "calibration_steps": 250,
                "scene_count": 10,
                "variant_count": 4,
            },
        )
        self.report_json = self.reference_run / "reports" / "stage_a.json"
        _write_json(
            self.report_json,
            {
                "schema": "conditional_hierarchical_vrfm.stage_a_report.v1",
                "git_commit": FAILED_GIT,
                "classification": "LATENT_LIFT_FAILED",
                "failed_gates": [
                    "teacher_retention",
                    "per_scene_harm",
                    "rotation_guard",
                    "uncovered_anchor",
                ],
                "scene_metrics": [{"scene": scene} for scene in SCENES],
                "provenance": {
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "basis_sha256": "89fecc83d51be8a1923a0c177c20b45dec8d8aa611fae0b615ef9293511dd213",
                    "long_manifest_sha256": _sha(self.long_manifest),
                    "teacher_manifest_sha256": _sha(self.teacher_manifest),
                },
            },
        )
        self.report_markdown = self.reference_run / "reports" / "stage_a.md"
        self.report_markdown.write_text("# LATENT_LIFT_FAILED\n", encoding="utf-8")
        required = {
            "config.json": self.config,
            "manifests/long_context.json": self.long_manifest,
            "manifests/teacher.json": self.teacher_manifest,
            "reports/stage_a.json": self.report_json,
            "reports/stage_a.md": self.report_markdown,
        }
        inventory_files.update({name: _sha(path) for name, path in required.items()})
        expected_inventory = {
            "config.json",
            "manifests/preflight_evidence.json",
            "manifests/long_context.json",
            "manifests/teacher.json",
            "smoke/completed.json",
            "calibration/completed.json",
            "reports/stage_a.json",
            "reports/stage_a.md",
            *(f"logs/preflight_{index}.log" for index in range(4)),
        }
        for scene in SCENES:
            expected_inventory.update(
                {
                    f"prediction_only/long_context/{scene}.npz",
                    f"privileged_labels/teacher/{scene}.npz",
                    f"privileged_labels/latent_targets/{scene}.npz",
                    *(
                        f"checkpoints/calibration/{scene}/variant_{variant}.pt"
                        for variant in range(4)
                    ),
                }
            )
        expected_inventory.add("smoke/latent_targets/scene0000_00.npz")
        expected_inventory.update(
            f"smoke/checkpoints/scene0000_00/variant_{variant}.pt"
            for variant in range(4)
        )
        self.unused_inventory_paths: list[Path] = []
        for relative in sorted(expected_inventory - set(inventory_files)):
            path = self.reference_run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"inventory:{relative}\n".encode())
            inventory_files[relative] = _sha(path)
            self.unused_inventory_paths.append(path)
        assert len(expected_inventory) == 87
        assert set(inventory_files) == expected_inventory
        self.inventory = self.reference_run / "manifests" / "verification_inventory.json"
        _write_json(
            self.inventory,
            {
                "schema": "conditional_hierarchical_vrfm.verification_inventory.v1",
                "git_commit": FAILED_GIT,
                "classification": "LATENT_LIFT_FAILED",
                "files": inventory_files,
            },
        )
        unsigned = {
            "schema": "conditional_hierarchical_vrfm.verified_completion.v1",
            "git_commit": FAILED_GIT,
            "classification": "LATENT_LIFT_FAILED",
            "inventory_sha256": _sha(self.inventory),
            "file_count": len(inventory_files),
        }
        self.reference_completion = self.reference_run / "verified_completion.json"
        _write_json(
            self.reference_completion,
            {**unsigned, "completion_digest": _canonical_digest(unsigned)},
        )
        self.reference_completion_sha256 = _sha(self.reference_completion)

    def make_inputs(self, **overrides: object):
        module = pipeline
        assert module is not None
        values = {
            "run_root": self.run_root,
            "git_commit": _current_git(),
            "source_run": self.source_run,
            "reference_run": self.reference_run,
            "formal_run": self.formal_run,
            "checkpoint_dir": self.checkpoint_dir,
            "expected_source_completion_sha256": self.source_completion_sha256,
            "expected_reference_completion_sha256": self.reference_completion_sha256,
            "expected_formal_completion_sha256": self.formal_completion_sha256,
            "expected_checkpoint_sha256": self.checkpoint_sha256,
            "device": torch.device("cpu"),
        }
        values.update(overrides)
        return module.PipelineInputs(**values)

    def formal_authentication(self) -> dict[str, object]:
        return {
            "labels": dict(self.formal_paths),
            "completion_sha256": self.formal_completion_sha256,
            "data_manifest_sha256": self.formal_manifest_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "formal_root": self.formal_run.resolve(),
        }

    @contextmanager
    def patched_authentication(self):
        module = pipeline
        assert module is not None
        with ExitStack() as stack:
            constants = {
                "FROZEN_SOURCE_COMPLETION_SHA256": self.source_completion_sha256,
                "FROZEN_REFERENCE_COMPLETION_SHA256": self.reference_completion_sha256,
                "FROZEN_FORMAL_COMPLETION_SHA256": self.formal_completion_sha256,
                "FROZEN_CHECKPOINT_SHA256": self.checkpoint_sha256,
                "FROZEN_SOURCE_MANIFEST_SHA256": self.source_manifest_sha256,
                "FROZEN_FORMAL_DATA_MANIFEST_SHA256": self.formal_manifest_sha256,
                "FROZEN_REFERENCE_LONG_MANIFEST_SHA256": _sha(self.long_manifest),
            }
            for name, value in constants.items():
                stack.enter_context(mock.patch.object(module, name, value))
            stack.enter_context(
                mock.patch.object(
                    module,
                    "_decode_source_snapshot",
                    side_effect=lambda payload, **_: {
                        "scene": payload.decode().strip().split(":", 1)[1]
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "_scene_from_source",
                    side_effect=lambda arrays: arrays["scene"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    module,
                    "_decode_old_long_snapshot",
                    side_effect=lambda payload, **_: {
                        "scene": np.asarray(
                            payload.decode().strip().split(":", 1)[1]
                        ),
                        "source_sha256": np.asarray(
                            self.source_sha256[
                                payload.decode().strip().split(":", 1)[1]
                            ]
                        ),
                    },
                )
            )
            controls = stack.enter_context(
                mock.patch.object(
                    module,
                    "_decode_teacher_snapshot",
                    side_effect=lambda payload, **_: SimpleNamespace(
                        scene=payload.decode().strip().split(":", 1)[1]
                    ),
                )
            )
            formal = stack.enter_context(
                mock.patch.object(
                    module,
                    "_decode_formal_label_snapshot",
                    side_effect=lambda payload, **_: {
                        "scene": np.asarray(
                            payload.decode().strip().split(":", 1)[1]
                        ),
                        "source_sha256": np.asarray(
                            self.source_sha256[
                                payload.decode().strip().split(":", 1)[1]
                            ]
                        ),
                        "checkpoint_sha256": np.asarray(self.checkpoint_sha256),
                    },
                )
            )
            yield SimpleNamespace(controls=controls, formal=formal)


class PipelineTask4aTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def api(self):
        self.assertIsNotNone(pipeline, "Task 4a pipeline module is missing")
        return pipeline

    def fixture(self) -> UpstreamFixture:
        self.api()
        return UpstreamFixture(self.root)

    def _publisher(self, fixture: UpstreamFixture, completion_seen: list[bool]):
        def publish(output_root: Path, **kwargs: object) -> PublishedTranslationSample:
            scene = kwargs["source_record"].scene  # type: ignore[union-attr]
            completion_seen.append(
                (fixture.run_root / "prepare/completed.json").exists()
            )
            paths = {
                "long": Path(output_root) / "prediction_only/long_context" / f"{scene}.npz",
                "short": Path(output_root) / "privileged_training/short_context" / f"{scene}.npz",
                "quality": Path(output_root) / "privileged_labels/quality" / f"{scene}.npz",
                "target": Path(output_root) / "privileged_labels/translation_targets" / f"{scene}.npz",
            }
            digests = {}
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{name}:{scene}\n".encode())
                digests[name] = _sha(path)
            return PublishedTranslationSample(
                sample_id=f"{scene}:frames_500",
                scene=scene,
                role=fixture.role(scene),
                long_path=paths["long"],
                short_path=paths["short"],
                quality_path=paths["quality"],
                target_path=paths["target"],
                long_sha256=digests["long"],
                short_sha256=digests["short"],
                quality_sha256=digests["quality"],
                target_sha256=digests["target"],
            )

        return publish

    def _bundle_loader(
        self,
        fixture: UpstreamFixture,
        overrides: dict[str, str] | None = None,
    ):
        changed = {} if overrides is None else dict(overrides)

        def load(
            long_bytes: bytes,
            short_bytes: bytes,
            target_bytes: bytes,
            quality_bytes: bytes,
        ) -> dict[str, dict[str, np.ndarray]]:
            del short_bytes, target_bytes, quality_bytes
            scene = long_bytes.decode("utf-8").strip().split(":", 1)[1]
            common = {
                "sample_id": changed.get("sample_id", f"{scene}:frames_500"),
                "scene": changed.get("scene", scene),
                "source_sha256": changed.get(
                    "source_sha256", fixture.source_sha256[scene]
                ),
                "checkpoint_sha256": changed.get(
                    "checkpoint_sha256", fixture.checkpoint_sha256
                ),
                "git_commit": changed.get("git_commit", fixture.inputs.git_commit),
            }
            bundle = {
                name: {
                    field: np.asarray(value)
                    for field, value in common.items()
                }
                for name in ("long", "short", "target", "quality")
            }
            teacher = changed.get(
                "teacher_reference_sha256", fixture.teacher_sha256[scene]
            )
            bundle["quality"]["formal_label_sha256"] = np.asarray(
                changed.get("formal_label_sha256", fixture.formal_sha256[scene])
            )
            bundle["quality"]["teacher_reference_sha256"] = np.asarray(teacher)
            bundle["target"]["teacher_reference_sha256"] = np.asarray(teacher)
            return bundle

        return load

    def test_public_contract_and_frozen_failed_marker_digest(self) -> None:
        module = self.api()
        self.assertEqual(
            module.FROZEN_REFERENCE_COMPLETION_SHA256,
            "7e63ca36e6fc4c08772e3356255f84c2853c9d46310ae546cc5e53dc1792048c",
        )
        self.assertEqual(
            tuple(module.PipelineInputs.__dataclass_fields__),
            (
                "run_root",
                "git_commit",
                "source_run",
                "reference_run",
                "formal_run",
                "checkpoint_dir",
                "expected_source_completion_sha256",
                "expected_reference_completion_sha256",
                "expected_formal_completion_sha256",
                "expected_checkpoint_sha256",
                "device",
            ),
        )
        self.assertEqual(
            tuple(module.AuthenticatedSceneInputs.__dataclass_fields__),
            (
                "scene",
                "role",
                "source_path",
                "source_record",
                "long_context_path",
                "long_context_sha256",
                "teacher_reference_path",
                "teacher_reference_sha256",
                "formal_label_path",
                "formal_label_sha256",
            ),
        )

    def test_authenticate_upstream_rebinds_stale_sources_and_checks_every_scene(self) -> None:
        module = self.api()
        fixture = self.fixture()
        with fixture.patched_authentication() as calls:
            authenticated = module.authenticate_upstream(fixture.inputs)
        self.assertEqual([row.scene for row in authenticated], list(SCENES))
        self.assertEqual(calls.controls.call_count, 10)
        self.assertEqual(calls.formal.call_count, 10)
        for row in authenticated:
            expected = fixture.source_run / "prediction_only/source" / f"{row.scene}.npz"
            self.assertEqual(row.source_path, expected.resolve())
            self.assertEqual(row.source_record.path, expected.resolve())
            self.assertEqual(row.source_record.overlap_count, 8)
            self.assertEqual(row.source_record.sha256, fixture.source_sha256[row.scene])
            self.assertNotIn("stale", str(row.source_path))
            self.assertEqual(row.teacher_reference_sha256, fixture.teacher_sha256[row.scene])
            self.assertEqual(row.formal_label_sha256, fixture.formal_sha256[row.scene])

    def test_authentication_rejects_each_digest_edge(self) -> None:
        module = self.api()
        cases = (
            ("source shard", lambda f: f.source_paths[SCENES[0]].write_bytes(b"changed")),
            ("long", lambda f: f.long_paths[SCENES[0]].write_bytes(b"changed")),
            ("teacher", lambda f: f.teacher_paths[SCENES[0]].write_bytes(b"changed")),
            ("checkpoint", lambda f: f.checkpoint_file.write_bytes(b"changed")),
            ("report", lambda f: f.report_json.write_bytes(b"{}\n")),
            ("formal completion", lambda f: f.formal_completion.write_bytes(b"{}\n")),
            (
                "reference inventory artifact",
                lambda f: f.unused_inventory_paths[0].write_bytes(b"changed"),
            ),
        )
        for index, (label, mutate) in enumerate(cases):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = UpstreamFixture(Path(directory))
                mutate(fixture)
                with fixture.patched_authentication(), self.assertRaisesRegex(
                    ValueError, label
                ):
                    module.authenticate_upstream(fixture.inputs)

    def test_authentication_terminal_barrier_revalidates_unused_inventory(self) -> None:
        module = self.api()
        fixture = self.fixture()
        unused = fixture.unused_inventory_paths[0]
        real_snapshot_checkpoint = module._snapshot_streaming_file

        def mutate_after_checkpoint_snapshot(
            path: Path, expected_sha256: str, *, label: str
        ):
            snapshot = real_snapshot_checkpoint(
                path, expected_sha256, label=label
            )
            if label == "checkpoint":
                unused.write_bytes(b"changed after its initial inventory check")
            return snapshot

        with fixture.patched_authentication(), mock.patch.object(
            module,
            "_snapshot_streaming_file",
            side_effect=mutate_after_checkpoint_snapshot,
        ), self.assertRaisesRegex(
            ValueError,
            "reference inventory artifact .* changed during authentication",
        ):
            module.authenticate_upstream(fixture.inputs)

    def test_preflight_failure_does_not_create_run_root_then_success_is_signed_idempotent(self) -> None:
        module = self.api()
        fixture = self.fixture()
        fixture.source_paths[SCENES[3]].write_bytes(b"corrupt")
        with fixture.patched_authentication(), self.assertRaises(ValueError):
            module.run_preflight(fixture.inputs)
        self.assertFalse(fixture.run_root.exists())

        # Restore the authenticated byte and then publish evidence.
        fixture.source_paths[SCENES[3]].write_bytes(f"source:{SCENES[3]}\n".encode())
        with fixture.patched_authentication():
            evidence_path = module.run_preflight(fixture.inputs)
            first = evidence_path.read_bytes()
            self.assertEqual(module.run_preflight(fixture.inputs), evidence_path)
        self.assertEqual(evidence_path.read_bytes(), first)
        self.assertEqual(
            {path.relative_to(fixture.run_root).as_posix() for path in fixture.run_root.rglob("*") if path.is_file()},
            {"manifests/preflight_evidence.json"},
        )
        evidence = json.loads(first)
        self.assertEqual(set(evidence), module.PREFLIGHT_EVIDENCE_FIELDS)
        digest = evidence.pop("completion_digest")
        self.assertEqual(digest, _canonical_digest(evidence))
        self.assertEqual(evidence["stage"], "preflight")
        self.assertEqual(evidence["run_id"], fixture.run_root.name)
        self.assertEqual([row["scene"] for row in evidence["scene_bindings"]], list(SCENES))

    def test_preflight_refuses_even_an_existing_empty_or_foreign_root(self) -> None:
        module = self.api()
        for foreign in (None, "preflight.log", "orphan.tmp"):
            with self.subTest(foreign=foreign), tempfile.TemporaryDirectory() as directory:
                fixture = UpstreamFixture(Path(directory))
                fixture.run_root.mkdir()
                if foreign is not None:
                    (fixture.run_root / foreign).write_bytes(b"preserve")
                with fixture.patched_authentication(), self.assertRaisesRegex(
                    ValueError, "existing run root|new run ID"
                ):
                    module.run_preflight(fixture.inputs)
                self.assertTrue(fixture.run_root.is_dir())
                if foreign is not None:
                    self.assertEqual((fixture.run_root / foreign).read_bytes(), b"preserve")

    def test_preflight_publication_failure_removes_owned_empty_run_root(self) -> None:
        module = self.api()
        fixture = self.fixture()
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "_publish_bytes_create_absent",
            side_effect=RuntimeError("injected evidence publication failure"),
        ), self.assertRaisesRegex(RuntimeError, "evidence publication failure"):
            module.run_preflight(fixture.inputs)
        self.assertFalse(fixture.run_root.exists())

    def test_preflight_cleanup_preserves_foreign_or_mismatched_race(self) -> None:
        module = self.api()
        for race in ("foreign", "mismatched_evidence"):
            with self.subTest(race=race), tempfile.TemporaryDirectory() as directory:
                fixture = UpstreamFixture(Path(directory))
                evidence = fixture.run_root / "manifests/preflight_evidence.json"

                def fail_after_race(path: Path, content: bytes) -> None:
                    if race == "foreign":
                        raced = fixture.run_root / "foreign.bin"
                    else:
                        raced = evidence
                    raced.parent.mkdir(parents=True, exist_ok=True)
                    raced.write_bytes(b"concurrent writer")
                    raise RuntimeError("injected evidence publication failure")

                with fixture.patched_authentication(), mock.patch.object(
                    module,
                    "_publish_bytes_create_absent",
                    side_effect=fail_after_race,
                ), self.assertRaisesRegex(
                    ValueError,
                    "preflight publication failed and run-root rollback failed",
                ):
                    module.run_preflight(fixture.inputs)
                self.assertTrue(fixture.run_root.is_dir())
                if race == "foreign":
                    self.assertEqual(
                        (fixture.run_root / "foreign.bin").read_bytes(),
                        b"concurrent writer",
                    )
                else:
                    self.assertEqual(evidence.read_bytes(), b"concurrent writer")

    def test_run_root_and_upstream_trees_must_be_bidirectionally_disjoint(self) -> None:
        module = self.api()
        for upstream_name in (
            "source_run",
            "reference_run",
            "formal_run",
            "checkpoint_dir",
        ):
            with self.subTest(run_root_below=upstream_name), tempfile.TemporaryDirectory() as directory:
                fixture = UpstreamFixture(Path(directory))
                upstream = Path(getattr(fixture, upstream_name))
                nested_root = upstream / "nested_stage_a_prime"
                inputs = fixture.make_inputs(run_root=nested_root)
                with fixture.patched_authentication(), self.assertRaisesRegex(
                    ValueError, "physically isolated"
                ):
                    module.run_preflight(inputs)
                self.assertFalse(nested_root.exists())

        with tempfile.TemporaryDirectory() as directory:
            fixture = UpstreamFixture(Path(directory))
            inputs = fixture.make_inputs(run_root=fixture.workspace)
            with fixture.patched_authentication(), self.assertRaisesRegex(
                ValueError, "physically isolated"
            ):
                module.authenticate_upstream(inputs)

    def test_authentication_decodes_one_immutable_snapshot_and_rehashes_original(self) -> None:
        module = self.api()
        fixture = self.fixture()
        scene = SCENES[0]
        original = fixture.source_paths[scene].read_bytes()
        with fixture.patched_authentication():
            decoder = module._decode_source_snapshot

            def restore_attack(payload: bytes, **kwargs: object):
                self.assertIs(type(payload), bytes)
                fixture.source_paths[scene].write_bytes(b"malicious replacement")
                try:
                    return {"scene": scene}
                finally:
                    fixture.source_paths[scene].write_bytes(original)

            with mock.patch.object(
                module,
                "_decode_source_snapshot",
                side_effect=lambda payload, **kwargs: (
                    restore_attack(payload, **kwargs)
                    if payload == original
                    else decoder(payload, **kwargs)
                ),
            ), self.assertRaisesRegex(ValueError, "changed during authentication"):
                module.authenticate_upstream(fixture.inputs)

            def mutate_after_decode(payload: bytes, **kwargs: object):
                result = (
                    {"scene": scene}
                    if payload == original
                    else decoder(payload, **kwargs)
                )
                if payload == original:
                    fixture.source_paths[scene].write_bytes(b"changed after parse")
                return result

            with mock.patch.object(
                module, "_decode_source_snapshot", side_effect=mutate_after_decode
            ), self.assertRaisesRegex(ValueError, "changed during authentication"):
                module.authenticate_upstream(fixture.inputs)

    def test_prepare_publishes_exact_signed_manifests_completion_last_and_loads(self) -> None:
        module = self.api()
        fixture = self.fixture()
        completion_seen: list[bool] = []
        publisher = self._publisher(fixture, completion_seen)

        def load_private_checkpoint(directory: Path):
            private = Path(directory)
            self.assertNotEqual(private.resolve(), fixture.checkpoint_dir.resolve())
            self.assertFalse(
                private.is_relative_to(fixture.run_root),
                "checkpoint staging must not be published below run_root",
            )
            self.assertEqual(
                (private / fixture.checkpoint_file.name).read_bytes(),
                fixture.checkpoint_file.read_bytes(),
            )
            return torch.nn.Identity(), fixture.checkpoint_sha256

        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            side_effect=load_private_checkpoint,
        ) as load_checkpoint, mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ) as publish, mock.patch.object(
            module,
            "load_bound_bundle_bytes",
            side_effect=self._bundle_loader(fixture),
        ):
            module.run_preflight(fixture.inputs)
            completion = module.run_prepare(fixture.inputs)
            cohort = module.load_published_cohort(fixture.run_root)

        self.assertEqual(completion, fixture.run_root / "prepare/completed.json")
        self.assertEqual(load_checkpoint.call_count, 1)
        self.assertEqual(publish.call_count, 10)
        self.assertEqual(completion_seen, [False] * 10)
        self.assertEqual([sample.scene for sample in cohort], list(SCENES))
        config = json.loads((fixture.run_root / "config.json").read_text())
        long_manifest = json.loads(
            (fixture.run_root / "manifests/long_context.json").read_text()
        )
        cohort_manifest = json.loads(
            (fixture.run_root / "manifests/cohort.json").read_text()
        )
        marker = json.loads(completion.read_text())
        self.assertEqual(set(config), module.RUN_CONFIG_FIELDS)
        self.assertEqual(set(long_manifest), module.LONG_CONTEXT_MANIFEST_FIELDS)
        self.assertTrue(
            all(set(row) == module.LONG_CONTEXT_RECORD_FIELDS for row in long_manifest["records"])
        )
        self.assertEqual(set(cohort_manifest), module.COHORT_MANIFEST_FIELDS)
        self.assertTrue(
            all(set(row) == module.COHORT_RECORD_FIELDS for row in cohort_manifest["records"])
        )
        self.assertEqual(set(marker), module.STAGE_COMPLETION_FIELDS)
        unsigned = dict(marker)
        digest = unsigned.pop("completion_digest")
        self.assertEqual(digest, _canonical_digest(unsigned))
        self.assertEqual(marker["stage"], "prepare")
        self.assertEqual(
            marker["previous_marker_sha256"],
            _sha(fixture.run_root / "manifests/preflight_evidence.json"),
        )
        self.assertEqual(marker["files"]["config.json"], _sha(fixture.run_root / "config.json"))
        serialized_long = json.dumps(long_manifest).lower()
        for forbidden in ("short", "teacher", "privileged", "gt", "prepared"):
            self.assertNotIn(forbidden, serialized_long)
        prediction_files = {
            path.relative_to(fixture.run_root / "prediction_only").as_posix()
            for path in (fixture.run_root / "prediction_only").rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            prediction_files,
            {f"long_context/{scene}.npz" for scene in SCENES},
        )

    def test_prepare_resume_never_loads_model_or_overwrites_completed_bytes(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module,
            "load_bound_bundle_bytes",
            side_effect=self._bundle_loader(fixture),
        ):
            module.run_preflight(fixture.inputs)
            module.run_prepare(fixture.inputs)
            before = {
                path.relative_to(fixture.run_root).as_posix(): path.read_bytes()
                for path in fixture.run_root.rglob("*")
                if path.is_file()
            }
            with mock.patch.object(
                module, "load_base_camera_head", side_effect=AssertionError("model loaded")
            ), mock.patch.object(
                module, "publish_translation_sample", side_effect=AssertionError("overwritten")
            ):
                module.run_prepare(fixture.inputs)
            after = {
                path.relative_to(fixture.run_root).as_posix(): path.read_bytes()
                for path in fixture.run_root.rglob("*")
                if path.is_file()
            }
        self.assertEqual(after, before)

    def test_prepare_data_race_is_create_if_absent_and_preserves_competitor(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        target = (
            fixture.run_root
            / "prediction_only"
            / "long_context"
            / f"{SCENES[0]}.npz"
        )
        real_publish = module._publish_bytes_create_absent

        def race(path: Path, content: bytes) -> None:
            if Path(path) == target and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"competing writer")
            real_publish(path, content)

        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", return_value={}
        ):
            module.run_preflight(fixture.inputs)
            with mock.patch.object(
                module, "_publish_bytes_create_absent", side_effect=race
            ), self.assertRaisesRegex(ValueError, "without overwrite|already exists"):
                module.run_prepare(fixture.inputs)
        self.assertEqual(target.read_bytes(), b"competing writer")
        self.assertFalse((fixture.run_root / "prepare/completed.json").exists())

    def test_prepare_checkpoint_swap_restore_is_rejected_after_private_load(self) -> None:
        module = self.api()
        fixture = self.fixture()
        original = fixture.checkpoint_file.read_bytes()

        def swap_restore(directory: Path):
            self.assertNotEqual(Path(directory).resolve(), fixture.checkpoint_dir.resolve())
            fixture.checkpoint_file.write_bytes(b"malicious checkpoint")
            fixture.checkpoint_file.write_bytes(original)
            return torch.nn.Identity(), fixture.checkpoint_sha256

        with fixture.patched_authentication(), mock.patch.object(
            module, "load_base_camera_head", side_effect=swap_restore
        ), mock.patch.object(
            module,
            "publish_translation_sample",
            side_effect=AssertionError("publisher reached after checkpoint attack"),
        ):
            module.run_preflight(fixture.inputs)
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                module.run_prepare(fixture.inputs)
        self.assertEqual(fixture.checkpoint_file.read_bytes(), original)
        self.assertFalse((fixture.run_root / "prepare/completed.json").exists())

    def test_prepare_rejects_preflight_marker_mutation_and_swap_restore(self) -> None:
        module = self.api()
        for restore in (False, True):
            with self.subTest(restore=restore), tempfile.TemporaryDirectory() as directory:
                fixture = UpstreamFixture(Path(directory))
                publisher = self._publisher(fixture, [])
                evidence = fixture.run_root / "manifests/preflight_evidence.json"
                with fixture.patched_authentication(), mock.patch.object(
                    module, "publish_translation_sample", side_effect=publisher
                ), mock.patch.object(
                    module, "load_bound_bundle_bytes", return_value={}
                ):
                    module.run_preflight(fixture.inputs)
                    original = evidence.read_bytes()

                    def mutate_during_model_load(directory: Path):
                        self.assertNotEqual(
                            Path(directory).resolve(), fixture.checkpoint_dir.resolve()
                        )
                        evidence.write_bytes(b"mutated preflight marker")
                        if restore:
                            evidence.write_bytes(original)
                        return torch.nn.Identity(), fixture.checkpoint_sha256

                    with mock.patch.object(
                        module,
                        "load_base_camera_head",
                        side_effect=mutate_during_model_load,
                    ), self.assertRaisesRegex(
                        ValueError, "preflight.*changed during authentication"
                    ):
                        module.run_prepare(fixture.inputs)
                self.assertFalse(
                    (fixture.run_root / "prepare/completed.json").exists()
                )
                self.assertEqual(
                    evidence.read_bytes(),
                    original if restore else b"mutated preflight marker",
                )

    def test_prepare_keeps_entire_upstream_graph_stable_through_compute(self) -> None:
        module = self.api()
        for attack in ("unused_during_model", "long_during_publisher"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as directory:
                fixture = UpstreamFixture(Path(directory))
                base_publisher = self._publisher(fixture, [])
                target = (
                    fixture.unused_inventory_paths[0]
                    if attack == "unused_during_model"
                    else fixture.long_paths[SCENES[0]]
                )
                mutated = False

                def load_model(_: Path):
                    nonlocal mutated
                    if attack == "unused_during_model":
                        target.write_bytes(b"mutated during model loading")
                        mutated = True
                    return torch.nn.Identity(), fixture.checkpoint_sha256

                def publish(output_root: Path, **kwargs: object):
                    nonlocal mutated
                    if attack == "long_during_publisher" and not mutated:
                        target.write_bytes(b"mutated during sample publication")
                        mutated = True
                    return base_publisher(output_root, **kwargs)

                with fixture.patched_authentication(), mock.patch.object(
                    module, "load_base_camera_head", side_effect=load_model
                ), mock.patch.object(
                    module, "publish_translation_sample", side_effect=publish
                ), mock.patch.object(
                    module, "load_bound_bundle_bytes", return_value={}
                ):
                    module.run_preflight(fixture.inputs)
                    with self.assertRaisesRegex(
                        ValueError, "changed during authentication"
                    ):
                        module.run_prepare(fixture.inputs)
                self.assertTrue(mutated)
                self.assertFalse(
                    (fixture.run_root / "prepare/completed.json").exists()
                )

    def test_prepare_revalidates_dependencies_after_completion_publication(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        completion = fixture.run_root / "prepare/completed.json"
        config = fixture.run_root / "config.json"
        real_publish = module._publish_bytes_create_absent

        def mutate_during_completion(path: Path, content: bytes) -> None:
            if Path(path) == completion:
                config.write_bytes(b"mutated during completion publication")
            real_publish(path, content)

        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", return_value={}
        ):
            module.run_preflight(fixture.inputs)
            with mock.patch.object(
                module,
                "_publish_bytes_create_absent",
                side_effect=mutate_during_completion,
            ), self.assertRaisesRegex(ValueError, "changed during authentication"):
                module.run_prepare(fixture.inputs)
        self.assertEqual(
            config.read_bytes(), b"mutated during completion publication"
        )
        self.assertFalse(completion.exists())

    def test_prepare_surfaces_secondary_failure_if_completion_changed_before_rollback(
        self,
    ) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        completion = fixture.run_root / "prepare/completed.json"
        config = fixture.run_root / "config.json"
        replacement = b"concurrent completion replacement"
        real_publish = module._publish_bytes_create_absent

        def replace_completion_and_dependency(path: Path, content: bytes) -> None:
            real_publish(path, content)
            if Path(path) == completion:
                completion.write_bytes(replacement)
                config.write_bytes(b"concurrent dependency mutation")

        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", return_value={}
        ):
            module.run_preflight(fixture.inputs)
            with mock.patch.object(
                module,
                "_publish_bytes_create_absent",
                side_effect=replace_completion_and_dependency,
            ), self.assertRaisesRegex(
                ValueError,
                "dependency validation failed and completion rollback failed",
            ):
                module.run_prepare(fixture.inputs)
        self.assertEqual(completion.read_bytes(), replacement)
        self.assertEqual(config.read_bytes(), b"concurrent dependency mutation")

    def test_completion_rollback_rejects_missing_nonregular_and_changed_target(
        self,
    ) -> None:
        module = self.api()
        expected = b"newly published completion"
        for state in ("missing", "nonregular", "changed"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "completed.json"
                if state == "nonregular":
                    target.mkdir()
                elif state == "changed":
                    target.write_bytes(b"concurrent replacement")
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot be rolled back|changed before rollback",
                ):
                    module._rollback_exact_new_file(target, expected)

    def test_missing_prepare_marker_with_any_stage_artifact_is_preserved_and_refused(self) -> None:
        module = self.api()
        fixture = self.fixture()
        with fixture.patched_authentication():
            module.run_preflight(fixture.inputs)
            conflict = fixture.run_root / "config.json"
            conflict.write_bytes(b"preserve me")
            before = conflict.read_bytes()
            with mock.patch.object(
                module, "load_base_camera_head", side_effect=AssertionError("model loaded")
            ), mock.patch.object(
                module, "publish_translation_sample", side_effect=AssertionError("published")
            ), self.assertRaisesRegex(ValueError, "new run ID"):
                module.run_prepare(fixture.inputs)
        self.assertEqual(conflict.read_bytes(), before)
        self.assertFalse((fixture.run_root / "prepare/completed.json").exists())

    def test_completed_prepare_with_missing_marker_is_not_resumed(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(module, "load_bound_bundle_bytes", return_value={}):
            module.run_preflight(fixture.inputs)
            marker = module.run_prepare(fixture.inputs)
            marker.unlink()
            sentinel = fixture.run_root / "prediction_only/long_context" / f"{SCENES[0]}.npz"
            before = sentinel.read_bytes()
            with mock.patch.object(
                module, "publish_translation_sample", side_effect=AssertionError("published")
            ), self.assertRaisesRegex(ValueError, "new run ID"):
                module.run_prepare(fixture.inputs)
        self.assertEqual(sentinel.read_bytes(), before)

    def test_load_cohort_terminal_barrier_rejects_late_foreign_entry(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", return_value={}
        ):
            module.run_preflight(fixture.inputs)
            module.run_prepare(fixture.inputs)

            late_foreign = fixture.run_root / "late_foreign.bin"
            valid_loader = self._bundle_loader(fixture)

            def inject_late_foreign(*payloads: bytes):
                if not late_foreign.exists():
                    late_foreign.write_bytes(b"appeared during cohort decode")
                return valid_loader(*payloads)

            with mock.patch.object(
                module,
                "load_bound_bundle_bytes",
                side_effect=inject_late_foreign,
            ), self.assertRaisesRegex(ValueError, "foreign|inventory"):
                module.load_published_cohort(fixture.run_root)
        self.assertEqual(late_foreign.read_bytes(), b"appeared during cohort decode")

    def test_load_cohort_directory_barrier_rejects_add_remove_restore(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", return_value={}
        ):
            module.run_preflight(fixture.inputs)
            module.run_prepare(fixture.inputs)

            transient = fixture.run_root / "transient_foreign.bin"
            valid_loader = self._bundle_loader(fixture)

            def add_remove_restore(*payloads: bytes):
                if not transient.exists():
                    transient.write_bytes(b"transient directory mutation")
                    transient.unlink()
                return valid_loader(*payloads)

            with mock.patch.object(
                module,
                "load_bound_bundle_bytes",
                side_effect=add_remove_restore,
            ), self.assertRaisesRegex(ValueError, "directory|inventory|changed"):
                module.load_published_cohort(fixture.run_root)
        self.assertFalse(transient.exists())

    def test_load_cohort_binds_validated_bundle_provenance_to_outer_chain(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        valid_loader = self._bundle_loader(fixture)
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", side_effect=valid_loader
        ):
            module.run_preflight(fixture.inputs)
            module.run_prepare(fixture.inputs)

            attacks = {
                "sample_id": "wrong_sample:frames_500",
                "scene": "scene9999_99",
                "source_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "git_commit": "c" * 40,
                "formal_label_sha256": "d" * 64,
                "teacher_reference_sha256": "e" * 64,
            }
            for field, replacement in attacks.items():
                with self.subTest(field=field), mock.patch.object(
                    module,
                    "load_bound_bundle_bytes",
                    side_effect=self._bundle_loader(
                        fixture, {field: replacement}
                    ),
                ), self.assertRaisesRegex(ValueError, "bundle .* binding mismatch"):
                    module.load_published_cohort(fixture.run_root)

    def test_load_cohort_accepts_only_exact_explicit_downstream_allowlist(
        self,
    ) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        valid_loader = self._bundle_loader(fixture)
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(
            module, "load_bound_bundle_bytes", side_effect=valid_loader
        ):
            module.run_preflight(fixture.inputs)
            module.run_prepare(fixture.inputs)
            downstream_relative = "smoke/completed.json"
            downstream = fixture.run_root / downstream_relative
            downstream.parent.mkdir(parents=True)
            downstream.write_bytes(b"signed downstream marker")

            with self.assertRaisesRegex(ValueError, "foreign|inventory"):
                module.load_published_cohort(fixture.run_root)

            cohort = module.load_published_cohort(
                fixture.run_root,
                allowed_downstream_files=frozenset({downstream_relative}),
            )
            self.assertEqual([row.scene for row in cohort], list(SCENES))

            extra = fixture.run_root / "smoke/unlisted.json"
            extra.write_bytes(b"not allowlisted")
            with self.assertRaisesRegex(ValueError, "foreign|inventory"):
                module.load_published_cohort(
                    fixture.run_root,
                    allowed_downstream_files=frozenset({downstream_relative}),
                )

    def test_load_cohort_rejects_fully_resigned_physical_path_escape(self) -> None:
        module = self.api()
        fixture = self.fixture()
        publisher = self._publisher(fixture, [])
        with fixture.patched_authentication(), mock.patch.object(
            module,
            "load_base_camera_head",
            return_value=(torch.nn.Identity(), fixture.checkpoint_sha256),
        ), mock.patch.object(
            module, "publish_translation_sample", side_effect=publisher
        ), mock.patch.object(module, "load_bound_bundle_bytes", return_value={}):
            module.run_preflight(fixture.inputs)
            completion = module.run_prepare(fixture.inputs)
            cohort_path = fixture.run_root / "manifests/cohort.json"
            cohort = json.loads(cohort_path.read_text())
            cohort["records"][0]["long_path"] = "privileged_labels/quality/scene0000_00.npz"
            _write_json(cohort_path, cohort)
            config_path = fixture.run_root / "config.json"
            config = json.loads(config_path.read_text())
            config["cohort_manifest_sha256"] = _sha(cohort_path)
            _write_json(config_path, config)
            marker = json.loads(completion.read_text())
            marker["run_config_sha256"] = _sha(config_path)
            marker["files"]["config.json"] = _sha(config_path)
            marker["files"]["manifests/cohort.json"] = _sha(cohort_path)
            unsigned = dict(marker)
            unsigned.pop("completion_digest")
            marker["completion_digest"] = _canonical_digest(unsigned)
            _write_json(completion, marker)
            with self.assertRaisesRegex(ValueError, "physical|path"):
                module.load_published_cohort(fixture.run_root)


if __name__ == "__main__":
    unittest.main()
