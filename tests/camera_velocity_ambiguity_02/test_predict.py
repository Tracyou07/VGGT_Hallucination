from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pre_experiments.camera_velocity_ambiguity_02.artifacts import (
    build_prediction_identity,
    load_completed_prediction,
)
from pre_experiments.camera_velocity_ambiguity_02.frames import FrameSelection
from pre_experiments.camera_velocity_ambiguity_02.predict import (
    PredictionContext,
    configure_camera_only,
    load_local_camera_model,
    run_scene_predictions,
    select_scene_shard,
)


GIT_COMMIT = "1" * 40
CHECKPOINT_SHA256 = "2" * 64
PROTOCOL_DIGEST = "3" * 64


class FakeModel:
    def __init__(self) -> None:
        self.camera_head = object()
        self.depth_head = object()
        self.point_head = object()
        self.track_head = object()
        self.calls: list[tuple[tuple[int, ...], dict[str, object]]] = []
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def to(self, device: torch.device) -> "FakeModel":
        self.to_calls.append(str(device))
        return self

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self

    def __call__(self, images: torch.Tensor, **kwargs: object) -> dict[str, object]:
        self.calls.append((tuple(images.shape), dict(kwargs)))
        sequence = images.shape[-4]
        pose_enc = torch.zeros((1, sequence, 9), dtype=torch.float32)
        tokens = torch.arange(sequence * 6, dtype=torch.float32).reshape(1, sequence, 6)
        return {
            "pose_enc_list": [pose_enc],
            "camera_trace": {"normalized_camera_tokens": tokens},
        }


def _selection() -> FrameSelection:
    ids = (2, 5, 9, 14, 20, 27)
    return FrameSelection(
        frame_ids=ids,
        image_paths=tuple(Path("color") / f"{value}.jpg" for value in ids),
        pose_indices=tuple(range(len(ids))),
    )


def _context() -> PredictionContext:
    return PredictionContext(
        run_id="cva02-smoke",
        checkpoint_sha256=CHECKPOINT_SHA256,
        git_commit=GIT_COMMIT,
        protocol_digest=PROTOCOL_DIGEST,
        preprocess="crop",
        camera_iterations=4,
    )


def _decoder(
    pose_enc: torch.Tensor,
    image_hw: tuple[int, int],
    *,
    build_intrinsics: bool,
) -> tuple[torch.Tensor, None]:
    del image_hw
    if build_intrinsics:
        raise AssertionError("camera prediction must not build unused intrinsics")
    sequence = pose_enc.shape[1]
    w2c = torch.zeros((1, sequence, 3, 4), dtype=torch.float32)
    w2c[..., :3] = torch.eye(3, dtype=torch.float32)
    w2c[0, :, 0, 3] = -torch.arange(sequence, dtype=torch.float32)
    return w2c, None


class PredictionRunnerTest(unittest.TestCase):
    def test_runs_one_global_then_each_local_window_with_frozen_camera_settings(self) -> None:
        model = FakeModel()
        loader_calls: list[tuple[list[str], str]] = []

        def image_loader(paths: list[str], mode: str) -> torch.Tensor:
            loader_calls.append((paths, mode))
            return torch.zeros((len(paths), 3, 8, 10), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = run_scene_predictions(
                model=model,
                scene="scene0000_00",
                selection=_selection(),
                output_dir=output,
                context=_context(),
                device=torch.device("cpu"),
                window_length=4,
                window_stride=2,
                image_loader=image_loader,
                pose_decoder=_decoder,
            )

            self.assertEqual(summary, {"global_ran": 1, "local_ran": 2, "resumed": 0})
            self.assertEqual([shape for shape, _ in model.calls], [(6, 3, 8, 10), (4, 3, 8, 10), (4, 3, 8, 10)])
            for _, kwargs in model.calls:
                self.assertEqual(
                    kwargs,
                    {"camera_num_iterations": 4, "return_camera_trace": True},
                )
            self.assertEqual([mode for _, mode in loader_calls], ["crop", "crop", "crop"])
            self.assertEqual(
                loader_calls[1][0],
                [str(path) for path in _selection().image_paths[:4]],
            )
            self.assertEqual(
                loader_calls[2][0],
                [str(path) for path in _selection().image_paths[2:]],
            )

            global_ids = np.asarray(_selection().frame_ids, dtype=np.int64)
            global_identity = build_prediction_identity(
                run_id=_context().run_id,
                scene="scene0000_00",
                artifact_kind="global",
                window_index=None,
                frame_ids=global_ids,
                checkpoint_sha256=CHECKPOINT_SHA256,
                git_commit=GIT_COMMIT,
                protocol_digest=PROTOCOL_DIGEST,
                preprocess="crop",
                camera_iterations=4,
            )
            global_artifact = load_completed_prediction(
                output / "global" / "prediction.npz",
                output / "global" / "complete.json",
                global_identity,
            )
            np.testing.assert_array_equal(global_artifact["frame_ids"], global_ids)
            np.testing.assert_array_equal(
                global_artifact["normalized_camera_tokens"],
                np.arange(36, dtype=np.float32).reshape(6, 6),
            )
            np.testing.assert_allclose(
                global_artifact["pred_c2w_raw"][:, 0, 3], np.arange(6)
            )

    def test_exact_completion_resumes_without_model_or_image_calls(self) -> None:
        selection = _selection()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = FakeModel()
            common = dict(
                scene="scene0000_00",
                selection=selection,
                output_dir=output,
                context=_context(),
                device=torch.device("cpu"),
                window_length=4,
                window_stride=2,
                image_loader=lambda paths, mode: torch.zeros((len(paths), 3, 8, 8)),
                pose_decoder=_decoder,
            )
            run_scene_predictions(model=first, **common)
            second = FakeModel()
            summary = run_scene_predictions(model=second, **common)

            self.assertEqual(summary, {"global_ran": 0, "local_ran": 0, "resumed": 3})
            self.assertEqual(second.calls, [])

    def test_tmp_marker_is_not_complete_and_only_that_unit_is_rerun(self) -> None:
        selection = _selection()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            common = dict(
                scene="scene0000_00",
                selection=selection,
                output_dir=output,
                context=_context(),
                device=torch.device("cpu"),
                window_length=4,
                window_stride=2,
                image_loader=lambda paths, mode: torch.zeros((len(paths), 3, 8, 8)),
                pose_decoder=_decoder,
            )
            run_scene_predictions(model=FakeModel(), **common)
            completion = output / "local" / "window_001" / "complete.json"
            completion.replace(completion.with_suffix(".json.tmp"))

            model = FakeModel()
            summary = run_scene_predictions(model=model, **common)
            self.assertEqual(summary, {"global_ran": 0, "local_ran": 1, "resumed": 2})
            self.assertEqual([shape for shape, _ in model.calls], [(4, 3, 8, 8)])

    def test_rejects_wrong_loader_shape_or_prediction_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            common = dict(
                model=FakeModel(),
                scene="scene0000_00",
                selection=_selection(),
                output_dir=Path(temporary),
                context=_context(),
                device=torch.device("cpu"),
                window_length=4,
                window_stride=2,
                pose_decoder=_decoder,
            )
            with self.assertRaisesRegex(ValueError, "image loader"):
                run_scene_predictions(
                    image_loader=lambda paths, mode: torch.zeros((1, 3, 8, 8)),
                    **common,
                )

    def test_scene_sharding_is_order_preserving_and_complete(self) -> None:
        scenes = tuple(f"scene{index:04d}_00" for index in range(7))
        shards = [select_scene_shard(scenes, shard_index=index, shard_count=3) for index in range(3)]
        self.assertEqual(shards[0], scenes[0::3])
        self.assertEqual(shards[1], scenes[1::3])
        self.assertEqual(shards[2], scenes[2::3])
        self.assertEqual(set().union(*map(set, shards)), set(scenes))
        with self.assertRaises(ValueError):
            select_scene_shard(scenes, shard_index=3, shard_count=3)

    def test_camera_only_local_loader_has_no_network_fallback(self) -> None:
        model = SimpleNamespace(
            camera_head=object(), depth_head=object(), point_head=object(), track_head=object()
        )
        self.assertIs(configure_camera_only(model), model)
        self.assertIsNone(model.depth_head)
        self.assertIsNone(model.point_head)
        self.assertIsNone(model.track_head)

        with mock.patch(
            "pre_experiments.camera_velocity_ambiguity_02.predict.load_local_model",
            side_effect=FileNotFoundError("missing local checkpoint"),
        ) as local_loader:
            with self.assertRaisesRegex(FileNotFoundError, "local checkpoint"):
                load_local_camera_model(Path("missing"), torch.device("cpu"))
        local_loader.assert_called_once_with(Path("missing"))


if __name__ == "__main__":
    unittest.main()
