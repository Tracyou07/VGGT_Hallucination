import unittest

import torch
from torch import nn

from vggt.heads.camera_head import CameraHead
from vggt.layers.mlp import Mlp
from vggt.models.vggt import VGGT


class _Aggregator(nn.Module):
    def forward(self, images):
        return [torch.zeros(images.shape[0], images.shape[1], 1, 8)], 1


class _Camera(nn.Module):
    def forward(self, tokens, *, num_iterations, return_trace, trace_pose_tokens):
        del tokens
        pose = torch.ones(1, 3, 9)
        trace = {"sentinel": (num_iterations, trace_pose_tokens)}
        return (list(pose for _ in range(num_iterations)), trace) if return_trace else [pose]


class CameraTrainingTraceTest(unittest.TestCase):
    def test_mlp_exposes_the_hidden_before_final_projection(self) -> None:
        torch.manual_seed(11)
        mlp = Mlp(8, hidden_features=4, out_features=3).eval()
        values = torch.randn(2, 8)

        with torch.no_grad():
            hidden = mlp.forward_features(values)
            actual = mlp.forward_head(hidden)
            expected = mlp(values)

        self.assertEqual(tuple(hidden.shape), (2, 4))
        torch.testing.assert_close(actual, expected)

    def test_trace_is_observational_and_contains_training_tensors(self) -> None:
        torch.manual_seed(7)
        head = CameraHead(
            dim_in=32,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        tokens = [torch.randn(1, 5, 3, 32)]

        with torch.no_grad():
            baseline = head(tokens, num_iterations=2)
            traced, trace = head(
                tokens,
                num_iterations=2,
                return_trace=True,
                trace_pose_tokens=True,
            )

        for expected, actual in zip(baseline, traced):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(trace["normalized_camera_tokens"].shape), (1, 5, 32))
        self.assertEqual(tuple(trace["pose_branch_hidden_list"][-1].shape), (1, 5, 16))
        self.assertEqual(tuple(trace["delta_norm"].shape), (2, 1, 5))

    def test_vggt_routes_trace_options_without_enabling_other_heads(self) -> None:
        model = VGGT.__new__(VGGT)
        nn.Module.__init__(model)
        model.aggregator = _Aggregator()
        model.camera_head = _Camera()
        model.depth_head = None
        model.point_head = None
        model.track_head = None
        model.eval()

        output = model(
            torch.zeros(3, 3, 2, 2),
            camera_num_iterations=2,
            return_camera_trace=True,
            camera_trace_pose_tokens=True,
        )

        self.assertEqual(output["camera_trace"]["sentinel"], (2, True))
        self.assertEqual(tuple(output["pose_enc"].shape), (1, 3, 9))


if __name__ == "__main__":
    unittest.main()
