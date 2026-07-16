import unittest

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT


class FakeAggregator(nn.Module):
    def forward(self, images):
        batch, sequence = images.shape[:2]
        tokens = torch.zeros(batch, sequence, 2, 8)
        return [tokens], 1


class FakeCameraHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        aggregated_tokens_list,
        num_iterations=4,
        return_trace=False,
        trace_pose_tokens=False,
    ):
        self.calls.append((num_iterations, return_trace, trace_pose_tokens))
        batch, sequence = aggregated_tokens_list[-1].shape[:2]
        poses = [torch.full((batch, sequence, 9), float(index + 1)) for index in range(num_iterations)]
        if not return_trace:
            return poses
        trace = {
            "normalized_camera_tokens": torch.zeros(batch, sequence, 8),
            "raw_pose_enc_list": poses,
            "pose_delta_list": poses,
            "delta_norm": torch.zeros(num_iterations, batch, sequence),
            "pose_tokens_modulated_list": poses if trace_pose_tokens else [],
        }
        return poses, trace


def make_model():
    model = VGGT.__new__(VGGT)
    nn.Module.__init__(model)
    model.aggregator = FakeAggregator()
    model.camera_head = FakeCameraHead()
    model.depth_head = None
    model.point_head = None
    model.track_head = None
    return model.eval()


class VGGTCameraOptionsTest(unittest.TestCase):
    def test_default_camera_contract_is_preserved(self):
        model = make_model()
        images = torch.zeros(1, 3, 3, 14, 14)

        with torch.no_grad():
            predictions = model(images)

        self.assertEqual(model.camera_head.calls, [(4, False, False)])
        self.assertEqual(len(predictions["pose_enc_list"]), 4)
        torch.testing.assert_close(predictions["pose_enc"], predictions["pose_enc_list"][-1])
        self.assertNotIn("camera_trace", predictions)

    def test_camera_options_are_forwarded_and_trace_is_conditional(self):
        model = make_model()
        images = torch.zeros(1, 3, 3, 14, 14)

        with torch.no_grad():
            predictions = model(
                images,
                camera_num_iterations=8,
                return_camera_trace=True,
                camera_trace_pose_tokens=True,
            )

        self.assertEqual(model.camera_head.calls, [(8, True, True)])
        self.assertEqual(len(predictions["pose_enc_list"]), 8)
        self.assertIn("camera_trace", predictions)
        self.assertEqual(len(predictions["camera_trace"]["pose_tokens_modulated_list"]), 8)


if __name__ == "__main__":
    unittest.main()
