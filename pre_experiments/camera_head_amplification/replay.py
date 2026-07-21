"""Hook-based Camera Head replay without changing the model forward API."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from vggt.heads.camera_head import CameraHead


@dataclass(frozen=True)
class CameraHeadReplay:
    """Transient outputs from one frozen Camera Head replay."""

    activated_pose: torch.Tensor
    raw_pose: torch.Tensor
    pose_delta: torch.Tensor
    representations: OrderedDict[str, torch.Tensor]


def replay_camera_head(
    camera_head: CameraHead,
    normalized_tokens: torch.Tensor,
    *,
    num_iterations: int = 4,
    capture_representations: bool = True,
) -> CameraHeadReplay:
    """Decode normalized tokens and capture each trunk stage with hooks."""
    if normalized_tokens.ndim != 3:
        raise ValueError("normalized_tokens must have shape [B, S, C]")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    captured: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
    handles = []
    if capture_representations:
        captured["adaln_input"] = []
        for index in range(len(camera_head.trunk)):
            captured[f"block_{index + 1}"] = []
        captured["trunk_norm"] = []

        def capture_input(_module, inputs):
            captured["adaln_input"].append(inputs[0].detach().clone())

        def capture_output(name):
            def hook(_module, _inputs, output):
                captured[name].append(output.detach().clone())

            return hook

        handles.append(camera_head.trunk[0].register_forward_pre_hook(capture_input))
        for index, block in enumerate(camera_head.trunk):
            handles.append(
                block.register_forward_hook(capture_output(f"block_{index + 1}"))
            )
        handles.append(
            camera_head.trunk_norm.register_forward_hook(
                capture_output("trunk_norm")
            )
        )

    try:
        with torch.no_grad():
            activated, trace = camera_head.decode_pose_tokens(
                normalized_tokens,
                num_iterations=num_iterations,
                return_trace=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    representations: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, values in captured.items():
        if len(values) != num_iterations:
            raise RuntimeError(
                f"expected {num_iterations} captures for {name}, found {len(values)}"
            )
        representations[name] = torch.stack(values)

    return CameraHeadReplay(
        activated_pose=torch.stack(activated),
        raw_pose=torch.stack(trace["raw_pose_enc_list"]),
        pose_delta=torch.stack(trace["pose_delta_list"]),
        representations=representations,
    )
