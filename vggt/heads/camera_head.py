# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose


class CameraTrace(TypedDict):
    normalized_camera_tokens: torch.Tensor
    raw_pose_enc_list: list[torch.Tensor]
    pose_delta_list: list[torch.Tensor]
    delta_norm: torch.Tensor
    trunk_output_list: list[torch.Tensor]
    pose_branch_hidden_list: list[torch.Tensor]


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        num_iterations: int = 4,
        return_trace: bool = False,
        trace_pose_tokens: bool = False,
        hidden_ablation_mask: torch.Tensor | None = None,
        hidden_additive_perturbation: torch.Tensor | None = None,
        pose_delta_additive_perturbation: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], CameraTrace]:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            return_trace (bool, optional): Return per-iteration raw pose updates and
                summary statistics. Defaults to False.
            trace_pose_tokens (bool, optional): Include high-dimensional modulated
                pose tokens in the trace. Requires return_trace=True.
            hidden_additive_perturbation (torch.Tensor, optional): Batch-specific
                post-GELU hidden additions with shape [iteration, batch, hidden].
            pose_delta_additive_perturbation (torch.Tensor, optional):
                Batch-specific pose-delta additions with shape
                [iteration, batch, 9].

        Returns:
            list or tuple: Predicted camera encodings from each iteration, optionally
                paired with a CameraTrace.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        normalized_pose_tokens = self.token_norm(tokens[:, :, 0])

        return self.decode_pose_tokens(
            normalized_pose_tokens,
            num_iterations=num_iterations,
            return_trace=return_trace,
            trace_pose_tokens=trace_pose_tokens,
            hidden_ablation_mask=hidden_ablation_mask,
            hidden_additive_perturbation=hidden_additive_perturbation,
            pose_delta_additive_perturbation=pose_delta_additive_perturbation,
        )

    def decode_pose_tokens(
        self,
        normalized_pose_tokens: torch.Tensor,
        num_iterations: int = 4,
        return_trace: bool = False,
        trace_pose_tokens: bool = False,
        hidden_ablation_mask: torch.Tensor | None = None,
        hidden_additive_perturbation: torch.Tensor | None = None,
        pose_delta_additive_perturbation: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], CameraTrace]:
        """Decode normalized camera tokens with iterative pose refinement."""
        if normalized_pose_tokens.ndim != 3:
            raise ValueError("normalized_pose_tokens must have shape [B, S, C]")
        if num_iterations < 1:
            raise ValueError("num_iterations must be at least 1")
        if trace_pose_tokens and not return_trace:
            raise ValueError("trace_pose_tokens requires return_trace=True")
        hidden_dim = self.pose_branch.fc2.in_features
        if hidden_ablation_mask is not None:
            expected = (num_iterations, hidden_dim)
            if (
                hidden_ablation_mask.dtype != torch.bool
                or tuple(hidden_ablation_mask.shape) != expected
            ):
                raise ValueError(
                    f"hidden_ablation_mask must be boolean with shape {expected}"
                )
        batch_size = normalized_pose_tokens.shape[0]
        _validate_additive_perturbation(
            "hidden_additive_perturbation",
            hidden_additive_perturbation,
            (num_iterations, batch_size, hidden_dim),
        )
        _validate_additive_perturbation(
            "pose_delta_additive_perturbation",
            pose_delta_additive_perturbation,
            (num_iterations, batch_size, self.target_dim),
        )

        return self.trunk_fn(
            normalized_pose_tokens,
            num_iterations=num_iterations,
            return_trace=return_trace,
            trace_pose_tokens=trace_pose_tokens,
            hidden_ablation_mask=hidden_ablation_mask,
            hidden_additive_perturbation=hidden_additive_perturbation,
            pose_delta_additive_perturbation=pose_delta_additive_perturbation,
        )

    def trunk_fn(
        self,
        pose_tokens: torch.Tensor,
        num_iterations: int,
        return_trace: bool = False,
        trace_pose_tokens: bool = False,
        hidden_ablation_mask: torch.Tensor | None = None,
        hidden_additive_perturbation: torch.Tensor | None = None,
        pose_delta_additive_perturbation: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], CameraTrace]:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.
            return_trace (bool): Return raw per-iteration state when True.
            trace_pose_tokens (bool): Retain modulated pose tokens when True.
            hidden_additive_perturbation (torch.Tensor, optional): Hidden
                additions with shape [iteration, batch, hidden].
            pose_delta_additive_perturbation (torch.Tensor, optional):
                Pose-delta additions with shape [iteration, batch, 9].

        Returns:
            list or tuple: Activated camera encodings, optionally paired with a trace.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        raw_pose_enc_list = [] if return_trace else None
        pose_delta_list = [] if return_trace else None
        delta_norm_list = [] if return_trace else None
        trunk_output_list = [] if trace_pose_tokens else None
        pose_branch_hidden_list = [] if trace_pose_tokens else None

        for iteration in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            trunk_output = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pose_branch_hidden = self.pose_branch.forward_features(
                self.trunk_norm(trunk_output)
            )
            if hidden_ablation_mask is not None:
                mask = hidden_ablation_mask[iteration].to(
                    device=pose_branch_hidden.device
                )
                pose_branch_hidden = pose_branch_hidden.masked_fill(mask, 0)
            if hidden_additive_perturbation is not None:
                hidden_addition = hidden_additive_perturbation[iteration].to(
                    device=pose_branch_hidden.device,
                    dtype=pose_branch_hidden.dtype,
                )
                pose_branch_hidden = (
                    pose_branch_hidden + hidden_addition[:, None, :]
                )
            pred_pose_enc_delta = self.pose_branch.forward_head(pose_branch_hidden)
            if pose_delta_additive_perturbation is not None:
                delta_addition = pose_delta_additive_perturbation[iteration].to(
                    device=pred_pose_enc_delta.device,
                    dtype=pred_pose_enc_delta.dtype,
                )
                pred_pose_enc_delta = (
                    pred_pose_enc_delta + delta_addition[:, None, :]
                )

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            if return_trace:
                assert raw_pose_enc_list is not None
                assert pose_delta_list is not None
                assert delta_norm_list is not None
                raw_pose_enc_list.append(pred_pose_enc)
                pose_delta_list.append(pred_pose_enc_delta)
                delta_norm_list.append(pred_pose_enc_delta.float().norm(dim=-1))
                if trace_pose_tokens:
                    assert trunk_output_list is not None
                    assert pose_branch_hidden_list is not None
                    trunk_output_list.append(trunk_output)
                    pose_branch_hidden_list.append(pose_branch_hidden)

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        if not return_trace:
            return pred_pose_enc_list

        assert raw_pose_enc_list is not None
        assert pose_delta_list is not None
        assert delta_norm_list is not None
        trace: CameraTrace = {
            "normalized_camera_tokens": pose_tokens,
            "raw_pose_enc_list": raw_pose_enc_list,
            "pose_delta_list": pose_delta_list,
            "delta_norm": torch.stack(delta_norm_list, dim=0),
            "trunk_output_list": trunk_output_list or [],
            "pose_branch_hidden_list": pose_branch_hidden_list or [],
        }
        return pred_pose_enc_list, trace


def _validate_additive_perturbation(
    name: str,
    perturbation: torch.Tensor | None,
    expected_shape: tuple[int, int, int],
) -> None:
    if perturbation is None:
        return
    if (
        not perturbation.is_floating_point()
        or tuple(perturbation.shape) != expected_shape
        or not bool(torch.isfinite(perturbation).all())
    ):
        raise ValueError(
            f"{name} must be finite floating-point data with shape "
            f"{expected_shape}"
        )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
