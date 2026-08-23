"""Float64 NumPy/Torch geometry for SO(3) and SE(3).

Twists always use translation first: ``[..., v_x, v_y, v_z, omega_x,
omega_y, omega_z]``.  Homogeneous matrices use the w2c-compatible convention
``[..., 4, 4]``; this module does not expose any raw pose encoding.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch


_SMALL_ANGLE = 1e-4
_NEAR_PI = 1e-5
_HOMOGENEOUS_ATOL = 1e-10
_ROTATION_ATOL = 1e-8


def _is_torch(value) -> bool:
    return isinstance(value, torch.Tensor)


def _require_array(value, name: str):
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be a NumPy array or torch tensor")
    if value.dtype != (torch.float64 if _is_torch(value) else np.float64):
        raise TypeError(f"{name} must use float64 geometry")
    finite = torch.isfinite(value).all().item() if _is_torch(value) else np.isfinite(value).all()
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_last_shape(value, suffix: tuple[int, ...], name: str):
    _require_array(value, name)
    if value.ndim < len(suffix) or tuple(value.shape[-len(suffix) :]) != suffix:
        raise ValueError(f"{name} must have shape [..., {', '.join(map(str, suffix))}]")
    return value


def _eye(size: int, like, batch_shape=()):
    if _is_torch(like):
        result = torch.eye(size, dtype=torch.float64, device=like.device)
        return result.expand(*batch_shape, size, size)
    return np.broadcast_to(np.eye(size, dtype=np.float64), (*batch_shape, size, size))


def _stack(values, axis: int, like):
    return torch.stack(values, dim=axis) if _is_torch(like) else np.stack(values, axis=axis)


def _hat(vector):
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = x * 0.0
    return _stack(
        (
            _stack((zero, -z, y), -1, vector),
            _stack((z, zero, -x), -1, vector),
            _stack((-y, x, zero), -1, vector),
        ),
        -2,
        vector,
    )


def _select(mask, small_value, regular_value, like):
    if _is_torch(like):
        return torch.where(mask, small_value, regular_value)
    return np.where(mask, small_value, regular_value)


def _sqrt(value, like):
    return torch.sqrt(value) if _is_torch(like) else np.sqrt(value)


def _sin(value, like):
    return torch.sin(value) if _is_torch(like) else np.sin(value)


def _cos(value, like):
    return torch.cos(value) if _is_torch(like) else np.cos(value)


def _atan2(y, x, like):
    return torch.atan2(y, x) if _is_torch(like) else np.arctan2(y, x)


def _clamp(value, minimum=None, maximum=None, like=None):
    if _is_torch(like):
        return torch.clamp(value, min=minimum, max=maximum)
    return np.clip(value, minimum, maximum)


def _norm(value, axis: int, like):
    return torch.linalg.vector_norm(value, dim=axis) if _is_torch(like) else np.linalg.norm(value, axis=axis)


def _transpose_matrices(value):
    return value.transpose(-1, -2) if _is_torch(value) else np.swapaxes(value, -1, -2)


def _validate_rotation(rotation, name: str = "rotation"):
    _require_last_shape(rotation, (3, 3), name)
    identity = _eye(3, rotation, rotation.shape[:-2])
    gram = _transpose_matrices(rotation) @ rotation
    if _is_torch(rotation):
        orthonormal = torch.allclose(gram, identity, atol=_ROTATION_ATOL, rtol=_ROTATION_ATOL)
        determinant = torch.linalg.det(rotation)
        positive = bool((determinant > 0.0).all().item())
        unit = torch.allclose(determinant, torch.ones_like(determinant), atol=_ROTATION_ATOL, rtol=_ROTATION_ATOL)
    else:
        orthonormal = np.allclose(gram, identity, atol=_ROTATION_ATOL, rtol=_ROTATION_ATOL)
        determinant = np.linalg.det(rotation)
        positive = bool(np.all(determinant > 0.0))
        unit = np.allclose(determinant, 1.0, atol=_ROTATION_ATOL, rtol=_ROTATION_ATOL)
    if not orthonormal:
        raise ValueError(f"{name} rotation must be orthonormal")
    if not positive:
        raise ValueError(f"{name} rotation determinant must be positive")
    if not unit:
        raise ValueError(f"{name} rotation determinant must equal one")
    return rotation


def _validate_transform(transform, name: str = "transform"):
    _require_last_shape(transform, (4, 4), name)
    expected = transform[..., 3, :] * 0.0
    if _is_torch(transform):
        expected = expected.clone()
        expected[..., 3] = 1.0
        valid_row = torch.allclose(
            transform[..., 3, :], expected, atol=_HOMOGENEOUS_ATOL, rtol=0.0
        )
    else:
        expected = np.array(expected, copy=True)
        expected[..., 3] = 1.0
        valid_row = np.allclose(
            transform[..., 3, :], expected, atol=_HOMOGENEOUS_ATOL, rtol=0.0
        )
    if not valid_row:
        raise ValueError(f"{name} must have final homogeneous row [0, 0, 0, 1]")
    _validate_rotation(transform[..., :3, :3], f"{name}")
    return transform


def _so3_coefficients(omega):
    theta2 = (omega * omega).sum(axis=-1) if not _is_torch(omega) else (omega * omega).sum(dim=-1)
    threshold2 = _SMALL_ANGLE**2
    safe_theta = _sqrt(_clamp(theta2, minimum=threshold2, like=omega), omega)
    regular_a = _sin(safe_theta, omega) / safe_theta
    half_theta = 0.5 * safe_theta
    regular_b = 0.5 * (_sin(half_theta, omega) / half_theta) ** 2
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    small_a = 1.0 - theta2 / 6.0 + theta4 / 120.0 - theta6 / 5040.0
    small_b = 0.5 - theta2 / 24.0 + theta4 / 720.0 - theta6 / 40320.0
    small = theta2 < threshold2
    return (
        _select(small, small_a, regular_a, omega),
        _select(small, small_b, regular_b, omega),
        theta2,
    )


def so3_exp(omega):
    """Exponentiate float64 axis-angle vectors of shape ``[..., 3]``."""
    _require_last_shape(omega, (3,), "omega")
    a, b, _ = _so3_coefficients(omega)
    skew = _hat(omega)
    identity = _eye(3, omega, omega.shape[:-1])
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def _near_pi_axis(rotation, cosine):
    identity = _eye(3, rotation, rotation.shape[:-2])
    symmetric = 0.5 * (rotation + _transpose_matrices(rotation))
    denominator = _clamp(1.0 - cosine, minimum=1e-15, like=rotation)
    outer = (symmetric - cosine[..., None, None] * identity) / denominator[..., None, None]
    flat_outer = outer.reshape(-1, 3, 3)
    flat_rotation = rotation.reshape(-1, 3, 3)
    if flat_outer.shape[0] == 0:
        return rotation[..., 0, :]
    axes = []
    for matrix, source_rotation in zip(flat_outer, flat_rotation):
        diagonal = matrix.diagonal()
        index = int((torch.argmax(diagonal).item() if _is_torch(rotation) else np.argmax(diagonal)))
        component = _sqrt(_clamp(diagonal[index], minimum=1e-30, like=rotation), rotation)
        values = [matrix[j, index] / component for j in range(3)]
        values[index] = component
        axis = _stack(values, 0, rotation)
        axis = axis / _norm(axis, -1, rotation)
        skew_vector = _stack(
            (
                0.5 * (source_rotation[2, 1] - source_rotation[1, 2]),
                0.5 * (source_rotation[0, 2] - source_rotation[2, 0]),
                0.5 * (source_rotation[1, 0] - source_rotation[0, 1]),
            ),
            0,
            rotation,
        )
        dot = (axis * skew_vector).sum()
        sign = _select(dot < 0.0, dot * 0.0 - 1.0, dot * 0.0 + 1.0, rotation)
        axes.append(axis * sign)
    return _stack(axes, 0, rotation).reshape(*rotation.shape[:-2], 3)


def so3_log(rotation):
    """Return float64 principal axis-angle vectors for valid rotations."""
    _validate_rotation(rotation)
    skew_vector = _stack(
        (
            0.5 * (rotation[..., 2, 1] - rotation[..., 1, 2]),
            0.5 * (rotation[..., 0, 2] - rotation[..., 2, 0]),
            0.5 * (rotation[..., 1, 0] - rotation[..., 0, 1]),
        ),
        -1,
        rotation,
    )
    sine2 = (
        (skew_vector * skew_vector).sum(dim=-1)
        if _is_torch(rotation)
        else (skew_vector * skew_vector).sum(axis=-1)
    )
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cosine = _clamp(0.5 * (trace - 1.0), minimum=-1.0, maximum=1.0, like=rotation)
    threshold2 = _SMALL_ANGLE**2
    safe_sine = _sqrt(_clamp(sine2, minimum=threshold2, like=rotation), rotation)
    regular_theta = _atan2(safe_sine, cosine, rotation)
    regular = skew_vector * (regular_theta / safe_sine)[..., None]
    sine4 = sine2 * sine2
    sine6 = sine4 * sine2
    small_factor = 1.0 + sine2 / 6.0 + 3.0 * sine4 / 40.0 + 5.0 * sine6 / 112.0
    small = skew_vector * small_factor[..., None]
    small_mask = (sine2 < threshold2) & (cosine > 0.0)
    regular_or_small = _select(small_mask[..., None], small, regular, rotation)

    near_sine = _sqrt(_clamp(sine2, minimum=1e-30, like=rotation), rotation)
    near_theta = _atan2(near_sine, cosine, rotation)
    near_pi = _near_pi_axis(rotation, cosine) * near_theta[..., None]
    near_pi_mask = cosine < -math.cos(_NEAR_PI)
    return _select(near_pi_mask[..., None], near_pi, regular_or_small, rotation)


def se3_exp(twist):
    """Exponentiate translation-first twists ``[..., v, omega]`` to ``[...,4,4]``."""
    _require_last_shape(twist, (6,), "twist")
    velocity = twist[..., :3]
    omega = twist[..., 3:]
    _, b, theta2 = _so3_coefficients(omega)
    threshold2 = _SMALL_ANGLE**2
    safe_theta = _sqrt(_clamp(theta2, minimum=threshold2, like=twist), twist)
    regular_c = (safe_theta - _sin(safe_theta, twist)) / (safe_theta**3)
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    small_c = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0 - theta6 / 362880.0
    c = _select(theta2 < threshold2, small_c, regular_c, twist)
    skew = _hat(omega)
    identity3 = _eye(3, twist, twist.shape[:-1])
    jacobian = identity3 + b[..., None, None] * skew + c[..., None, None] * (skew @ skew)
    translation = (jacobian @ velocity[..., None])[..., 0]
    rotation = so3_exp(omega)
    if _is_torch(twist):
        top = torch.cat((rotation, translation[..., None]), dim=-1)
        bottom = torch.zeros((*twist.shape[:-1], 1, 4), dtype=torch.float64, device=twist.device)
        bottom[..., 0, 3] = 1.0
        return torch.cat((top, bottom), dim=-2)
    top = np.concatenate((rotation, translation[..., None]), axis=-1)
    bottom = np.zeros((*twist.shape[:-1], 1, 4), dtype=np.float64)
    bottom[..., 0, 3] = 1.0
    return np.concatenate((top, bottom), axis=-2)


def se3_log(transform):
    """Log valid transforms to translation-first twists ``[..., v, omega]``."""
    _validate_transform(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = so3_log(rotation)
    theta2 = (omega * omega).sum(dim=-1) if _is_torch(omega) else (omega * omega).sum(axis=-1)
    threshold2 = _SMALL_ANGLE**2
    safe_theta = _sqrt(_clamp(theta2, minimum=threshold2, like=omega), omega)
    if _is_torch(omega):
        regular_d = (1.0 - 0.5 * safe_theta / torch.tan(0.5 * safe_theta)) / (safe_theta**2)
    else:
        regular_d = (1.0 - 0.5 * safe_theta / np.tan(0.5 * safe_theta)) / (safe_theta**2)
    small_d = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    d = _select(theta2 < threshold2, small_d, regular_d, omega)
    skew = _hat(omega)
    identity = _eye(3, omega, omega.shape[:-1])
    jacobian_inverse = identity - 0.5 * skew + d[..., None, None] * (skew @ skew)
    velocity = (jacobian_inverse @ translation[..., None])[..., 0]
    if _is_torch(transform):
        return torch.cat((velocity, omega), dim=-1)
    return np.concatenate((velocity, omega), axis=-1)


def inverse(transform):
    """Invert one or a batch of valid homogeneous transforms."""
    _validate_transform(transform)
    rotation_t = _transpose_matrices(transform[..., :3, :3])
    translation = -(rotation_t @ transform[..., :3, 3, None])[..., 0]
    if _is_torch(transform):
        top = torch.cat((rotation_t, translation[..., None]), dim=-1)
        bottom = torch.zeros((*transform.shape[:-2], 1, 4), dtype=torch.float64, device=transform.device)
        bottom[..., 0, 3] = 1.0
        return torch.cat((top, bottom), dim=-2)
    top = np.concatenate((rotation_t, translation[..., None]), axis=-1)
    bottom = np.zeros((*transform.shape[:-2], 1, 4), dtype=np.float64)
    bottom[..., 0, 3] = 1.0
    return np.concatenate((top, bottom), axis=-2)


def compose(first, second):
    """Compose valid transforms as ``first @ second`` with batch broadcasting."""
    _validate_transform(first, "first transform")
    _validate_transform(second, "second transform")
    if _is_torch(first) != _is_torch(second):
        raise TypeError("transforms must use the same NumPy or torch backend")
    if _is_torch(first) and first.device != second.device:
        raise ValueError("torch transforms must use the same device")
    try:
        result = first @ second
    except (RuntimeError, ValueError) as error:
        raise ValueError("transform batch shapes are not broadcast-compatible") from error
    return result


def _broadcast_transform_pair(first, second):
    if _is_torch(first) != _is_torch(second):
        raise TypeError("transforms must use the same NumPy or torch backend")
    if _is_torch(first) and first.device != second.device:
        raise ValueError("torch transforms must use the same device")
    try:
        if _is_torch(first):
            batch_shape = torch.broadcast_shapes(first.shape[:-2], second.shape[:-2])
            return (
                first.expand(*batch_shape, 4, 4),
                second.expand(*batch_shape, 4, 4),
            )
        batch_shape = np.broadcast_shapes(first.shape[:-2], second.shape[:-2])
        return (
            np.broadcast_to(first, (*batch_shape, 4, 4)),
            np.broadcast_to(second, (*batch_shape, 4, 4)),
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError("transform batch shapes are not broadcast-compatible") from error


def _parameter_value(parameter, like):
    if isinstance(parameter, bool):
        raise TypeError("t must be a real scalar")
    if isinstance(parameter, Real):
        value = float(parameter)
    elif isinstance(parameter, np.ndarray):
        if _is_torch(like):
            raise TypeError("t and transforms must use the same NumPy or torch backend")
        if parameter.shape != () or parameter.dtype != np.float64:
            raise TypeError("t must be a float64 scalar")
        value = parameter
    elif isinstance(parameter, torch.Tensor):
        if not _is_torch(like):
            raise TypeError("t and transforms must use the same NumPy or torch backend")
        if parameter.shape != () or parameter.dtype != torch.float64:
            raise TypeError("t must be a float64 scalar")
        if parameter.device != like.device:
            raise ValueError("torch t must use the same device as transforms")
        value = parameter
    else:
        raise TypeError("t must be a real scalar")
    if _is_torch(value):
        if not torch.isfinite(value).item():
            raise ValueError("t must be finite")
        if ((value < 0.0) | (value > 1.0)).item():
            raise ValueError("t must lie in [0, 1]")
    else:
        if not np.isfinite(value):
            raise ValueError("t must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError("t must lie in [0, 1]")
    return value


def _is_exact_parameter(value, expected: float) -> bool:
    if _is_torch(value):
        return bool((value == expected).item())
    return bool(value == expected)


def _exact_endpoint(transform):
    return transform.clone() if _is_torch(transform) else transform.copy()


def _straight_through_endpoint(transform, interpolated):
    return transform.detach().clone() + interpolated - interpolated.detach()


def geodesic_interpolate(start, end, t):
    """Interpolate as ``Exp(t Log(end @ inv(start))) @ start``."""
    _validate_transform(start, "start transform")
    _validate_transform(end, "end transform")
    start, end = _broadcast_transform_pair(start, end)
    value = _parameter_value(t, start)
    at_start = _is_exact_parameter(value, 0.0)
    at_end = _is_exact_parameter(value, 1.0)
    trainable_parameter = _is_torch(value) and value.requires_grad
    if at_start and not trainable_parameter:
        return _exact_endpoint(start)
    if at_end and not trainable_parameter:
        return _exact_endpoint(end)
    difference = compose(end, inverse(start))
    interpolated = compose(se3_exp(se3_log(difference) * value), start)
    if at_start:
        return _straight_through_endpoint(start, interpolated)
    if at_end:
        return _straight_through_endpoint(end, interpolated)
    return interpolated


__all__ = [
    "so3_exp",
    "so3_log",
    "se3_exp",
    "se3_log",
    "inverse",
    "compose",
    "geodesic_interpolate",
]
