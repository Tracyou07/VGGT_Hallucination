"""Camera-translation H-VRFM Stage A-prime contracts."""

from .geometry import (
    apply_translation_endpoint,
    baseline_fill_teacher_centers,
    build_translation_endpoint,
    prediction_scale,
)

__all__ = [
    "apply_translation_endpoint",
    "baseline_fill_teacher_centers",
    "build_translation_endpoint",
    "prediction_scale",
]
