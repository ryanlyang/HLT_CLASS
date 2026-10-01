"""CMS-calibrated proxy-HLT persistent-spine benchmark."""

from .campaign import (
    BRANCHES, DIRECT_COARSE_BRANCHES, build_direct_coarse_plan,
    build_scientific_plan,
)
from .contracts import artifact, validate

__all__ = [
    "BRANCHES", "DIRECT_COARSE_BRANCHES", "artifact",
    "build_direct_coarse_plan", "build_scientific_plan", "validate",
]
