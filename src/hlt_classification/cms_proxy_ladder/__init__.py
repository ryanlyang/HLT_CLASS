"""CMS-calibrated proxy-HLT persistent-spine benchmark."""

from .campaign import BRANCHES, build_scientific_plan
from .contracts import artifact, validate

__all__ = ["BRANCHES", "artifact", "build_scientific_plan", "validate"]
