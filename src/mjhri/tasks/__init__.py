"""Tasks: geometric success scoring + reset randomization for any MuJoCo scene."""

from .randomize import apply_randomization
from .scorer import CriterionResult, Outcome, Scorer

__all__ = ["CriterionResult", "Outcome", "Scorer", "apply_randomization"]
