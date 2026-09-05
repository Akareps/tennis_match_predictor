"""Leakage-aware, reproducible tennis forecasting experiments.

The original top-level scripts are retained as an archive.  New work lives in
this package so that each experiment can share the same data, chronology, and
evaluation contracts.
"""

from .data import canonicalize_matches, deduplicate_canonical_matches, load_cached_matches
from .models import EloModel, RankTransform, RankAugmentedEloModel, TierWeightedRankAugmentedEloModel

__all__ = [
    "EloModel",
    "RankAugmentedEloModel",
    "RankTransform",
    "TierWeightedRankAugmentedEloModel",
    "canonicalize_matches",
    "deduplicate_canonical_matches",
    "load_cached_matches",
]

__version__ = "0.1.0"
