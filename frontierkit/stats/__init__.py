"""Margins of error and paired comparisons for measured scores."""

from .intervals import (
    Interval,
    PairedDifference,
    mean_interval,
    metric_interval,
    paired_difference,
)
from .selection import BestOf, best_of, scores_by_pipeline

__all__ = [
    "Interval",
    "PairedDifference",
    "mean_interval",
    "metric_interval",
    "paired_difference",
    "BestOf",
    "best_of",
    "scores_by_pipeline",
]
