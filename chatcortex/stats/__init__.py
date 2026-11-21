"""Margins of error and paired comparisons for measured scores."""

from .intervals import (
    Interval,
    PairedDifference,
    mean_interval,
    metric_interval,
    paired_difference,
)

__all__ = [
    "Interval",
    "PairedDifference",
    "mean_interval",
    "metric_interval",
    "paired_difference",
]
