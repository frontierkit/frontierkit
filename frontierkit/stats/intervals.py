"""Margins of error for measured scores.

A pipeline scored on a handful of examples has a quality that could
easily have come out higher or lower. These functions put a range around
it, and say whether two pipelines can be told apart::

    from frontierkit.stats import mean_interval, paired_difference

    mean_interval(best.scores)                       # range around quality
    gap = paired_difference(best.scores, other.scores)
    gap.verdict                                      # "too close to call"

Two methods are offered:

- ``"betting"`` (the default for means) needs only that scores stay
  inside known bounds, ``[0, 1]`` unless told otherwise. Its error rate
  holds for any number of examples, and it still holds when the interval
  is recomputed as more examples are added in the same order, so looking
  early and often is allowed. The price is a wider interval. It follows
  the betting confidence sequences of Waudby-Smith and Ramdas (2023).
- ``"bootstrap"`` resamples the examples. It is narrower and works for
  any figure, including whole-set metrics that are not a mean, but it is
  approximate: below about 20 examples it tends to be too narrow, and it
  is not valid under repeated looking.

Both assume the examples are a fair sample of the questions you care
about. Neither corrects for picking the best of many pipelines: the
winner's own score is biased upwards, so re-measure it on examples that
played no part in the choice.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

_METHODS = ("betting", "bootstrap")
_GRID = 10000  # resolution of the betting method: candidate means per unit of range
_BET_CAP = 0.5  # largest share of capital staked on one example


@dataclass(frozen=True)
class Interval:
    """An estimate and the range it plausibly lies in.

    Attributes:
        estimate: The figure measured on all examples.
        low: Lower end of the range.
        high: Upper end of the range.
        confidence: Share of intervals built this way that should hold
            the true value, e.g. 0.95.
        method: ``"betting"`` or ``"bootstrap"``.
        n: Number of examples used.
    """

    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
            "n": self.n,
        }


@dataclass(frozen=True)
class PairedDifference(Interval):
    """The mean per-example difference, first minus second, with its range."""

    @property
    def decided(self) -> bool:
        """True when the range excludes zero."""
        return self.low > 0 or self.high < 0

    @property
    def verdict(self) -> str:
        if self.low > 0:
            return "first is better"
        if self.high < 0:
            return "second is better"
        return "too close to call"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "verdict": self.verdict}


def _check_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number between 0 and 1, {confidence!r} given")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, {confidence!r} given")
    return 1 - confidence


def _check_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, {method!r} given")


def _check_resamples(resamples: int) -> None:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError(f"resamples must be a whole number of at least 100, {resamples!r} given")


def _numbers(scores: Sequence[float], name: str) -> List[float]:
    values = list(scores)
    if not values:
        raise ValueError(f"{name} must hold at least one score")
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must hold finite numbers, {value!r} given")
    return [float(value) for value in values]


def _check_bounds(values: Sequence[float], lower: float, upper: float, name: str) -> None:
    if not lower < upper:
        raise ValueError(f"lower must be below upper, {lower!r} and {upper!r} given")
    for value in values:
        if not lower <= value <= upper:
            raise ValueError(
                f'the "betting" method needs every score in {name} between '
                f"{lower} and {upper}, {value!r} given; pass the true bounds as "
                'lower and upper, or use method="bootstrap"'
            )


def _betting_unit_bounds(values: Sequence[float], alpha: float) -> Tuple[float, float]:
    """Betting confidence bounds for the mean of ``values``, each in ``[0, 1]``.

    For every candidate mean ``m`` a gambler bets, example by example,
    that the true mean is above ``m`` and, separately, that it is below.
    If ``m`` is the true mean neither wager can be expected to grow, so
    capital reaching ``2 / alpha`` at any point is evidence against ``m``
    and it is dropped. Bet sizes depend on earlier examples only, which
    is what makes the guarantee hold at every step.
    """
    # Bet sizes, before capping: larger while the scores seen so far vary little.
    bets = []
    total = 0.5  # running sum, started with one imagined score of 1/2
    spread = 0.25  # running squared deviations, started at variance 1/4
    scale = 2 * math.log(2 / alpha)
    for step, value in enumerate(values, start=1):
        variance = spread / step
        bets.append(math.sqrt(scale / (variance * step * math.log(step + 1))))
        mean = (total + value) / (step + 1)
        total += value
        spread += (value - mean) ** 2

    threshold = math.log(2 / alpha)

    def too_low(index: int) -> bool:
        """The wager that the mean is above this candidate wins."""
        candidate = index / _GRID
        cap = _BET_CAP / candidate if candidate > 0 else math.inf
        capital = 0.0  # in logs
        for bet, value in zip(bets, values):
            capital += math.log1p(min(bet, cap) * (value - candidate))
            if capital >= threshold:
                return True
        return False

    def too_high(index: int) -> bool:
        """The wager that the mean is below this candidate wins."""
        candidate = index / _GRID
        cap = _BET_CAP / (1 - candidate) if candidate < 1 else math.inf
        capital = 0.0
        for bet, value in zip(bets, values):
            capital += math.log1p(-min(bet, cap) * (value - candidate))
            if capital >= threshold:
                return True
        return False

    # Raising the candidate can only shrink the first wager and grow the
    # second, at every step. So the candidates that are too low form a
    # block at the bottom, those that are too high a block at the top,
    # and each edge can be found by bisection. Candidate 1 is never too
    # low and candidate 0 never too high.
    start, end = 0, _GRID
    while start < end:
        middle = (start + end) // 2
        if too_low(middle):
            start = middle + 1
        else:
            end = middle
    kept_low = start
    start, end = 0, _GRID
    while start < end:
        middle = (start + end + 1) // 2
        if too_high(middle):
            end = middle - 1
        else:
            start = middle
    kept_high = start

    if kept_low > kept_high:
        # Every candidate was ruled out, which a fair sample makes very
        # unlikely; fall back to the plain mean rather than an empty range.
        mean = sum(values) / len(values)
        return mean, mean
    # The true mean may lie between a kept grid point and its dropped neighbour.
    return max(0.0, (kept_low - 1) / _GRID), min(1.0, (kept_high + 1) / _GRID)


def _betting_bounds(
    values: Sequence[float], alpha: float, lower: float, upper: float
) -> Tuple[float, float]:
    width = upper - lower
    unit = [min(1.0, max(0.0, (value - lower) / width)) for value in values]
    low, high = _betting_unit_bounds(unit, alpha)
    return lower + low * width, lower + high * width


def _bootstrap_bounds(
    statistic: Callable[[List[int]], float], n: int, alpha: float, resamples: int, seed: int
) -> Tuple[float, float]:
    """Percentile bounds of ``statistic`` over examples drawn with replacement."""
    rng = random.Random(seed)
    draws = sorted(
        statistic([rng.randrange(n) for _ in range(n)]) for _ in range(resamples)
    )
    low = draws[math.floor((alpha / 2) * (resamples - 1))]
    high = draws[math.ceil((1 - alpha / 2) * (resamples - 1))]
    return low, high


def _mean_bounds(
    values: List[float],
    alpha: float,
    method: str,
    lower: float,
    upper: float,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    if method == "betting":
        return _betting_bounds(values, alpha, lower, upper)
    n = len(values)
    return _bootstrap_bounds(
        lambda picked: sum(values[i] for i in picked) / n, n, alpha, resamples, seed
    )


def mean_interval(
    scores: Sequence[float],
    confidence: float = 0.95,
    method: str = "betting",
    lower: float = 0.0,
    upper: float = 1.0,
    resamples: int = 2000,
    seed: int = 0,
) -> Interval:
    """The mean of ``scores`` and the range the true mean plausibly lies in.

    Args:
        scores: One score per example, e.g. ``PipelineEvaluation.scores``.
        confidence: How often the range should hold the true mean.
        method: ``"betting"`` or ``"bootstrap"``; see the module notes.
        lower, upper: The bounds every score must respect; used by the
            betting method only.
        resamples, seed: Number of resamples and random seed for the
            bootstrap method; the same seed gives the same interval.

    Raises:
        ValueError: On an empty or non-numeric ``scores``, a bad
            ``confidence`` or ``method``, or, for the betting method, a
            score outside the bounds.
    """
    alpha = _check_confidence(confidence)
    _check_method(method)
    _check_resamples(resamples)
    values = _numbers(scores, "scores")
    if method == "betting":
        _check_bounds(values, lower, upper, "scores")
    low, high = _mean_bounds(values, alpha, method, lower, upper, resamples, seed)
    return Interval(
        estimate=sum(values) / len(values),
        low=low,
        high=high,
        confidence=confidence,
        method=method,
        n=len(values),
    )


def paired_difference(
    first: Sequence[float],
    second: Sequence[float],
    confidence: float = 0.95,
    method: str = "betting",
    lower: float = 0.0,
    upper: float = 1.0,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedDifference:
    """Compare two pipelines scored on the same examples.

    Each example's score under ``second`` is subtracted from its score
    under ``first``. Comparing example by example removes the part of
    the noise that comes from some examples simply being harder.

    Args:
        first, second: Scores in the same example order, e.g. the
            ``scores`` of two evaluations of one eval set.
        confidence, method, resamples, seed: As in :func:`mean_interval`.
        lower, upper: The bounds of a single score, not of a difference.

    Returns:
        The mean difference with its range. ``verdict`` reads
        ``"first is better"``, ``"second is better"``, or
        ``"too close to call"`` when the range includes zero.

    Raises:
        ValueError: If the two do not hold the same number of scores, or
            as in :func:`mean_interval`.
    """
    alpha = _check_confidence(confidence)
    _check_method(method)
    _check_resamples(resamples)
    one = _numbers(first, "first")
    two = _numbers(second, "second")
    if len(one) != len(two):
        raise ValueError(
            "first and second must score the same examples: "
            f"{len(one)} and {len(two)} scores given"
        )
    if method == "betting":
        _check_bounds(one, lower, upper, "first")
        _check_bounds(two, lower, upper, "second")
    differences = [a - b for a, b in zip(one, two)]
    reach = upper - lower
    low, high = _mean_bounds(differences, alpha, method, -reach, reach, resamples, seed)
    return PairedDifference(
        estimate=sum(differences) / len(differences),
        low=low,
        high=high,
        confidence=confidence,
        method=method,
        n=len(differences),
    )


def metric_interval(
    outputs: Sequence[Any],
    expecteds: Sequence[Any],
    metric: Callable[[Sequence[Any], Sequence[Any]], float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> Interval:
    """A whole-set metric and its range, by resampling the examples.

    Use this for figures that are not a mean of per-example scores, such
    as the pooled triple metrics. Examples are drawn with replacement and
    the metric recomputed each time, so the range reflects how much the
    figure depends on which examples happened to be in the set. With few
    examples the range is unreliable however many triples each holds.

    Args:
        outputs: One output per example, e.g. ``PipelineEvaluation.outputs``.
        expecteds: The reference of each example, in the same order.
        metric: ``metric(outputs, expecteds) -> number``.
        confidence, resamples, seed: As in :func:`mean_interval`.

    Raises:
        ValueError: If there are no examples or the lengths differ.
    """
    alpha = _check_confidence(confidence)
    _check_resamples(resamples)
    outputs = list(outputs)
    expecteds = list(expecteds)
    if not outputs:
        raise ValueError("outputs must hold at least one example")
    if len(outputs) != len(expecteds):
        raise ValueError(
            "outputs and expecteds must be the same length, "
            f"{len(outputs)} and {len(expecteds)} given"
        )
    low, high = _bootstrap_bounds(
        lambda picked: float(
            metric([outputs[i] for i in picked], [expecteds[i] for i in picked])
        ),
        len(outputs),
        alpha,
        resamples,
        seed,
    )
    return Interval(
        estimate=float(metric(outputs, expecteds)),
        low=low,
        high=high,
        confidence=confidence,
        method="bootstrap",
        n=len(outputs),
    )
