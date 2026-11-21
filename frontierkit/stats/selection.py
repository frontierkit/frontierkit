"""Report the best of many pipelines without flattering it.

Pick the highest-scoring of twenty pipelines and its score is too high:
part of what put it on top is luck on these particular examples, and
that luck does not carry over to new ones. The more pipelines compared
and the fewer examples, the larger the effect.

:func:`best_of` takes the per-example scores of every pipeline that was
compared, all on the same examples, and returns for the winner:

- the score as measured, which is the flattering one;
- a corrected score, an estimate of what the winner would score on new
  examples, with the luck of the pick taken out;
- a range that stays valid although the winner was chosen after looking
  at the scores;
- the contenders: the pipelines the examples cannot tell apart from the
  winner.

How it works. The corrected score splits the examples in two at random,
many times; each time it picks the winner on one half and scores that
pick on the other half, which played no part in picking it. The average
of those scores is the corrected score. It errs on the cautious side,
because a pick made on half the examples is a little worse than one made
on all of them. The range splits the allowed error over all pipelines (a
Bonferroni correction), so it holds whichever one is picked; the
contenders are found the same way, by comparing each pipeline with the
winner example by example.

When you can afford it, the plain alternative is better still: choose on
one part of the examples and report the winner's score on the rest.
"""

import random
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Sequence, Tuple

from .intervals import Interval, mean_interval, paired_difference


@dataclass(frozen=True)
class BestOf:
    """The winner among several pipelines, reported honestly.

    Attributes:
        winner: The name of the pipeline with the highest mean score;
            the first one in the order given when several tie.
        candidates: How many pipelines were compared.
        measured: The winner's mean score as measured.
        corrected: What the winner is estimated to score on new
            examples. Never above ``measured``.
        optimism: ``measured - corrected``: how much of the measured
            score is the luck of having been picked.
        interval: The range for the winner's true score, valid although
            it was picked after looking.
        contenders: Other pipelines the examples cannot separate from
            the winner, best first.
    """

    winner: Hashable
    candidates: int
    measured: float
    corrected: float
    optimism: float
    interval: Interval
    contenders: Tuple[Hashable, ...]

    @property
    def clear(self) -> bool:
        """True when no other pipeline is too close to call."""
        return not self.contenders

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "candidates": self.candidates,
            "measured": self.measured,
            "corrected": self.corrected,
            "optimism": self.optimism,
            "interval": self.interval.to_dict(),
            "contenders": list(self.contenders),
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _argmax(means: Sequence[float]) -> int:
    best = 0
    for index in range(1, len(means)):
        if means[index] > means[best]:
            best = index
    return best


def best_of(
    scores: Mapping[Hashable, Sequence[float]],
    confidence: float = 0.95,
    method: str = "betting",
    lower: float = 0.0,
    upper: float = 1.0,
    resamples: int = 2000,
    seed: int = 0,
) -> BestOf:
    """The best pipeline, its score corrected for having been picked.

    Args:
        scores: ``{name: scores}`` with one score per example for each
            pipeline, e.g. ``{e.components: e.scores for e in
            evaluations}``. Every pipeline must be scored on the same
            examples, in the same order.
        confidence: How often the winner's range should hold its true
            score, and how sure a difference must be before a pipeline
            is dropped from the contenders.
        method: ``"betting"`` or ``"bootstrap"``, as in
            :func:`~frontierkit.stats.mean_interval`.
        lower, upper: The bounds every score respects.
        resamples, seed: Number of resamples and random seed; the same
            seed gives the same result.

    Raises:
        ValueError: If ``scores`` is empty, a pipeline has no scores,
            or the pipelines were not scored on the same number of
            examples.
    """
    if not scores:
        raise ValueError("best_of needs the scores of at least one pipeline")
    names = list(scores)
    table: List[List[float]] = [list(scores[name]) for name in names]
    n = len(table[0])
    if n == 0:
        raise ValueError("every pipeline needs at least one score")
    for name, row in zip(names, table):
        if len(row) != n:
            raise ValueError(
                "every pipeline must be scored on the same examples: "
                f"{names[0]!r} has {n} scores, {name!r} has {len(row)}"
            )
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, {resamples} given")

    k = len(names)
    means = [_mean(row) for row in table]
    winner = _argmax(means)
    measured = means[winner]

    # Pick on a random half, score the pick on the other half.
    rng = random.Random(seed)
    corrected = measured
    if k > 1 and n > 1:
        order = list(range(n))
        half = n // 2
        held_out: List[float] = []
        for _ in range(resamples):
            rng.shuffle(order)
            picking, scoring = order[:half], order[half:]
            pick = _argmax([_mean([row[i] for i in picking]) for row in table])
            held_out.append(_mean([table[pick][i] for i in scoring]))
        corrected = min(measured, _mean(held_out))

    # Split the allowed error over every pipeline, so the range holds
    # whichever one comes out on top.
    alpha = 1.0 - confidence
    interval = mean_interval(
        table[winner],
        confidence=1.0 - alpha / k,
        method=method,
        lower=lower,
        upper=upper,
        resamples=resamples,
        seed=seed,
    )
    interval = Interval(
        estimate=interval.estimate,
        low=interval.low,
        high=interval.high,
        confidence=confidence,
        method=interval.method,
        n=interval.n,
    )

    contenders: List[int] = []
    if k > 1:
        each = 1.0 - alpha / (k - 1)
        for index in range(k):
            if index == winner:
                continue
            difference = paired_difference(
                table[winner],
                table[index],
                confidence=each,
                method=method,
                lower=lower,
                upper=upper,
                resamples=resamples,
                seed=seed,
            )
            if not difference.low > 0:
                contenders.append(index)
    contenders.sort(key=lambda index: -means[index])

    return BestOf(
        winner=names[winner],
        candidates=k,
        measured=measured,
        corrected=corrected,
        optimism=measured - corrected,
        interval=interval,
        contenders=tuple(names[index] for index in contenders),
    )


def scores_by_pipeline(evaluations: Sequence) -> Dict[Tuple[str, ...], Tuple[float, ...]]:
    """``{components: scores}`` for a list of evaluations, for :func:`best_of`."""
    return {evaluation.components: evaluation.scores for evaluation in evaluations}
