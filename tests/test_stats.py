"""Margins of error: intervals for scores and paired comparisons."""

import random

import pytest

from frontierkit import (
    Interval,
    PairedDifference,
    mean_interval,
    metric_interval,
    paired_difference,
    pooled_triple_f1,
)
from frontierkit.stats import intervals

METHODS = ("betting", "bootstrap")


@pytest.mark.parametrize("method", METHODS)
def test_interval_surrounds_the_mean_and_narrows_with_more_examples(method):
    rng = random.Random(1)
    few = [float(rng.random() < 0.7) for _ in range(15)]
    many = few + [float(rng.random() < 0.7) for _ in range(185)]

    small = mean_interval(few, method=method)
    large = mean_interval(many, method=method)
    for interval, scores in ((small, few), (large, many)):
        assert isinstance(interval, Interval)
        assert interval.estimate == pytest.approx(sum(scores) / len(scores))
        assert 0.0 <= interval.low <= interval.estimate <= interval.high <= 1.0
        assert interval.contains(interval.estimate)
        assert (interval.n, interval.method, interval.confidence) == (len(scores), method, 0.95)
    assert large.high - large.low < small.high - small.low


def test_betting_interval_is_wider_than_bootstrap_and_honest_about_one_example():
    rng = random.Random(2)
    scores = [rng.random() for _ in range(40)]
    betting = mean_interval(scores)
    bootstrap = mean_interval(scores, method="bootstrap")
    assert betting.high - betting.low > bootstrap.high - bootstrap.low

    # One perfect score says almost nothing; the bootstrap cannot see that.
    assert mean_interval([1.0]).low < 0.1
    assert mean_interval([1.0], method="bootstrap").low == 1.0


def test_higher_confidence_gives_a_wider_interval():
    rng = random.Random(3)
    scores = [rng.random() for _ in range(50)]
    assert (
        mean_interval(scores, confidence=0.99).low
        < mean_interval(scores, confidence=0.95).low
        < mean_interval(scores, confidence=0.80).low
    )


def test_betting_interval_respects_custom_bounds():
    rng = random.Random(4)
    unit = [rng.random() for _ in range(30)]
    shifted = [10 + 5 * score for score in unit]
    base = mean_interval(unit)
    scaled = mean_interval(shifted, lower=10, upper=15)
    assert scaled.low == pytest.approx(10 + 5 * base.low)
    assert scaled.high == pytest.approx(10 + 5 * base.high)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        mean_interval(shifted)
    # The bootstrap needs no bounds.
    assert mean_interval(shifted, method="bootstrap").contains(sum(shifted) / 30)


def test_betting_interval_holds_the_true_mean_at_least_as_often_as_promised():
    # 300 small samples from a skewed score distribution with mean 0.2.
    # A 95% interval may miss in at most 5% of them; this method misses less.
    rng = random.Random(5)
    misses = 0
    for _ in range(300):
        scores = [float(rng.random() < 0.2) for _ in range(12)]
        misses += not mean_interval(scores).contains(0.2)
    assert misses <= 15


def test_betting_interval_stays_valid_when_checked_after_every_example():
    # Recompute after each new example and count a sample as wrong if any
    # of its 40 intervals excludes the truth. Looking 40 times must not
    # inflate the error rate.
    rng = random.Random(6)
    ever_wrong = 0
    for _ in range(150):
        scores = [rng.random() for _ in range(40)]  # true mean 0.5
        bounds = [mean_interval(scores[:n]) for n in range(1, 41)]
        ever_wrong += any(not interval.contains(0.5) for interval in bounds)
        # A candidate ruled out once stays out, so the intervals only shrink.
        for earlier, later in zip(bounds, bounds[1:]):
            assert earlier.low <= later.low and later.high <= earlier.high
    assert ever_wrong <= 150 * 0.05


@pytest.mark.parametrize("method", METHODS)
def test_paired_difference_separates_a_clear_gap_and_declines_a_small_one(method):
    rng = random.Random(7)
    strong = [float(rng.random() < 0.9) for _ in range(40)]
    weak = [float(rng.random() < 0.2) for _ in range(40)]

    gap = paired_difference(strong, weak, method=method)
    assert isinstance(gap, PairedDifference)
    assert gap.estimate == pytest.approx(sum(strong) / 40 - sum(weak) / 40)
    assert gap.low > 0 and gap.decided
    assert gap.verdict == "first is better"

    reverse = paired_difference(weak, strong, method=method)
    assert reverse.verdict == "second is better"
    assert reverse.estimate == pytest.approx(-gap.estimate)

    same = paired_difference(strong, strong, method=method)
    assert same.estimate == 0.0 and not same.decided
    assert same.verdict == "too close to call"
    assert same.to_dict()["verdict"] == "too close to call"


def test_pairing_removes_noise_shared_by_both_pipelines():
    # Both pipelines find the same examples hard; one is 0.1 better on each.
    rng = random.Random(8)
    difficulty = [rng.uniform(0.0, 0.9) for _ in range(60)]
    better = [score + 0.1 for score in difficulty]
    paired = paired_difference(better, difficulty, method="bootstrap")
    assert paired.low == pytest.approx(0.1) and paired.high == pytest.approx(0.1)
    # Their separate intervals overlap heavily, so they alone decide nothing.
    assert mean_interval(better, method="bootstrap").low < mean_interval(
        difficulty, method="bootstrap"
    ).high


def test_paired_difference_rarely_calls_a_winner_when_there_is_none():
    rng = random.Random(9)
    false_calls = 0
    for _ in range(200):
        first = [float(rng.random() < 0.6) for _ in range(20)]
        second = [float(rng.random() < 0.6) for _ in range(20)]
        false_calls += paired_difference(first, second).decided
    assert false_calls <= 10


def test_metric_interval_resamples_whole_examples():
    rng = random.Random(10)
    expecteds, outputs = [], []
    for document in range(30):
        wanted = [(f"s{document}", "must", f"o{i}") for i in range(rng.randint(1, 6))]
        expecteds.append(wanted)
        outputs.append([triple for triple in wanted if rng.random() < 0.7])

    interval = metric_interval(outputs, expecteds, pooled_triple_f1, resamples=400)
    assert interval.method == "bootstrap" and interval.n == 30
    assert interval.estimate == pytest.approx(pooled_triple_f1(outputs, expecteds))
    assert 0.0 < interval.low < interval.estimate < interval.high <= 1.0

    # Same seed, same interval; a perfect extractor has no spread.
    assert metric_interval(outputs, expecteds, pooled_triple_f1, resamples=400) == interval
    assert metric_interval(outputs, expecteds, pooled_triple_f1, resamples=400, seed=1) != interval
    perfect = metric_interval(expecteds, expecteds, pooled_triple_f1, resamples=400)
    assert (perfect.low, perfect.estimate, perfect.high) == (1.0, 1.0, 1.0)


def test_input_validation():
    with pytest.raises(ValueError, match="at least one score"):
        mean_interval([])
    with pytest.raises(ValueError, match="finite numbers"):
        mean_interval([0.5, float("nan")])
    with pytest.raises(ValueError, match="finite numbers"):
        mean_interval([0.5, "high"])
    with pytest.raises(ValueError, match="method must be one of"):
        mean_interval([0.5], method="t-test")
    with pytest.raises(ValueError, match="resamples"):
        mean_interval([0.5], method="bootstrap", resamples=10)
    for confidence in (0, 1, 95, "high"):
        with pytest.raises(ValueError, match="confidence"):
            mean_interval([0.5], confidence=confidence)
    with pytest.raises(ValueError, match="same examples"):
        paired_difference([0.5, 0.5], [0.5])
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        paired_difference([0.5], [1.5])
    with pytest.raises(ValueError, match="at least one example"):
        metric_interval([], [], pooled_triple_f1)
    with pytest.raises(ValueError, match="same length"):
        metric_interval([None], [], pooled_triple_f1)


def test_an_impossible_sample_falls_back_to_the_mean(monkeypatch):
    # With an absurd threshold every candidate mean is ruled out.
    monkeypatch.setattr(intervals.math, "log", lambda value: 1e-9)
    low, high = intervals._betting_unit_bounds([0.0, 1.0, 0.0, 1.0], 0.05)
    assert low == high == 0.5
