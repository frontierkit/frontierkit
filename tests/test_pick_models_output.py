"""The guide script reports ranges, and does not name a winner it cannot support."""

import random
import sys
from pathlib import Path

from chatcortex import PipelineEvaluation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import pick_models  # noqa: E402


def evaluation(model, prompt, scores, latency_ms=100.0, errors=None):
    errors = tuple(errors or [None] * len(scores))
    return PipelineEvaluation(
        candidate=None,
        components=(prompt, model),
        runs=len(scores),
        quality=sum(scores) / len(scores),
        success_rate=sum(kind is None for kind in errors) / len(scores),
        cost=0.0,
        latency_ms=latency_ms,
        traces=(),
        scores=tuple(scores),
        errors=errors,
    )


def printed(evaluations):
    lines = []
    result = pick_models.print_results(evaluations, "f1", out=lines.append)
    return result, "\n".join(lines)


def test_every_row_shows_its_likely_range():
    _, text = printed(
        [
            evaluation("big", "short", [1.0] * 40),
            evaluation("small", "short", [0.0] * 40),
        ]
    )

    assert "likely range" in text
    rows = [line for line in text.splitlines() if " to " in line and "ms" in line]
    assert len(rows) == 2
    assert "Best quality: big with short" in text
    assert "Too close to call" not in text


def test_few_questions_are_reported_as_too_close_to_call():
    _, text = printed(
        [
            evaluation("big", "short", [1.0, 1.0, 1.0, 0.0], latency_ms=900),
            evaluation("small", "short", [1.0, 0.0, 1.0, 0.0], latency_ms=200),
        ]
    )

    assert "Too close to call: big with short scored highest (0.75)" in text
    assert "4 questions cannot tell it apart from 1 other setup:" in text
    assert "small with short (0.50); the gap is +0.25" in text
    assert "Faster and within" not in text  # 0.25 behind is not a substitute
    assert "Best quality:" not in text
    assert "because there are 4 questions" in text


def test_a_faster_setup_is_offered_only_when_it_scored_about_the_same():
    _, text = printed(
        [
            evaluation("big", "short", [1.0] * 40, latency_ms=900),
            evaluation("small", "short", [1.0] * 39 + [0.0], latency_ms=200),
        ]
    )

    assert "Too close to call: big with short scored highest (1.00)" in text
    assert "Faster and within 0.05 of the top score: small with short" in text


def test_listed_gaps_agree_with_the_verdict():
    """Every setup called too close has a gap range that includes zero."""
    rng = random.Random(4)
    setups = [
        evaluation("a", "short", [float(rng.random() < 0.9) for _ in range(12)]),
        evaluation("b", "short", [float(rng.random() < 0.8) for _ in range(12)]),
        evaluation("c", "short", [float(rng.random() < 0.6) for _ in range(12)]),
        evaluation("d", "short", [0.0] * 12),
    ]
    _, text = printed(setups)

    gaps = [line for line in text.splitlines() if "likely between" in line]
    assert gaps
    for line in gaps:
        low = float(line.split("likely between ")[1].split(" and ")[0])
        high = float(line.split(" and ")[-1])
        assert low <= 0 <= high


def test_a_flattered_top_score_is_called_out():
    rng = random.Random(11)
    many = [
        evaluation(f"m{i}", "short", [float(rng.random() < 0.5) for _ in range(20)])
        for i in range(12)
    ]

    result, text = printed(many)

    assert result.measured - result.corrected >= 0.01
    assert "flattered by being picked from 12 setups" in text
    assert f"expect about {result.corrected:.2f}" in text


def test_failed_calls_among_the_leaders_are_mentioned():
    _, text = printed(
        [
            evaluation("big", "short", [1.0, 1.0, 0.0, 0.0], errors=[None, None, "rate_limited", None]),
            evaluation("small", "short", [1.0, 0.0, 1.0, 0.0]),
        ]
    )

    assert "failed calls, which count as 0" in text


def test_a_single_setup_is_reported_without_comparison():
    _, text = printed([evaluation("only", "short", [1.0, 0.0, 1.0])])

    assert "Best quality: only with short" in text
    assert "ahead of every other" not in text
