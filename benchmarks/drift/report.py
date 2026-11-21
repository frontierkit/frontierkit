"""Read the daily answers and say whether the best setup changed.

For every sample in the log (one per label, normally one per day) this
scores each setup, model x context x postprocess, on the questions, and
prints the best one, the setups too close to it to call, and how many
calls each model failed. A change is reported only when yesterday's
choice is today neither the best nor too close to call: a different name
on top is not news if the two cannot be told apart.

    python -m benchmarks.drift.report

A setup counts for a sample only when every one of its questions was
asked; a failed call scores 0, as it would for someone relying on it.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.qa_frontier.qa import (
    CONTEXT_VARIANTS,
    POSTPROCESS_VARIANTS,
    SQUAD_SUBSET,
    Question,
    Retriever,
    context_step,
    load_questions,
    score,
)
from frontierkit.stats.selection import BestOf, best_of
from frontierkit.store import AnswerLog

from .collect import DEFAULT_LOG

Setup = Tuple[str, str, str]  # model, context, postprocess


def scores_by_sample(
    records: Sequence[dict], questions: Sequence[Question], pool: Sequence[Question]
) -> Dict[str, Dict[Setup, List[float]]]:
    """``{label: {setup: one score per question}}`` for complete setups."""
    retriever = Retriever(q.context for q in pool)
    where: Dict[str, Tuple[str, int]] = {}  # prompt -> (context name, question index)
    for name, k in CONTEXT_VARIANTS.items():
        step = context_step(retriever, k)
        for index, question in enumerate(questions):
            where[step(question.question)] = (name, index)

    # label -> model -> context -> question index -> output (None: failed)
    seen: Dict[str, Dict[str, Dict[str, Dict[int, Optional[str]]]]] = {}
    for record in records:
        place = where.get(record.get("input"))
        if place is None:
            continue
        context, index = place
        cell = (
            seen.setdefault(str(record.get("label", "")), {})
            .setdefault(record["component"], {})
            .setdefault(context, {})
        )
        if record.get("success"):
            cell[index] = str(record.get("output") or "")
        else:
            cell.setdefault(index, None)

    result: Dict[str, Dict[Setup, List[float]]] = {}
    for label, models in seen.items():
        for model, contexts in models.items():
            for context, answers in contexts.items():
                if len(answers) != len(questions):
                    continue
                for post, clean in POSTPROCESS_VARIANTS.items():
                    result.setdefault(label, {})[(model, context, post)] = [
                        0.0 if answers[i] is None else score(clean(answers[i]), q.answers)
                        for i, q in enumerate(questions)
                    ]
    return result


def failures_by_sample(records: Sequence[dict]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """``{label: {model: {error kind: count}}}``."""
    result: Dict[str, Dict[str, Dict[str, int]]] = {}
    for record in records:
        if record.get("success"):
            continue
        kinds = result.setdefault(str(record.get("label", "")), {}).setdefault(
            record["component"], {}
        )
        kind = record.get("error") or "error"
        kinds[kind] = kinds.get(kind, 0) + 1
    return result


def choices(samples: Dict[str, Dict[Setup, List[float]]]) -> Dict[str, BestOf]:
    """The best setup of each sample, with its contenders."""
    return {label: best_of(setups) for label, setups in sorted(samples.items()) if setups}


def changes(picked: Dict[str, BestOf]) -> List[Tuple[str, Setup, Setup]]:
    """``(label, previous choice, new best)`` wherever the choice stopped holding.

    The choice carries over from sample to sample while it remains the
    best or too close to call, and moves to the new best when it is not.
    """
    moved = []
    current: Optional[Setup] = None
    for label, best in picked.items():
        if current is not None and current != best.winner and current not in best.contenders:
            moved.append((label, current, best.winner))
            current = best.winner
        elif current is None:
            current = best.winner
    return moved


def _name(setup: Setup) -> str:
    return " / ".join(setup)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="answer log to read")
    parser.add_argument("--questions", type=int, default=None, help="the first N questions only")
    args = parser.parse_args(argv)

    records = AnswerLog(args.log).records()
    if not records:
        print(f"No answers in {args.log} yet. Run: python -m benchmarks.drift.collect")
        return 0
    pool = load_questions(SQUAD_SUBSET)
    questions = pool[: args.questions] if args.questions else pool

    samples = scores_by_sample(records, questions, pool)
    failures = failures_by_sample(records)
    picked = choices(samples)
    if not picked:
        print("No sample holds a complete setup yet; rerun the collector to finish one.")
        return 0

    print(f"{'sample':12s} {'setups':>6s}  {'best':44s} quality  corrected  too close")
    for label, best in picked.items():
        print(
            f"{label:12s} {best.candidates:6d}  {_name(best.winner):44s} "
            f"{best.measured:7.2f}  {best.corrected:9.2f}  {len(best.contenders):9d}"
        )
        for model, kinds in sorted(failures.get(label, {}).items()):
            listed = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
            print(f"{'':12s} {'':6s}  failed calls, {model}: {listed}")

    moved = changes(picked)
    print("")
    if not moved:
        print(f"The first choice held in all {len(picked)} samples.")
    for label, before, after in moved:
        print(f"{label}: {_name(before)} no longer holds; best is now {_name(after)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
