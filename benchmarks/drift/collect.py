"""Ask the same questions of the same models every day, and keep the answers.

Each run asks every model every question under every context setting
once, and appends the answers, failures included, to an answer log under
a label (today's date unless you pass one). Run it daily and the log
shows whether the best setup stays the best::

    python -m benchmarks.drift.collect              # today's sample
    python -m benchmarks.drift.collect --ask-keys   # type keys instead of exporting them
    python -m benchmarks.drift.report               # what changed, day by day

A run that is interrupted, or rate limited part-way, can be started
again the same day: answers already in the log for that label are not
asked for twice. Models whose key is not set are skipped.

Keys are read from the environment (SAIA_API_KEY, NVIDIA_API_KEY,
GROQ_API_KEY, OPENROUTER_API_KEY); every model in the list is free to
call. Nothing here needs a paid account.
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from benchmarks.qa_frontier.collect import ask_for_keys
from benchmarks.qa_frontier.qa import (
    CONTEXT_VARIANTS,
    FREE_MODELS,
    PROVIDER_MIN_INTERVAL_S,
    REQUEST_OPTIONS,
    SQUAD_SUBSET,
    ModelSpec,
    Question,
    Retriever,
    context_step,
    load_questions,
)
from chatcortex.registry.metadata import ComponentMetadata
from chatcortex.store import AnswerLog, LoggedBackend, fingerprint

HERE = Path(__file__).parent
DEFAULT_LOG = HERE / "answers.jsonl"

# A model that fails this many times in a row is left alone for the rest
# of the run: one retired model should not cost ninety failed calls a day.
GIVE_UP_AFTER = 3

Backend = Callable[[ComponentMetadata, str], object]


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def model_metadata(spec: ModelSpec) -> ComponentMetadata:
    return ComponentMetadata(
        name=spec.name,
        component_type="model",
        capabilities=["answer"],
        cost_per_call=0.0,
        avg_latency_ms=spec.declared_latency_ms,
        reliability_score=0.9,
        privacy_level="external",
    )


def openai_backend(spec: ModelSpec, api_key: str) -> Backend:
    from chatcortex.adapters.openai_backend import OpenAIBackend

    return OpenAIBackend(
        model=spec.model,
        base_url=spec.base_url,
        api_key=api_key,
        request_options=REQUEST_OPTIONS,
    )


def prompts_for(questions: Sequence[Question], pool: Sequence[Question]) -> List[str]:
    """Every prompt of one sample: each question under each context setting."""
    retriever = Retriever(q.context for q in pool)
    steps = [context_step(retriever, k) for k in CONTEXT_VARIANTS.values()]
    return [step(q.question) for step in steps for q in questions]


def collect_sample(
    questions: Sequence[Question],
    pool: Sequence[Question],
    log: AnswerLog,
    label: str,
    models: Sequence[ModelSpec] = FREE_MODELS,
    env: Mapping[str, str] = os.environ,
    make_backend: Callable[[ModelSpec, str], Backend] = openai_backend,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], None] = print,
) -> Dict[str, Dict[str, int]]:
    """Take one sample under ``label``; return per-model counts.

    Each count holds ``answered`` (in the log for this label, from this
    run or an earlier one), ``failed`` (failed calls made by this run)
    and ``needed``.
    """
    prompts = prompts_for(questions, pool)
    counts: Dict[str, Dict[str, int]] = {}
    notes: Dict[str, str] = {}
    paused: Dict[str, str] = {}  # provider -> reason
    last_call: Dict[str, float] = {}

    for spec in models:
        tally = counts.setdefault(spec.name, {"answered": 0, "failed": 0, "needed": len(prompts)})
        api_key = env.get(spec.key_var)
        if not api_key:
            notes[spec.name] = f"{spec.key_var} not set"
            continue
        backend = LoggedBackend(make_backend(spec, api_key), log, label=label)
        metadata = model_metadata(spec)
        identity = backend.identity(metadata)
        failures_in_a_row = 0

        for prompt in prompts:
            if log.find(fingerprint(identity, prompt), label) is not None:
                tally["answered"] += 1
                continue
            if spec.provider in paused:
                notes[spec.name] = paused[spec.provider]
                break
            if failures_in_a_row >= GIVE_UP_AFTER:
                break

            wait = PROVIDER_MIN_INTERVAL_S.get(spec.provider, 0.0) - (
                clock() - last_call.get(spec.provider, float("-inf"))
            )
            if wait > 0:
                sleep(wait)
            last_call[spec.provider] = clock()

            probe = backend(metadata, prompt)
            if probe.success:
                tally["answered"] += 1
                failures_in_a_row = 0
                out(f"  {spec.name:22s} ok   {probe.latency_ms:8.0f} ms")
                continue

            tally["failed"] += 1
            failures_in_a_row += 1
            kind = getattr(probe, "error", None) or "error"
            out(f"  {spec.name:22s} {kind}: {str(probe.output)[:120]}")
            if kind == "rate_limited":
                paused[spec.provider] = "rate limited; run again later today"
                notes[spec.name] = paused[spec.provider]
                break
            if failures_in_a_row >= GIVE_UP_AFTER:
                notes[spec.name] = f"{kind} {GIVE_UP_AFTER} times in a row; left for this run"

    out("")
    out(f"Sample {label}")
    out(f"{'model':22s} {'answered':>8s} / needed  failed  note")
    for spec in models:
        tally = counts[spec.name]
        out(
            f"{spec.name:22s} {tally['answered']:8d} / {tally['needed']:<6d} "
            f"{tally['failed']:6d}  {notes.get(spec.name, '')}"
        )
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", default=None, help="name of this sample (default: today's date, UTC)")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="answer log to append to")
    parser.add_argument("--questions", type=int, default=None, help="use only the first N questions")
    parser.add_argument("--ask-keys", action="store_true", help="prompt for missing API keys")
    args = parser.parse_args(argv)

    if args.ask_keys:
        ask_for_keys()
    pool = load_questions(SQUAD_SUBSET)
    questions = pool[: args.questions] if args.questions else pool
    counts = collect_sample(questions, pool, AnswerLog(args.log), args.label or today())
    complete = all(c["answered"] == c["needed"] for c in counts.values())
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
