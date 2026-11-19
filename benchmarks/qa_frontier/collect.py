"""Collect model replies for the QA frontier benchmark.

Asks every free model every distinct prompt the full grid of pipelines
needs, and stores each reply in the response cache. It is safe to stop
and rerun: replies already cached are never requested again.

Set the keys you have, then run from the repo root:

    export SAIA_API_KEY=...  NVIDIA_API_KEY=...  GROQ_API_KEY=...  OPENROUTER_API_KEY=...
    python -m benchmarks.qa_frontier.collect --budget 100

It uses the bundled 30 SQuAD questions unless --data names another
SQuAD-format file (sample.json holds six quick ones). --budget caps how many
calls this run makes. Models whose key is not set are skipped. A provider
that hits its rate limit is paused for the rest of the run; rerun later
to continue. Only free models are used.
"""

import argparse
import getpass
import os
import sys
import time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from chatcortex import ComponentMetadata

from .qa import (
    DEFAULT_CACHE,
    FREE_MODELS,
    PROVIDER_MIN_INTERVAL_S,
    REQUEST_OPTIONS,
    SQUAD_SUBSET,
    ModelSpec,
    Question,
    ResponseCache,
    load_questions,
    needed_calls,
)

# (success, output or error text, latency in ms)
Reply = Tuple[bool, str, float]
Caller = Callable[[str], Reply]


def openai_caller(spec: ModelSpec, api_key: str) -> Caller:
    """A caller backed by ChatCortex's OpenAI-compatible backend."""
    from chatcortex.adapters.openai_backend import OpenAIBackend

    backend = OpenAIBackend(
        model=spec.model,
        base_url=spec.base_url,
        api_key=api_key,
        request_options=REQUEST_OPTIONS,
    )
    metadata = ComponentMetadata(
        name=spec.name,
        component_type="model",
        capabilities=["answer"],
        cost_per_call=0.0,
        avg_latency_ms=spec.declared_latency_ms,
        reliability_score=0.9,
        privacy_level="external",
    )

    def call(prompt: str) -> Reply:
        probe = backend(metadata, prompt)
        return probe.success, str(probe.output or ""), probe.latency_ms

    return call


def _is_rate_limit(error: str) -> bool:
    return "RateLimitError" in error or "429" in error


def collect(
    questions: Sequence[Question],
    cache: ResponseCache,
    models: Sequence[ModelSpec] = FREE_MODELS,
    budget: Optional[int] = None,
    env: Mapping[str, str] = os.environ,
    make_caller: Callable[[ModelSpec, str], Caller] = openai_caller,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> Dict[str, Dict[str, int]]:
    """Fill ``cache`` with the replies the grid needs; return per-model counts."""
    callers: Dict[str, Caller] = {}
    stopped: Dict[str, str] = {}  # model or provider name -> reason
    for spec in models:
        api_key = env.get(spec.key_var)
        if api_key:
            callers[spec.name] = make_caller(spec, api_key)
        else:
            stopped[spec.name] = f"{spec.key_var} not set"

    calls = needed_calls(questions, models)
    pending = [(spec, prompt) for spec, prompt in calls if cache.get(spec.name, prompt) is None]
    log(f"{len(calls)} replies needed, {len(calls) - len(pending)} already cached")

    last_call: Dict[str, float] = {}
    made = 0
    for spec, prompt in pending:
        if budget is not None and made >= budget:
            log(f"Budget of {budget} calls reached; rerun to continue.")
            break
        if spec.name in stopped or spec.provider in stopped:
            continue

        wait = PROVIDER_MIN_INTERVAL_S.get(spec.provider, 0.0) - (
            clock() - last_call.get(spec.provider, float("-inf"))
        )
        if wait > 0:
            sleep(wait)
        last_call[spec.provider] = clock()

        ok, output, latency_ms = callers[spec.name](prompt)
        made += 1
        if ok:
            cache.put(spec.name, prompt, output, latency_ms)
            log(f"  [{made}] {spec.name:22s} ok   {latency_ms:8.0f} ms")
        elif _is_rate_limit(output):
            stopped[spec.provider] = "rate limited"
            log(f"  [{made}] {spec.name:22s} rate limited; pausing {spec.provider} for this run")
        else:
            stopped[spec.name] = output
            log(f"  [{made}] {spec.name:22s} failed; skipping it this run: {output[:160]}")

    summary: Dict[str, Dict[str, int]] = {}
    for spec, prompt in calls:
        counts = summary.setdefault(spec.name, {"cached": 0, "needed": 0})
        counts["needed"] += 1
        counts["cached"] += cache.get(spec.name, prompt) is not None
    log("")
    log(f"{'model':22s} {'cached':>7s} / needed   note")
    for spec in models:
        counts = summary[spec.name]
        note = stopped.get(spec.name) or stopped.get(spec.provider) or ""
        log(f"{spec.name:22s} {counts['cached']:7d} / {counts['needed']:<6d}   {note[:60]}")
    return summary


def ask_for_keys(prompt=getpass.getpass) -> None:
    """Prompt for each missing key; it lives only in this process."""
    for key_var in dict.fromkeys(spec.key_var for spec in FREE_MODELS):
        if os.environ.get(key_var):
            continue
        value = prompt(f"{key_var} (input hidden, Enter to skip): ").strip()
        if value:
            os.environ[key_var] = value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(SQUAD_SUBSET), help="SQuAD-format JSON file")
    parser.add_argument("--questions", type=int, default=None, help="sample this many questions")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--budget", type=int, default=None, help="max calls this run")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="response cache file")
    parser.add_argument(
        "--ask-keys", action="store_true",
        help="prompt for keys that are not set; typing is hidden and nothing is saved",
    )
    args = parser.parse_args(argv)

    if args.ask_keys:
        ask_for_keys()

    if not any(os.environ.get(spec.key_var) for spec in FREE_MODELS):
        print(__doc__)
        return 0

    questions = load_questions(args.data, limit=args.questions, seed=args.seed)
    print(f"{len(questions)} questions from {args.data}")
    collect(questions, ResponseCache(args.cache), budget=args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
