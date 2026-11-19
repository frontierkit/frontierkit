"""Pick the best model setup for your own questions, using free models.

Tries every combination of prompt style, context and free model on a
small set of questions you supply, scores each answer against your
reference answers, and recommends a setup.

Set the keys you have (each one is optional), then run from the repo root:

    export SAIA_API_KEY=...        # GWDG SAIA / Chat AI
    export NVIDIA_API_KEY=...      # NVIDIA API catalog
    export GROQ_API_KEY=...        # Groq; also set GROQ_MODEL
    export OPENROUTER_API_KEY=...  # OpenRouter, free models only
    pip install "chatcortex[real]"
    python examples/pick_models.py

Your questions go in a CSV file with the columns question, answer and,
optionally, context. Separate acceptable answers with "|". Point the
script at your file with --questions; it defaults to
examples/pick_models_questions.csv.

Every option is tried on every question, so the number of calls is
models x prompt styles x context options x questions. Keep the question
set small at first (--max-questions). Without any key the script prints
this setup card and exits 0.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from chatcortex import (
    AgentExecutor,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    RoutedBackend,
    TaskSpecification,
    contains_answer,
    count_pipelines,
    enumerate_pipelines,
    evaluate_pipelines,
    exact_match,
    quality_frontier,
    token_f1,
)

QUESTIONS_FILE = Path(__file__).with_name("pick_models_questions.csv")

# --- the options to compare: edit these --------------------------------

PROMPT_STYLES = {
    "plain": "{question}",
    "short": "Answer with the shortest possible phrase. Do not explain.\n\n{question}",
}

# "with_context" puts the context column in front of the question.
CONTEXT_OPTIONS = ("no_context", "with_context")

# name, key variable, base URL, model variable, default model,
# seconds between calls to that provider (to stay under free-tier limits)
MODELS = [
    ("saia_llama_8b", "SAIA_API_KEY", "https://chat-ai.academiccloud.de/v1",
     "SAIA_LLAMA_MODEL", "meta-llama-3.1-8b-instruct", 6.5),
    ("saia_gemma_31b", "SAIA_API_KEY", "https://chat-ai.academiccloud.de/v1",
     "SAIA_GEMMA_MODEL", "gemma-4-31b-it", 6.5),
    ("nvidia_gemma_31b", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1",
     "NVIDIA_MODEL", "google/gemma-4-31b-it", 2.0),
    # Groq's model names change often: set GROQ_MODEL to one your key lists.
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
     "GROQ_MODEL", None, 2.5),
    ("openrouter_gemma_31b", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
     "OPENROUTER_MODEL", "google/gemma-4-31b-it:free", 3.5),
]

REQUEST_OPTIONS = {"max_tokens": 200, "temperature": 0, "timeout": 120}

SCORERS = {"f1": token_f1, "contains": contains_answer, "exact": exact_match}

# --- loading questions ---------------------------------------------------


def load_questions(path, limit=None):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "question" not in rows[0] or "answer" not in rows[0]:
        raise SystemExit(f"{path} needs the columns question and answer")
    questions = [
        {
            "question": row["question"].strip(),
            "answers": [a.strip() for a in row["answer"].split("|") if a.strip()],
            "context": (row.get("context") or "").strip(),
        }
        for row in rows
        if row.get("question", "").strip()
    ]
    return questions[:limit] if limit else questions


# --- building the pipeline steps ----------------------------------------


def prompt_steps(questions):
    """One local step per prompt style and context option: question in, prompt out."""
    contexts = {q["question"]: q["context"] for q in questions}
    has_context = any(contexts.values())
    steps = {}
    for style, template in PROMPT_STYLES.items():
        for option in CONTEXT_OPTIONS:
            if option == "with_context" and not has_context:
                continue

            def step(question, template=template, option=option):
                prompt = template.format(question=question)
                context = contexts.get(question, "")
                if option == "with_context" and context:
                    prompt = f"Context:\n{context}\n\n{prompt}"
                return prompt

            steps[f"{style}+{option}"] = step
    return steps


class SpacedBackend:
    """Keeps a minimum gap between calls that share a provider key."""

    last_call = {}

    def __init__(self, backend, provider, interval_s, sleep=time.sleep, clock=time.monotonic):
        self.backend, self.provider, self.interval_s = backend, provider, interval_s
        self.sleep, self.clock = sleep, clock

    def __call__(self, metadata, node_input):
        wait = self.interval_s - (self.clock() - self.last_call.get(self.provider, float("-inf")))
        if wait > 0:
            self.sleep(wait)
        self.last_call[self.provider] = self.clock()
        return self.backend(metadata, node_input)


def metadata(name, capability):
    # Declared numbers are placeholders: every option gets measured.
    return ComponentMetadata(
        name=name,
        component_type="model" if capability == "answer" else "tool",
        capabilities=[capability],
        cost_per_call=0.0,
        avg_latency_ms=0.0,
        reliability_score=1.0,
        privacy_level="external" if capability == "answer" else "internal",
    )


def available_models(base_url, api_key):
    try:
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        return sorted(model.id for model in client.models.list())
    except Exception:  # listing is a convenience only
        return []


def model_backends(questions):
    """Backends for the models whose keys are set and that answer a test call."""
    from chatcortex.adapters.openai_backend import OpenAIBackend

    backends = {}
    print("Checking models")
    for name, key_var, base_url, model_var, default_model, interval in MODELS:
        api_key = os.environ.get(key_var)
        if not api_key:
            continue
        model = os.environ.get(model_var, default_model)
        if not model:
            names = available_models(base_url, api_key)
            print(f"  {name:22s} skipped: set {model_var}. Models for this key: "
                  f"{', '.join(names[:12]) or 'could not list'}")
            continue
        backend = SpacedBackend(
            OpenAIBackend(model=model, base_url=base_url, api_key=api_key,
                          request_options=REQUEST_OPTIONS),
            provider=key_var,
            interval_s=interval,
        )
        print(f"  {name:22s} {model:32s}", end=" ", flush=True)
        probe = backend(metadata(name, "answer"), questions[0]["question"])
        if probe.success:
            print(f"ok ({probe.latency_ms:.0f} ms)")
            backends[name] = backend
        else:
            print(f"failed, skipped: {str(probe.output)[:140]}")
            names = available_models(base_url, api_key)
            if names:
                print(f"      models for this key: {', '.join(names[:12])}")
    return backends


# --- the comparison -------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", default=str(QUESTIONS_FILE), help="CSV file")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="f1")
    args = parser.parse_args(argv)

    if not any(os.environ.get(row[1]) for row in MODELS):
        print(__doc__)
        return 0
    try:
        import openai  # noqa: F401
    except ImportError:
        print('Install the real-backend extra first: pip install "chatcortex[real]"')
        return 1

    questions = load_questions(args.questions, args.max_questions)
    steps = prompt_steps(questions)
    models = model_backends(questions)
    if not models:
        print("No model answered. Check the keys and model names above.")
        return 1

    registry = CapabilityRegistry()
    for name in steps:
        registry.register(metadata(name, "prompt"))
    for name in models:
        registry.register(metadata(name, "answer"))
    task = TaskSpecification(required_capabilities=["prompt", "answer"])
    pipelines = enumerate_pipelines(registry, task)

    executor = AgentExecutor(backend=RoutedBackend(models, default=CallableBackend(steps)))
    eval_set = EvalSet(
        examples=[Example(q["question"], expected=q["answers"]) for q in questions],
        scorer=SCORERS[args.scorer],
    )

    calls = count_pipelines(registry, task) * len(questions)
    print(f"\n{len(pipelines)} setups x {len(questions)} questions = {calls} model calls")
    evaluations = []
    for candidate in pipelines:
        prompt, model = [m.name for _, m in candidate.graph.iter_components()]
        print(f"  {model:22s} {prompt:24s}", end=" ", flush=True)
        (evaluation,) = evaluate_pipelines([candidate], executor, eval_set)
        evaluations.append(evaluation)
        print(f"quality {evaluation.quality:.2f}, {evaluation.latency_ms:6.0f} ms per answer")

    frontier = {e.components for e in quality_frontier(evaluations)}
    ranked = sorted(evaluations, key=lambda e: (-e.quality, e.latency_ms))
    print(f"\nResults ({args.scorer} score; * = not beaten on both quality and speed)")
    print(f"  {'model':22s} {'prompt':24s} quality  answered  latency")
    for e in ranked:
        prompt, model = e.components
        mark = "*" if e.components in frontier else " "
        print(f"{mark} {model:22s} {prompt:24s} {e.quality:7.2f}  {e.success_rate:7.0%}  "
              f"{e.latency_ms:6.0f} ms")

    best = ranked[0]
    good_enough = [e for e in ranked if e.quality >= best.quality - 0.05]
    fastest = min(good_enough, key=lambda e: e.latency_ms)
    print(f"\nBest quality:  {best.components[1]} with {best.components[0]} "
          f"({best.quality:.2f}, {best.latency_ms:.0f} ms)")
    if fastest is not best:
        print(f"Faster option: {fastest.components[1]} with {fastest.components[0]} "
              f"({fastest.quality:.2f}, {fastest.latency_ms:.0f} ms), within 0.05 of the best")
    print("\nWith only a few questions, small differences are noise; add questions "
          "before relying on a close call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
