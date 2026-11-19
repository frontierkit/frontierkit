"""Compare models from different vendors on the same questions.

Any vendor or local server that accepts OpenAI-style chat-completions
requests works: set its key in the environment and the script scores it
on three short questions, reporting measured quality and latency.

Set any of these keys, then run from the repo root:

    export SAIA_API_KEY=...        # GWDG SAIA / Chat AI
    export NVIDIA_API_KEY=...      # NVIDIA API catalog
    export GROQ_API_KEY=...        # Groq
    export OPENROUTER_API_KEY=...  # OpenRouter (free models only by default)
    export ANTHROPIC_API_KEY=...   # Claude, through its OpenAI-compatible endpoint
    pip install "chatcortex[real]"
    python examples/compare_endpoints.py

Optional: SAIA_MODEL, NVIDIA_MODEL, GROQ_MODEL, OPENROUTER_MODEL and
ANTHROPIC_MODEL choose the model.

Every default is a free model. A paid model runs only when you opt in
with ALLOW_PAID_MODELS=1: that covers the Claude row and any OpenRouter
model without the ":free" suffix. Paid rows report cost per question,
measured from token usage when the row lists the model's price.

Each provider gets 4 small requests (one connection check, three
questions). A provider that fails the check is skipped, and the script
lists the model names its key can use. Without any key the script prints
this setup card and exits 0.
"""

import os
import sys

from chatcortex import (
    AgentExecutor,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    ExhaustiveSynthesizer,
    RoutedBackend,
    TaskSpecification,
    evaluate_pipelines,
)
from chatcortex.adapters.openai_backend import TokenPrice

# Vendors differ in which request options they accept, so each row
# carries its own. Current Claude models reject a temperature below 1 and
# think before answering, so that row sets no temperature and leaves room
# for the answer.
# REQUEST_TIMEOUT_S bounds one request; the SDK retries a timed-out
# request, so a dead endpoint gives up after a few minutes, not hours.
REQUEST_TIMEOUT_S = 120
SHORT_ANSWER = {"max_tokens": 300, "temperature": 0, "timeout": REQUEST_TIMEOUT_S}

# Free tiers are marked FREE. Paid rows list prices in dollars per million
# tokens, checked against each vendor's published prices on 2026-11-18.
FREE = "free"

# name, key variable, base URL, model variable, default model,
# request options, prices
PROVIDERS = [
    (
        "saia",
        "SAIA_API_KEY",
        "https://chat-ai.academiccloud.de/v1",
        "SAIA_MODEL",
        "meta-llama-3.1-8b-instruct",
        SHORT_ANSWER,
        FREE,
    ),
    (
        "nvidia",
        "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_MODEL",
        "google/gemma-4-31b-it",
        SHORT_ANSWER,
        FREE,
    ),
    (
        "groq",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
        SHORT_ANSWER,
        FREE,
    ),
    (
        "openrouter",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_MODEL",
        "google/gemma-4-31b-it:free",
        SHORT_ANSWER,
        # Free while the model name ends in ":free"; any other OpenRouter
        # model is paid. Example price for one paid choice:
        {"anthropic/claude-haiku-5.5": TokenPrice(0.10, 0.50)},
    ),
    (
        "claude",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        "ANTHROPIC_MODEL",
        "claude-opus-5-5",
        {"max_tokens": 1024, "timeout": REQUEST_TIMEOUT_S},
        {"claude-opus-5-5": TokenPrice(4.00, 20.00)},
    ),
]

INSTRUCTION = "Answer in one short sentence. "
EXAMPLES = [
    Example(INSTRUCTION + "What is the capital of Australia?", expected="Canberra"),
    Example(INSTRUCTION + "What is 17 multiplied by 6?", expected="102"),
    Example(INSTRUCTION + "Which planet is known as the Red Planet?", expected="Mars"),
]


def contains_answer(output, expected):
    return expected.lower() in str(output).lower()


def component(name, cost, latency):
    # Declared values are placeholders; the point is what gets measured.
    return ComponentMetadata(
        name=name,
        component_type="model",
        capabilities=["answer"],
        cost_per_call=cost,
        avg_latency_ms=latency,
        reliability_score=0.9,
        privacy_level="external",
    )


def suggest_models(row, limit=40):
    """After a failed check, print the model names this key can use."""
    name, key_var, base_url, model_var = row[:4]
    try:
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=os.environ[key_var])
        names = sorted(model.id for model in client.models.list())
    except Exception as exc:  # listing is a convenience; never fail on it
        print(f"           (could not list models: {type(exc).__name__})")
        return
    if not names:
        return
    print(f"           Models available to this key; set {model_var} to one of them:")
    for model_name in names[:limit]:
        print(f"             {model_name}")
    if len(names) > limit:
        print(f"             ... and {len(names) - limit} more")


def main():
    try:
        from chatcortex.adapters.openai_backend import OpenAIBackend
    except ImportError:  # pragma: no cover
        OpenAIBackend = None

    configured = {}
    cost_labels = {}
    for index, row in enumerate(PROVIDERS):
        name, key_var, base_url, model_var, default_model, options, prices = row
        api_key = os.environ.get(key_var)
        if not api_key:
            continue
        model = os.environ.get(model_var, default_model)
        if name == "openrouter" and model.endswith(":free"):
            prices = FREE
        if prices != FREE and os.environ.get("ALLOW_PAID_MODELS") != "1":
            print(
                f"Skipping {name} ({model}): it is a paid model. "
                "Set ALLOW_PAID_MODELS=1 to include it."
            )
            continue
        try:
            provider_backend = OpenAIBackend(
                model=model,
                base_url=base_url,
                api_key=api_key,
                request_options=options,
                prices=None if prices == FREE else prices,
            )
        except ImportError as exc:
            print(exc)
            return 0
        # Distinct placeholder values so no pipeline is discarded before
        # it has been measured.
        metadata = component(name, cost=0.001 * (index + 1), latency=1000 - 100 * index)
        if prices == FREE:
            cost_labels[name] = "free tier"
        elif model in prices:
            cost_labels[name] = "measured"
        else:
            cost_labels[name] = "not priced"
        configured[name] = (provider_backend, metadata, model, row)

    if not configured:
        print(__doc__)
        return 0

    print("Connection check")
    registry = CapabilityRegistry()
    routes = {}
    models = {}
    for name, (provider_backend, metadata, model, row) in configured.items():
        print(f"  {name:10s} {model:34s}", end=" ", flush=True)
        probe = provider_backend(metadata, EXAMPLES[0].input)
        status = "ok" if probe.success else "FAILED"
        print(f"{status}  {probe.latency_ms:7.0f} ms", flush=True)
        if probe.success:
            registry.register(metadata)
            routes[name] = provider_backend
            models[name] = model
            continue
        print(f"           {probe.output}")
        suggest_models(row)

    if not routes:
        print("No provider answered. Fix the errors above and rerun.")
        return 1
    skipped = [name for name in configured if name not in routes]
    if skipped:
        print(f"Continuing without: {', '.join(skipped)}")

    executor = AgentExecutor(backend=RoutedBackend(routes))

    task = TaskSpecification(required_capabilities=["answer"])
    frontier = ExhaustiveSynthesizer(registry).synthesize(task)
    eval_set = EvalSet(examples=EXAMPLES, scorer=contains_answer)

    # Free and shared endpoints can take a minute per answer, so report
    # each provider as it finishes instead of going quiet.
    print()
    print(f"Asking {len(EXAMPLES)} questions per provider (slow endpoints can take minutes)")
    evaluations = []
    for candidate in frontier:
        name = next(metadata.name for _, metadata in candidate.graph.iter_components())
        print(f"  {name} ...", end=" ", flush=True)
        (evaluation,) = evaluate_pipelines([candidate], executor, eval_set)
        evaluations.append(evaluation)
        print(f"done, {evaluation.latency_ms:.0f} ms per answer", flush=True)

    print()
    print(f"{len(EXAMPLES)} questions per provider")
    print(
        f"  {'provider':10s} {'model':34s} quality  success  mean latency  "
        "cost per question"
    )
    for evaluation in sorted(evaluations, key=lambda e: -e.quality):
        name = evaluation.components[0]
        cost = (
            f"${evaluation.cost:.6f}"
            if cost_labels[name] == "measured"
            else cost_labels[name]
        )
        print(
            f"  {name:10s} {models[name]:34s} {evaluation.quality:7.2f}  "
            f"{evaluation.success_rate:7.2f}  {evaluation.latency_ms:9.0f} ms  {cost}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
