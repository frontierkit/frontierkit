"""Compare models from different vendors on the same questions.

Any vendor or local server that accepts OpenAI-style chat-completions
requests works: set its key in the environment and the script scores it
on three short questions, reporting measured quality and latency.

Set any of these keys, then run from the repo root:

    export SAIA_API_KEY=...        # GWDG SAIA / Chat AI
    export NVIDIA_API_KEY=...      # NVIDIA API catalog
    export ANTHROPIC_API_KEY=...   # Claude, through its OpenAI-compatible endpoint
    pip install "chatcortex[real]"
    python examples/compare_endpoints.py

Optional: SAIA_MODEL, NVIDIA_MODEL and ANTHROPIC_MODEL choose the model.

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

# Vendors differ in which request options they accept, so each row
# carries its own. Current Claude models reject a temperature below 1 and
# think before answering, so that row sets no temperature and leaves room
# for the answer.
# REQUEST_TIMEOUT_S bounds one request; the SDK retries a timed-out
# request, so a dead endpoint gives up after a few minutes, not hours.
REQUEST_TIMEOUT_S = 120
SHORT_ANSWER = {"max_tokens": 300, "temperature": 0, "timeout": REQUEST_TIMEOUT_S}

# name, key variable, base URL, model variable, default model, request options
PROVIDERS = [
    (
        "saia",
        "SAIA_API_KEY",
        "https://chat-ai.academiccloud.de/v1",
        "SAIA_MODEL",
        "meta-llama-3.1-8b-instruct",
        SHORT_ANSWER,
    ),
    (
        "nvidia",
        "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_MODEL",
        "google/gemma-4-31b-it",
        SHORT_ANSWER,
    ),
    (
        "claude",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        "ANTHROPIC_MODEL",
        "claude-opus-5-5",
        {"max_tokens": 1024, "timeout": REQUEST_TIMEOUT_S},
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
    name, key_var, base_url, model_var, _default_model, _options = row
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
    for index, row in enumerate(PROVIDERS):
        name, key_var, base_url, model_var, default_model, options = row
        api_key = os.environ.get(key_var)
        if not api_key:
            continue
        model = os.environ.get(model_var, default_model)
        try:
            provider_backend = OpenAIBackend(
                model=model,
                base_url=base_url,
                api_key=api_key,
                request_options=options,
            )
        except ImportError as exc:
            print(exc)
            return 0
        # Distinct placeholder values so no pipeline is discarded before
        # it has been measured.
        metadata = component(name, cost=0.001 * (index + 1), latency=1000 - 100 * index)
        configured[name] = (provider_backend, metadata, model, row)

    if not configured:
        print(__doc__)
        return 0

    print("Connection check")
    registry = CapabilityRegistry()
    routes = {}
    models = {}
    for name, (provider_backend, metadata, model, row) in configured.items():
        print(f"  {name:8s} {model:34s}", end=" ", flush=True)
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
    print(f"  {'provider':8s} {'model':34s} quality  success  mean latency")
    for evaluation in sorted(evaluations, key=lambda e: -e.quality):
        name = evaluation.components[0]
        print(
            f"  {name:8s} {models[name]:34s} {evaluation.quality:7.2f}  "
            f"{evaluation.success_rate:7.2f}  {evaluation.latency_ms:9.0f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
