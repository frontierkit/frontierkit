"""Run synthesized pipelines as plain Python functions - no API keys.

Every component in the registry is backed by an ordinary function. The
synthesizer proposes the pipelines worth choosing between, and the
executor runs each one on a real question: every function receives the
previous step's output and returns its own.

Run from the repo root:

    python examples/callable_pipeline.py
"""

from chatcortex import (
    AgentExecutor,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    ExhaustiveSynthesizer,
    StepResult,
    TaskSpecification,
)

FACTS = {
    "boiling point of water": "Water boils at 100 degrees Celsius at sea level.",
    "speed of light": "Light travels at 299,792 kilometres per second in a vacuum.",
}

QUESTION = "What is the boiling point of water?"


# --- retrieval: two ways to find a passage --------------------------------

def retrieve_exact(question):
    """Cheap and strict: the question must contain a known topic."""
    for topic, passage in FACTS.items():
        if topic in question.lower():
            return {"question": question, "passage": passage}
    return StepResult(output=None, success=False)


def retrieve_keyword(question):
    """Looser: picks the passage sharing the most words with the question."""
    words = set(question.lower().strip("?").split())
    passage = max(FACTS.values(), key=lambda p: len(words & set(p.lower().split())))
    return {"question": question, "passage": passage}


# --- generation: two ways to answer ---------------------------------------

def answer_verbatim(found):
    return {**found, "answer": found["passage"]}


def answer_short(found):
    return {**found, "answer": found["passage"].split(" at ")[0] + "."}


# --- verification ---------------------------------------------------------

def verify_grounded(draft):
    """Pass only answers whose every word appears in the passage."""
    passage_words = set(draft["passage"].lower().strip(".").split())
    answer_words = set(draft["answer"].lower().strip(".").split())
    if not answer_words <= passage_words:
        return StepResult(output=None, success=False)
    return draft["answer"]


IMPLEMENTATIONS = {
    "retrieve_exact": retrieve_exact,
    "retrieve_keyword": retrieve_keyword,
    "answer_verbatim": answer_verbatim,
    "answer_short": answer_short,
    "verify_grounded": verify_grounded,
}

# name, capability, declared cost, declared latency (ms), declared reliability
DECLARED = [
    ("retrieve_exact", "retrieval", 0.001, 5, 0.80),
    ("retrieve_keyword", "retrieval", 0.002, 20, 0.95),
    ("answer_verbatim", "generation", 0.004, 300, 0.97),
    ("answer_short", "generation", 0.002, 150, 0.90),
    ("verify_grounded", "verification", 0.001, 10, 0.99),
]


def build_registry():
    registry = CapabilityRegistry()
    for name, capability, cost, latency, reliability in DECLARED:
        registry.register(
            ComponentMetadata(
                name=name,
                component_type="tool",
                capabilities=[capability],
                cost_per_call=cost,
                avg_latency_ms=latency,
                reliability_score=reliability,
                privacy_level="internal",
            )
        )
    return registry


def main():
    registry = build_registry()
    backend = CallableBackend(IMPLEMENTATIONS)
    assert backend.missing(registry) == []

    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"]
    )
    frontier = ExhaustiveSynthesizer(registry).synthesize(task)
    executor = AgentExecutor(backend=backend)

    print(f"Question: {QUESTION}")
    print(f"{len(frontier)} pipelines on the frontier, each run once:")
    print()
    for candidate in frontier:
        steps = " -> ".join(m.name for _, m in candidate.graph.iter_components())
        result = executor.execute(candidate.graph, task_input=QUESTION)
        measured_ms = sum(r["latency_ms"] for r in result.telemetry.records)
        print(steps)
        print(
            f"  declared: cost ${candidate.total_cost:.3f}, "
            f"{candidate.total_latency:.0f} ms, "
            f"reliability {candidate.total_reliability:.3f}"
        )
        print(f"  measured: {measured_ms:.3f} ms, success={result.success}")
        print(f"  output:   {result.output!r}")
        print()


if __name__ == "__main__":
    main()
