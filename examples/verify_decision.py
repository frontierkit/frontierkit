"""Record a choice, then check later that it still holds.

Runs offline with stand-in "models", so it needs no key::

    python examples/verify_decision.py

It compares two setups on 40 questions, saves the choice to
``decision.json`` in a temporary folder, and then checks that choice
three times: with nothing changed, after the chosen model has got worse,
and after it has been retired.

The same check from the command line, for a scheduled job::

    frontierkit verify decision.json --setup examples/verify_decision.py

``setup()`` below is what that command calls.
"""

import tempfile
from pathlib import Path

from frontierkit import (
    AgentExecutor,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    TaskSpecification,
    enumerate_pipelines,
    evaluate_pipelines,
    exact_match,
    record_decision,
    verify_decision,
)

QUESTIONS = [(f"What is {n} + {n}?", str(n + n)) for n in range(1, 41)]

# How the stand-in "careful" model behaves; the demo changes this later.
STATE = {"careful": "good"}


class ModelRetired(Exception):
    status_code = 404  # what a vendor answers for a model that no longer exists


def careful(question: str) -> str:
    n = int(question.split()[2])
    if STATE["careful"] == "retired":
        raise ModelRetired("model careful-1 not found")
    if STATE["careful"] == "worse" and n % 2 == 0:
        return "I am not sure."
    return str(n + n)


def sloppy(question: str) -> str:
    n = int(question.split()[2])
    return str(n + n) if n % 3 else str(n + n + 1)


def component(name: str) -> ComponentMetadata:
    return ComponentMetadata(
        name=name,
        component_type="model",
        capabilities=["answer"],
        cost_per_call=0.0,
        avg_latency_ms=1,
        reliability_score=0.9,
        privacy_level="internal",
    )


def setup() -> dict:
    """What the comparison is made with; ``frontierkit verify`` calls this."""
    registry = CapabilityRegistry()
    registry.register(component("careful"))
    registry.register(component("sloppy"))
    task = TaskSpecification(required_capabilities=["answer"])
    return {
        "pipelines": enumerate_pipelines(registry, task),
        "executor": AgentExecutor(backend=CallableBackend({"careful": careful, "sloppy": sloppy})),
        "eval_set": EvalSet(
            examples=[Example(question, expected=answer) for question, answer in QUESTIONS],
            scorer=exact_match,
        ),
    }


def main() -> int:
    given = setup()
    pipelines, executor, eval_set = given["pipelines"], given["executor"], given["eval_set"]

    evaluations = evaluate_pipelines(pipelines, executor, eval_set)
    decision = record_decision(evaluations, eval_set, note="demo")
    path = decision.save(Path(tempfile.mkdtemp()) / "decision.json")
    print(f"Chosen: {decision.chosen['components'][0]} "
          f"(quality {decision.chosen['quality']:.2f}); saved to {path}\n")

    for label, state in (
        ("Nothing changed", "good"),
        ("The model got worse", "worse"),
        ("The model was retired", "retired"),
    ):
        STATE["careful"] = state
        check = verify_decision(decision, pipelines, executor, eval_set)
        print(f"{label}\n  {check.status} (exit code {check.exit_code}), "
              f"{check.asked} of {check.examples} questions asked\n  {check.message}\n")
    STATE["careful"] = "good"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
