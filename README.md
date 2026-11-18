# ChatCortex

[![CI](https://github.com/siddharth1012/chatCortex/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharth1012/chatCortex/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/chatcortex)](https://pypi.org/project/chatcortex/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ChatCortex** is a framework for **automated synthesis and optimization of AI agent architectures** — think *AutoML for AI agents*.

Instead of manually wiring LLM pipelines (`retrieval → LLM → verifier → tool`), ChatCortex treats agent design as a **multi-objective architecture search problem** and automatically discovers architectures that optimize **cost**, **latency**, and **reliability** simultaneously.

```
Task Specification → Architecture Search → Pareto-Optimal Architectures
```

**ChatCortex re-synthesizes your architecture from what was measured.** Execution telemetry is calibrated back into the component registry, so every new synthesis round runs against observed behavior instead of declared guesses. `examples/closed_loop.py` tests this offline against a **hidden ground truth**: the starting registry is wrong about four components, and the simulator draws outcomes from true values the loop never sees. After sixteen iterations with evidence-weighted calibration (`prior_strength=20`, seed 42, 677 probes):

| Component | Field | Declared | Calibrated | Truth |
|---|---|---|---|---|
| `retrieval_speedster` | reliability | 0.78 | 0.707 | 0.55 |
| `retrieval_budget` | reliability | 0.82 | 0.945 | 0.95 |
| `generation_balanced` | latency (ms) | 180 | 281.1 | 320 |
| `verification_reliable` | cost per call | 0.006 | 0.0080 | 0.009 |

One seed is one draw, so the example also repeats the run over 20 seeds and reports mean reliability error across all twelve components:

| Calibration rule | Mean reliability error | Runs that beat the declared registry |
|---|---|---|
| None (declared values) | 0.030 | n/a |
| Fixed `alpha=0.4` | 0.066 | 0 of 20 |
| Evidence-weighted, `prior_strength=20` | 0.024 | 17 of 20 |

> **What this does and does not show.** Evidence-weighted calibration improves on the declared registry given enough probes; a fixed blend weight makes reliability worse. It needs the evidence: in the same test at four iterations it does not yet beat the declared values on average. A component that drops off the frontier stops being probed, which is why `retrieval_speedster` above is still far from its true value.

## Installation

```bash
pip install chatcortex
```

Or from source:

```bash
git clone https://github.com/siddharth1012/chatCortex
cd chatCortex
pip install -e .
```

## Quickstart

```python
from chatcortex import (
    TaskSpecification,
    BeamSynthesizer,
    build_demo_registry,
)

# A registry of components with cost/latency/reliability tradeoffs
registry = build_demo_registry(
    capabilities=("retrieval", "generation", "verification")
)

# The task: an ordered capability chain with hard constraints
task = TaskSpecification(
    required_capabilities=["retrieval", "generation", "verification"],
    max_cost=0.02,
    max_latency=1500,
)

# Synthesize Pareto-optimal architectures
synth = BeamSynthesizer(registry, beam_width=5)
for arch in synth.synthesize(task):
    print(
        f"cost=${arch.total_cost:.4f} "
        f"latency={arch.total_latency:.0f}ms "
        f"reliability={arch.total_reliability:.3f}"
    )
```

Or run the full demo comparing three synthesis strategies:

```bash
python examples/quickstart.py
```

To see the full loop — load a JSON registry, synthesize a frontier, execute one
architecture, and calibrate the registry's metadata from real telemetry — run:

```bash
python examples/registry_from_json.py
```

Run the flagship closed loop — synthesize → execute → calibrate →
re-synthesize, four iterations of measured behavior flowing back into the
registry:

```bash
python examples/closed_loop.py
```

With an OpenAI key you can run the same loop against real backends:

```bash
pip install "chatcortex[real]"
export OPENAI_API_KEY=...   # every reply spends your credits
python examples/closed_loop_real.py
```

Give the backend your prices and each call is costed from the token usage the
API reports, instead of the declared `cost_per_call`:

```python
from chatcortex.adapters.openai_backend import OpenAIBackend, TokenPrice

backend = OpenAIBackend(
    model="your-model",
    prices={"your-model": TokenPrice(input_per_million=2.0, output_per_million=8.0)},
)
```

Prices are yours to supply because they are model-specific and change over
time. A model without a price falls back to the declared cost.

### Other vendors and local servers

Many model providers and local servers accept the same chat-completions
request format. Point the backend at one with `base_url` and `api_key`; no
extra dependency is needed. `RoutedBackend` sends each component to its own
backend, so one search can compare models from several vendors:

```python
import os
from chatcortex import RoutedBackend
from chatcortex.adapters.openai_backend import OpenAIBackend

backend = RoutedBackend({
    "writer_a": OpenAIBackend(model="model-a", base_url="https://vendor-a.example/v1",
                              api_key=os.environ["VENDOR_A_KEY"]),
    "writer_b": OpenAIBackend(model="model-b", base_url="https://vendor-b.example/v1",
                              api_key=os.environ["VENDOR_B_KEY"]),
})
```

`examples/compare_endpoints.py` scores two providers on the same three
questions and prints measured quality and latency for each. It reads
`SAIA_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` and
`ANTHROPIC_API_KEY` from the environment and runs whichever are set; edit its `PROVIDERS` table to use others. Request
options are set per provider, because vendors differ in what they accept:
current Claude models, for example, reject a `temperature` below 1.

To search over your own components, build a `CapabilityRegistry` and register
`ComponentMetadata` describing each model, tool, or verifier — see
`examples/` for complete walkthroughs.

## Run your own functions

Back each component with a plain Python function and the executor runs a
synthesized pipeline for real: every function receives the previous step's
output and returns its own.

```python
from chatcortex import AgentExecutor, CallableBackend

backend = CallableBackend({
    "retrieve_keyword": lambda question: search(question),
    "answer_short": lambda passages: draft(passages),
    "verify_grounded": lambda answer: check(answer),
})
backend.missing(registry)   # [] when every registry component is covered

result = AgentExecutor(backend=backend).execute(arch.graph, task_input="your question")
result.output               # what the last step returned
result.success              # False if any step raised or reported failure
```

Latency is measured around each call. A function that raises, or returns
`StepResult(output=None, success=False)`, is recorded as a failed step and
stops the pipeline. Cost is the declared `cost_per_call` unless the function
returns `StepResult(output, cost=...)`. A complete runnable version:

```bash
python examples/callable_pipeline.py
```

## Score pipelines on your own examples

Declared numbers say nothing about whether a pipeline's answers are any good.
Give ChatCortex example inputs and a scoring function, and it runs each
pipeline on every example and reports measured quality, cost and latency.

```python
from chatcortex import EvalSet, Example, evaluate_pipelines, quality_frontier

eval_set = EvalSet(
    examples=[
        Example("What is the boiling point of water?", expected="100"),
        Example("How quickly does light travel?", expected="299,792"),
    ],
    scorer=lambda output, expected: expected in str(output),   # bool or 0..1
)

evaluations = evaluate_pipelines(frontier, executor, eval_set)
best = quality_frontier(evaluations)[0]     # highest measured quality first
best.components, best.quality, best.cost, best.success_rate
```

A run that fails part-way scores 0. `quality_frontier` keeps the pipelines not
beaten on measured cost, latency and quality. In `examples/scored_pipeline.py`
the three frontier pipelines score 1.00, 0.00 and 0.00 on four questions,
although their declared reliabilities are 0.91, 0.85 and 0.71.

Only pipelines the synthesizer proposes are scored. Pass `eval_set=` to
`run_closed_loop` and every example run also feeds calibration, so a component
that was declared better than it is loses its place and pipelines it was hiding
get proposed and scored in later rounds.

```bash
python examples/scored_pipeline.py
```

## Registry packs

A registry ships as a JSON pack in the loader's document shape. Saving it
proves the round-trip, and one command validates any pack offline:

```python
from chatcortex import build_demo_registry, save_registry, validate_registry_file

save_registry(build_demo_registry(), "registry.json")  # writes, then reloads to prove it
print(validate_registry_file("registry.json").valid)   # True
```

```bash
chatcortex-registry validate registry.json
#   components validated: 12
#   result: VALID          (exit 0; issues and duplicates are named when invalid)
```

## How It Works

ChatCortex is organized as a layered architecture synthesis system:

```
TaskSpecification → CapabilityRegistry → Synthesis Engine → AgentGraph (DAG)
    → Execution Engine → Telemetry → Evaluation Harness → Pareto Optimization
```

**Core concepts:**

- **ComponentMetadata** — declarative description of an agent component (model, retriever, tool, verifier, memory) with cost per call, latency, reliability score, and privacy level.
- **CapabilityRegistry** — stores components and filters candidates by capability and privacy constraints.
- **TaskSpecification** — the synthesis problem: an ordered capability chain plus hard constraints (max cost, max latency, privacy) and objective weights.
- **AgentGraph** — architectures represented as DAGs; cost is additive, latency sequential, reliability multiplicative.
- **Synthesizers** — search strategies that map a task to an (approximate) Pareto frontier.

**Available synthesizers:**

| Synthesizer | Strategy |
|---|---|
| `HeuristicSynthesizer` | Greedy deterministic construction |
| `RandomSynthesizer` | Random baseline |
| `BeamSynthesizer` | Budget-aware beam search |
| `ParetoPartialBeamSynthesizer` | Beam search with per-stage Pareto pruning |
| `ProgressiveParetoBeamSynthesizer` | Depth-aware beam widening (best Pareto recovery) |
| `ExhaustiveSynthesizer` | Exact Pareto frontier (small spaces) |

Approximation quality is measured with **Pareto coverage**, **hypervolume loss**, and **average regret** — see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system design and component layers
- [Benchmarks](docs/BENCHMARKS.md) — experimental evaluation of synthesis strategies
- [Examples](examples/) — runnable demos and budget-sweep experiments
- [QA frontier benchmark](benchmarks/qa_frontier/) — finds the pipelines worth choosing between on real questions, using free models only
- [Contributing](CONTRIBUTING.md) — development setup and PR guidelines
- [Changelog](CHANGELOG.md)

## Research Context

ChatCortex is a controlled experimental platform for studying automated agent architecture synthesis: multi-objective optimization, architecture search, reliability–cost tradeoffs, and AutoML-style agent design. Experiments cover 5-stage pipelines with Pareto frontiers up to 95 architectures and evaluation budgets from 20 to 180 evaluations; progressive beam widening shows improved Pareto recovery over static beam strategies.

If you use ChatCortex in your research, please cite it (see [CITATION.cff](CITATION.cff)).

## Roadmap

- [x] Component modeling, capability registry, task specification, DAG representation
- [x] Exact Pareto frontier computation (exhaustive search)
- [x] Budget-aware beam search with Pareto pruning and progressive widening
- [ ] Graph-structured (non-chain) agent synthesis
- [ ] Real model / tool integrations (OpenAI chat-completions backend already available behind the `chatcortex[real]` extra; more backends planned)
- [x] Registry loading from JSON (`load_registry`, with error messages naming the failing field); YAML planned

## Status

Research framework under active development (alpha). APIs may change between minor versions.

## License

[MIT](LICENSE) — developed by **Siddharth Saraswat**
