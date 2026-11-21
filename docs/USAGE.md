# Using ChatCortex from Python

The [get-started guide](GUIDE.md) runs a ready-made script. This page shows
the pieces that script is built from, so you can compare anything: your own
retrievers, prompts, checkers and models.

Everything here is importable from `chatcortex` unless noted.

## The three ideas

- A **component** is one option for one step of a pipeline: a model, a prompt
  template, a retriever. It is described by a `ComponentMetadata` record and
  kept in a `CapabilityRegistry`.
- A **task** (`TaskSpecification`) names the steps a pipeline needs, in order,
  plus optional limits on cost, latency and privacy.
- A **pipeline** is one component per step. ChatCortex lists or searches the
  possible pipelines, runs them, and measures what they do.

```python
from chatcortex import CapabilityRegistry, ComponentMetadata, TaskSpecification

registry = CapabilityRegistry()
registry.register(ComponentMetadata(
    name="short_prompt", component_type="tool", capabilities=["prompt"],
    cost_per_call=0.0, avg_latency_ms=0.0, reliability_score=1.0,
    privacy_level="internal",
))
# ... one record per option ...

task = TaskSpecification(required_capabilities=["prompt", "answer"])
```

The declared numbers are starting guesses. They matter less once you measure.

## Run your own functions

Back each component with a plain Python function. Each function receives the
previous step's output and returns its own.

```python
from chatcortex import AgentExecutor, CallableBackend

backend = CallableBackend({
    "retrieve_keyword": lambda question: search(question),
    "answer_short": lambda passages: draft(passages),
    "verify_grounded": lambda answer: check(answer),
})
backend.missing(registry)   # [] when every registry component is covered

result = AgentExecutor(backend=backend).execute(pipeline.graph, task_input="your question")
result.output               # what the last step returned
result.success              # False if any step raised or reported failure
```

Latency is measured around each call. A function that raises, or returns
`StepResult(output=None, success=False)`, is recorded as a failed step and
stops the pipeline. Cost is the declared `cost_per_call` unless the function
returns `StepResult(output, cost=...)`.

Runnable version:
[examples/callable_pipeline.py](https://github.com/siddharth1012/chatCortex/blob/main/examples/callable_pipeline.py).

## Call models from any vendor

`OpenAIBackend` talks to any vendor or local server that accepts OpenAI-style
chat requests. Install the extra first: `pip install "chatcortex[real]"`.

```python
import os
from chatcortex import RoutedBackend
from chatcortex.adapters.openai_backend import OpenAIBackend

backend = RoutedBackend({
    "model_a": OpenAIBackend(model="model-a", base_url="https://vendor-a.example/v1",
                             api_key=os.environ["VENDOR_A_KEY"]),
    "model_b": OpenAIBackend(model="model-b", base_url="https://vendor-b.example/v1",
                             api_key=os.environ["VENDOR_B_KEY"]),
}, default=CallableBackend({...}))   # local steps go to the default
```

- `RoutedBackend` sends each component to its own backend, so one comparison
  can mix vendors and local functions.
- `request_options={"max_tokens": 300, "temperature": 0}` is sent with every
  request. Vendors differ in what they accept: current Claude models, for
  example, reject a `temperature` below 1.
- Read keys from the environment. Do not write them into source files.

### Measure cost from token usage

Give the backend your prices and each call is costed from the token counts the
vendor reports:

```python
from chatcortex.adapters.openai_backend import OpenAIBackend, TokenPrice

backend = OpenAIBackend(
    model="your-model",
    prices={"your-model": TokenPrice(input_per_million=2.0, output_per_million=8.0)},
)
```

Prices are yours to supply because they change. A model without a price
records the declared `cost_per_call`.

## Score pipelines on your examples

An `EvalSet` pairs example inputs with a scorer. `evaluate_pipelines` runs each
pipeline on every example and reports what was measured.

```python
from chatcortex import (
    EvalSet, Example, enumerate_pipelines, evaluate_pipelines,
    quality_frontier, token_f1,
)

eval_set = EvalSet(
    examples=[
        Example("Which organ produces insulin?", expected="pancreas"),
        Example("What is the largest ocean?", expected=["Pacific Ocean", "Pacific"]),
    ],
    scorer=token_f1,
)

pipelines = enumerate_pipelines(registry, task)      # every combination
evaluations = evaluate_pipelines(pipelines, executor, eval_set)

best = quality_frontier(evaluations)[0]              # highest quality first
best.components, best.quality, best.latency_ms, best.cost, best.success_rate
```

- **Scorers.** `token_f1` (word overlap, partial credit), `contains_answer`
  (the reference appears in the output) and `exact_match`. Each accepts one
  reference or a list. Or pass your own `scorer(output, expected)` returning a
  bool or a number from 0 to 1.
- **Failures.** A run that fails part-way scores 0. See
  [When a run fails](#when-a-run-fails) to tell an outage from a wrong answer.
- **`quality_frontier`** keeps the pipelines not beaten on measured cost,
  latency and quality together.
- **`enumerate_pipelines`** lists every pipeline the task allows;
  `count_pipelines` tells you how many that is before you spend any calls.

### When a run fails

A run can fail because the answer was wrong for the pipeline (a step rejected
it) or because the service never answered. Each evaluation records which:

```python
best.errors                  # (None, "rate_limited", None, ...): one entry per example
best.error_counts            # {"rate_limited": 2}
best.quality                 # failed runs count as 0
best.quality_when_answered   # mean over the runs that finished; None if none did
```

| Kind | Meaning |
|---|---|
| `rate_limited` | Too many calls; the service refused this one |
| `timeout` | The call did not finish in time |
| `unavailable` | The service could not be reached or returned a server error |
| `not_found` | The model or address does not exist: retired, renamed or misspelt |
| `auth` | The key is missing, wrong or not allowed to use the model |
| `empty_output` | The model answered with nothing |
| `step_failed` | A step returned `StepResult(success=False)` |
| `error` | Any other exception, usually a bug in a step |

The first five describe the service, not the pipeline. If a pipeline shows any
of them, run it again before comparing its `quality` with others: a setup that
was rate limited on half its examples looks half as good as it is.
`quality_when_answered` shows what it scored on the rest, but it covers fewer
examples, so treat it as a hint and not as the result.

Your own step can name a kind with
`StepResult(None, success=False, error="timeout")`, and `classify_error(exc)`
gives the kind of any exception.

### Look at each example

`quality` is a mean. Each evaluation also keeps what happened on every
example, in the order of the eval set:

```python
best.scores     # (1.0, 0.0, 1.0, ...): one score per example, 0 for a failed run
best.outputs    # what the pipeline returned for each example, None for a failed run
```

Use `scores` to see which examples a pipeline gets wrong, or to compare two
pipelines example by example. Use `outputs` to grade the same run with another
scorer without paying for the calls again.

### Score extracted triples

When a pipeline extracts triples rather than answering a question,
`triple_precision`, `triple_recall` and `triple_f1` compare what it extracted
with a reference set:

```python
from chatcortex import triple_f1, triple_precision, triple_recall

eval_set = EvalSet(
    examples=[Example(document_text, expected=[
        ("provider", "must document", "training data"),
        ("user", "may request", "an explanation"),
    ])],
    scorer=triple_f1,
)
evaluation = evaluate_pipelines(pipelines, executor, eval_set)[0]

# Precision and recall for the same run, from the kept outputs:
precision = [triple_precision(output, example.expected)
             for output, example in zip(evaluation.outputs, eval_set.examples)]
```

- The last step must return a collection of triples, each a tuple or list of
  fields. Text or dicts are rejected, so parse the model's reply first.
- A triple counts only when every field matches after the same normalisation
  the answer scorers use. Add a field, such as a modality, to make the match
  stricter.
- Scores are per example, so every example weighs the same however many
  triples it holds. For figures over all triples together, use metrics.

### Figures over the whole set

Some figures are not a mean of per-example scores: precision over all triples
in a corpus, or an F1 averaged over labels. Pass those as `metrics`. Each is a
function of all outputs and all references, computed once per pipeline:

```python
from chatcortex import (
    label_macro_f1, pooled_triple_f1, pooled_triple_precision, pooled_triple_recall,
)

eval_set = EvalSet(
    examples=examples,
    scorer=triple_f1,
    metrics={
        "precision": pooled_triple_precision,
        "recall": pooled_triple_recall,
        "f1": pooled_triple_f1,
        "modality_f1": label_macro_f1,
    },
)
evaluation = evaluate_pipelines(pipelines, executor, eval_set)[0]
evaluation.metrics      # {"precision": 0.67, "recall": 0.40, ...}
```

- **Pooled triple metrics** count matches over all examples together, so a
  long document weighs more than a short one. A triple is only matched against
  the reference of its own example.
- **`label_macro_f1`** treats one field of each triple as its label (the last
  by default, or `field=...`) and averages the F1 of each label, so a rare
  label counts as much as a common one. By default a missed or invented triple
  lowers the F1 of its label, so the figure covers extraction and labelling
  together. With `matched_only=True` only triples found in both the output and
  the reference count, so it covers labelling alone:
  `lambda outputs, expecteds: label_macro_f1(outputs, expecteds, matched_only=True)`.
- **Your own metric** is any `metric(outputs, expecteds)` returning a number.
  The output of a failed run is `None`.
- Metrics are reported, not searched on: pipelines are still ranked by
  `quality`.

## How sure is the result

A quality measured on a few examples could easily have come out higher or
lower. `mean_interval` puts a range around it, and `paired_difference` says
whether two pipelines can be told apart:

```python
from chatcortex import mean_interval, paired_difference

interval = mean_interval(best.scores)
interval.estimate, interval.low, interval.high     # 0.92, 0.55, 1.0

gap = paired_difference(best.scores, other.scores)  # same examples, same order
gap.estimate, gap.low, gap.high
gap.verdict     # "first is better", "second is better" or "too close to call"
```

`paired_difference` compares example by example, which removes the noise from
some examples simply being harder. Two pipelines whose own ranges overlap can
still be told apart this way.

There are two methods:

| `method` | Use it for | Keep in mind |
|---|---|---|
| `"betting"` (default) | Scores between 0 and 1, any number of examples | Wider. Stays valid if you recompute it as examples are added, so you may look early and stop |
| `"bootstrap"` | A narrower range once you have many examples; figures that are not a mean | Approximate, and too narrow on small sets |

On simulated scores, the default range held the true value in at least 97% of
2,000 runs at every size from 12 to 200 examples. The bootstrap range held it
in 71% to 94% of runs at 12 examples and about 94% at 30. Expect a wide range
from a small set: about 0.5 from end to end at 12 examples, 0.3 at 30 and 0.15
at 200. The range need not be centred on the estimate.

For a metric over the whole set, `metric_interval` resamples the examples:

```python
from chatcortex import metric_interval

expecteds = [example.expected for example in eval_set.examples]
metric_interval(evaluation.outputs, expecteds, pooled_triple_f1)
```

- The range reflects the number of examples, not the number of triples. Five
  long documents are five examples; split them into sections if the sections
  can be judged on their own.
- These ranges describe one pipeline or one pair. The best of many pipelines
  looks better than it is, so re-measure the winner on examples that played no
  part in choosing it.

## When there are too many pipelines to try

`run_closed_loop` chooses what to try. Each round it proposes pipelines,
scores the ones it has not scored yet, estimates how good each component is
from the scores, and proposes again.

```python
from chatcortex import ExhaustiveSynthesizer, run_closed_loop

records = run_closed_loop(
    registry, task,
    synthesizer=lambda reg: ExhaustiveSynthesizer(reg, keep_ties=True),
    executor=executor, eval_set=eval_set,
    prior_strength=20, iterations=8, explore=2,
)
scored = {e.components: e for r in records for e in r.evaluations}
```

- `keep_ties=True` keeps pipelines that look identical on paper, so each gets
  measured.
- `explore=N` also tries the N most promising untried pipelines each round.
  Without it the loop stops looking once its own shortlist is scored.
- A pipeline is never scored twice.

On the project's question-answering benchmark, this finds the best pipeline in
about as many calls as picking pipelines at random. Use it when trying
everything is out of reach; otherwise `enumerate_pipelines` is simpler and
complete. See [Benchmarks](BENCHMARKS.md).

## Search on declared numbers only

Without running anything, a synthesizer returns the pipelines that are not
beaten on declared cost, latency and reliability:

```python
from chatcortex import BeamSynthesizer, build_demo_registry

registry = build_demo_registry(capabilities=("retrieval", "generation", "verification"))
task = TaskSpecification(
    required_capabilities=["retrieval", "generation", "verification"],
    max_cost=0.02, max_latency=1500,
)
for pipeline in BeamSynthesizer(registry, beam_width=5).synthesize(task):
    print(pipeline.total_cost, pipeline.total_latency, pipeline.total_reliability)
```

| Synthesizer | Strategy |
|---|---|
| `ExhaustiveSynthesizer` | Exact frontier; for small spaces |
| `ProgressiveParetoBeamSynthesizer` | Beam search that widens with depth; the best approximate one |
| `ParetoPartialBeamSynthesizer` | Beam search with per-stage pruning |
| `BeamSynthesizer` | Beam search on one weighted score |
| `HeuristicSynthesizer` | Greedy, one result |
| `RandomSynthesizer` | Random baseline |

## Correct declared numbers from runs

`calibrate` blends what was measured into the registry. With
`prior_strength=n`, the declared value counts as `n` observations, so a
component seen a few times barely moves and one seen often converges on what
was measured.

`examples/closed_loop.py` tests this against a hidden ground truth: the
simulator draws outcomes from true values the loop never sees. Over 20 seeds
and 16 rounds, mean reliability error across twelve components was:

| Calibration | Mean error | Runs that beat the declared values |
|---|---|---|
| None (declared values) | 0.030 | n/a |
| Fixed weight, `alpha=0.4` | 0.066 | 0 of 20 |
| Evidence-weighted, `prior_strength=20` | 0.024 | 17 of 20 |

It needs the evidence: at four rounds the evidence-weighted rule does not yet
beat the declared values on average.

## Save and check registries

```python
from chatcortex import load_registry, save_registry, validate_registry_file

save_registry(registry, "registry.json")     # writes, then reloads to prove it
validate_registry_file("registry.json").valid
registry = load_registry("registry.json")
```

```bash
chatcortex-registry validate registry.json
```

Errors name the row and the field. Measured quality estimates are saved with
the file.
