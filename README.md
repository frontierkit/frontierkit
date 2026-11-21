# FrontierKit

[![CI](https://github.com/frontierkit/frontierkit/actions/workflows/ci.yml/badge.svg)](https://github.com/frontierkit/frontierkit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/frontierkit)](https://pypi.org/project/frontierkit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/frontierkit/frontierkit/blob/main/LICENSE)

**Find out which model and prompt answer your questions best.**

FrontierKit runs your questions through every combination of model, prompt and
pipeline step you want to compare, scores the answers against your references,
and tells you which setup to use. It works with free model endpoints, so trying
it costs nothing.

**Website and guide: [frontierkit.github.io/frontierkit](https://frontierkit.github.io/frontierkit/)**

FrontierKit was called ChatCortex up to version 0.12. `pip install chatcortex`
still works: it installs FrontierKit and keeps `import chatcortex` running, with
a reminder to switch.

## What you get

One command, four sample questions, one free model on Groq:

| Prompt | Context given | Quality | Time per answer |
|---|---|---|---|
| "shortest possible phrase" | yes | **0.92** | 292 ms |
| "shortest possible phrase" | no | 0.67 | 275 ms |
| plain question | yes | 0.14 | 369 ms |
| plain question | no | 0.06 | 508 ms |

Same model, same questions: the prompt and the context moved quality from 0.06
to 0.92. Quality is word overlap with the reference answers, from 0 to 1.

## Try it in five minutes

```bash
pip install "frontierkit[real]"
git clone https://github.com/frontierkit/frontierkit
cd frontierkit
```

Set a free API key from any of SAIA, NVIDIA, Groq or OpenRouter, for example:

```bash
export GROQ_API_KEY=...          # PowerShell: $env:GROQ_API_KEY = "..."
```

Then run it on the sample questions, or on your own CSV of questions and
reference answers:

```bash
python examples/pick_models.py --max-questions 4
python examples/pick_models.py --questions my_questions.csv
```

The [get-started guide](https://frontierkit.github.io/frontierkit/GUIDE/) walks through each step: getting keys,
writing questions, reading the results, and changing what is compared.

## What you can compare

- **Models and vendors.** Any vendor or local server that accepts OpenAI-style
  chat requests: SAIA, NVIDIA, Groq, OpenRouter, Ollama and more. No extra
  dependencies.
- **Prompts and context.** Prompt templates, with or without reading material.
- **Your own code.** Each pipeline step can be a plain Python function: a
  retriever, a re-ranker, a check on the answer.
- **Quality, speed and cost together.** Results show the setups that are not
  beaten on all three, so you can pick the trade-off you want.

## Use it from Python

```python
from frontierkit import (
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

pipelines = enumerate_pipelines(registry, task)            # every combination
evaluations = evaluate_pipelines(pipelines, executor, eval_set)

best = quality_frontier(evaluations)[0]                    # highest quality first
print(best.components, best.quality, best.latency_ms)
```

[Using FrontierKit from Python](https://frontierkit.github.io/frontierkit/USAGE/) covers the rest: describing
components and tasks, running your own functions, calling models from several
vendors, measuring cost from token usage, and letting the library choose what
to try when there are too many combinations.

## Why measure

On twelve reading-comprehension questions from SQuAD, the same Gemma model
scored 0.95 on two vendors, but answered in 230 ms on one and 14,516 ms on the
other. Without reading material it scored 0.10. A smaller model was no faster
and scored 0.33. None of that is visible from a model card. The benchmark that
produced these numbers runs on free models and is in
[benchmarks/qa_frontier/](https://github.com/frontierkit/frontierkit/tree/main/benchmarks/qa_frontier).

## Where it stands

FrontierKit is an alpha-stage research library; its API may change between
minor versions.

- Trying every combination is complete and dependable.
- For spaces too large to try in full, `run_closed_loop` chooses what to try
  from the scores it has seen. On the project's benchmark it currently finds
  the best setup in about as many calls as random picking.
- Pipelines are straight chains of steps; branching pipelines are not
  searched.

## Documentation

- [Website](https://frontierkit.github.io/frontierkit/)
- [Get started](https://frontierkit.github.io/frontierkit/GUIDE/): compare free models and prompts on your own questions
- [Using FrontierKit from Python](https://frontierkit.github.io/frontierkit/USAGE/)
- [How it works](https://frontierkit.github.io/frontierkit/ARCHITECTURE/)
- [Benchmarks](https://frontierkit.github.io/frontierkit/BENCHMARKS/)
- [Examples](https://github.com/frontierkit/frontierkit/tree/main/examples)
- [Contributing](https://github.com/frontierkit/frontierkit/blob/main/CONTRIBUTING.md) and [Changelog](https://github.com/frontierkit/frontierkit/blob/main/CHANGELOG.md)

## Citing

If you use FrontierKit in your research, please cite it; see
[CITATION.cff](https://github.com/frontierkit/frontierkit/blob/main/CITATION.cff).

## License

[MIT](https://github.com/frontierkit/frontierkit/blob/main/LICENSE). Developed by Siddharth Saraswat.
