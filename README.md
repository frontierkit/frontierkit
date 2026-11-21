# FrontierKit

[![CI](https://github.com/frontierkit/frontierkit/actions/workflows/ci.yml/badge.svg)](https://github.com/frontierkit/frontierkit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/frontierkit)](https://pypi.org/project/frontierkit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/frontierkit/frontierkit/blob/main/LICENSE)

**Find out which model, prompt and pipeline answer your own questions best,
and how sure you can be.**

FrontierKit runs your questions through every setup you want to compare,
scores the answers against your references, and tells you which setups are
really better and which are too close to call. It then lets you check, weeks
later, that the choice still holds. It works with free model endpoints, so
trying it costs nothing.

**Website and guide: [frontierkit.github.io/frontierkit](https://frontierkit.github.io/frontierkit/)**

## What you get

Thirty reading-comprehension questions, two free models, with and without
reading material:

| Setup | Quality | Likely range | Time per answer | Against the top row |
|---|---|---|---|---|
| Gemma 4 31B, three passages | **0.95** | 0.74 to 1.00 | 362 ms | |
| Gemma 4 31B, one passage | 0.90 | 0.69 to 1.00 | 285 ms | too close to call |
| Llama 3.1 8B, three passages | 0.34 | 0.12 to 0.48 | 247 ms | worse |
| Llama 3.1 8B, one passage | 0.30 | 0.11 to 0.50 | 259 ms | worse |
| Gemma 4 31B, no passages | 0.14 | 0.01 to 0.35 | 257 ms | worse |
| Llama 3.1 8B, no passages | 0.01 | 0.00 to 0.22 | 225 ms | worse |

A leaderboard would name the first row the winner. FrontierKit says the first
two cannot be told apart on these questions, so you may as well take the
faster one, and that both clearly beat the smaller model and the same model
without reading material. Quality is word overlap with the reference answers,
from 0 to 1. The benchmark behind the table runs on free models and is in
[benchmarks/qa_frontier/](https://github.com/frontierkit/frontierkit/tree/main/benchmarks/qa_frontier).

## What it does

- **Uses your questions, not a leaderboard.** A CSV of questions and reference
  answers is enough. Scorers cover short answers and extracted triples, or
  pass your own.
- **Says how sure it is.** Every score comes with a range. Two setups are
  compared question by question, and the answer can be "too close to call".
  The best of many setups is reported with the luck of the pick taken out.
- **Weighs quality, speed and cost together.** You see the setups that are not
  beaten on all three, and pick the trade-off you want.
- **Never pays for an answer twice.** Every model reply is kept in a local
  file. Run the comparison again and no model is called; add one setup and
  only its new steps are.
- **Lets you re-check a choice.** Save what you chose and why, then run
  `frontierkit verify` on a schedule. It reports whether quality dropped, the
  setup got slower, or the model is no longer there.
- **Names failures instead of hiding them.** A rate limit, an outage or a
  retired model is recorded as what it is, not as a wrong answer.
- **Works with any vendor and your own code.** Any service or local server
  that accepts OpenAI-style chat requests: SAIA, NVIDIA, Groq, OpenRouter,
  Ollama and more. Each pipeline step can also be a plain Python function: a
  retriever, a re-ranker, a check on the answer.

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

No key yet? These two run offline:

```bash
python examples/scored_pipeline.py     # rank pipelines by measured quality
python examples/verify_decision.py     # save a choice and check it three times
```

The [get-started guide](https://frontierkit.github.io/frontierkit/GUIDE/) walks through each step: getting keys,
writing questions, reading the results, and changing what is compared.

## Use it from Python

```python
from frontierkit import (
    EvalSet, Example, enumerate_pipelines, evaluate_pipelines,
    paired_difference, quality_frontier, record_decision, token_f1,
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

for kept in quality_frontier(evaluations):                 # not beaten on quality, speed and cost
    print(kept.components, kept.quality, kept.latency_ms)

best, runner_up = sorted(evaluations, key=lambda e: e.quality, reverse=True)[:2]
print(paired_difference(best.scores, runner_up.scores).verdict)   # may be "too close to call"

record_decision(evaluations, eval_set).save("decision.json")      # to verify later
```

`registry`, `task` and `executor` describe what you are comparing;
[examples/scored_pipeline.py](https://github.com/frontierkit/frontierkit/blob/main/examples/scored_pipeline.py)
is a complete version that runs as it is.
[Using FrontierKit from Python](https://frontierkit.github.io/frontierkit/USAGE/) covers the rest: running your
own functions, calling models from several vendors, measuring cost from token
usage, keeping answers, and checking a saved choice.

## Why measure

On twelve reading-comprehension questions from SQuAD, the same Gemma model
scored 0.95 on two vendors, but answered in 230 ms on one and 14,516 ms on the
other. Without reading material it scored 0.10. A smaller model was no faster
and scored 0.33. None of that is visible from a model card.

## Where it stands

FrontierKit is an alpha-stage research library; its API may change between
minor versions.

- Trying every combination is complete and dependable.
- A few dozen questions can only separate large differences. The ranges in the
  table above are wide for that reason, and the output says what it cannot
  tell apart.
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
- [Contributing](https://github.com/frontierkit/frontierkit/blob/main/CONTRIBUTING.md) and [Changelog](https://frontierkit.github.io/frontierkit/CHANGELOG/)

## Coming from ChatCortex

FrontierKit was called ChatCortex up to version 0.12. `pip install chatcortex`
still works: it installs FrontierKit and keeps `import chatcortex` running, with
a reminder to switch.

## Citing

If you use FrontierKit in your research, please cite it; see
[CITATION.cff](https://github.com/frontierkit/frontierkit/blob/main/CITATION.cff).

## License

[MIT](https://github.com/frontierkit/frontierkit/blob/main/LICENSE). Developed by Siddharth Saraswat.
