---
title: ChatCortex
hide:
  - navigation
---

# Find out which model and prompt answer your questions best

ChatCortex runs your questions through every combination of model, prompt and
pipeline step you want to compare, scores the answers against your references,
and tells you which setup to use. It works with free model endpoints, so trying
it costs nothing.

[Get started in five minutes](GUIDE.md){ .md-button .md-button--primary }
[Use it from Python](USAGE.md){ .md-button }

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

```bash
pip install "chatcortex[real]"
git clone https://github.com/siddharth1012/chatCortex && cd chatCortex
python examples/pick_models.py --questions my_questions.csv
```

## What you can compare

<div class="grid cards" markdown>

-   **Models and vendors**

    Any vendor or local server that accepts OpenAI-style chat requests: SAIA,
    NVIDIA, Groq, OpenRouter, Ollama and more. No extra dependencies.

-   **Prompts and context**

    Prompt templates, with or without reading material, in any combination.

-   **Your own code**

    Each pipeline step can be a plain Python function: a retriever, a
    re-ranker, a check on the answer.

-   **Quality, speed and cost together**

    Results show the setups that are not beaten on all three, so you can pick
    the trade-off you want.

</div>

## Why measure

On twelve reading-comprehension questions, the same Gemma model scored 0.95 on
two vendors, but answered in 230 ms on one and 14,516 ms on the other. Without
reading material it scored 0.10. A smaller model was no faster and scored 0.33.
None of that is visible from a model card. [See the benchmark](BENCHMARKS.md).

## Where it stands

ChatCortex is an alpha-stage research library by Siddharth Saraswat, MIT
licensed, with about 250 tests. Trying every combination is complete and
dependable. Its search for large spaces currently finds the best setup in about
as many calls as random picking, and the docs say so wherever it matters.

[Source on GitHub](https://github.com/siddharth1012/chatCortex){ .md-button }
[Package on PyPI](https://pypi.org/project/chatcortex/){ .md-button }
