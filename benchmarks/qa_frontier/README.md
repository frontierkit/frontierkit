# QA frontier benchmark

Which pipelines are worth choosing between, and how many model calls does it
take to find them?

Each pipeline has three steps:

| Step | Options |
|---|---|
| Context | no context, the best-matching paragraph, the best three paragraphs |
| Answer | free models from SAIA, NVIDIA, Groq and OpenRouter |
| Postprocess | the reply as it is, or cut to its first clause |

With five models that is 30 pipelines. Answers are scored with the SQuAD token
F1 against the reference answers.

Only the answer step calls a model. Every reply is stored in a response cache,
so no model is asked the same thing twice, collection can stop and resume
around free-tier rate limits, and scoring runs offline. Only free models are
used.

## Run it

Set the keys you have, then collect replies. This uses the 30 bundled SQuAD
questions; `--data benchmarks/qa_frontier/sample.json` runs six quick ones
instead, and `--data` accepts any SQuAD-format file.

```bash
python -m benchmarks.qa_frontier.collect --budget 100
```

Rerun the same command until every model shows all replies cached. Then score
every pipeline and print the true frontier, offline:

```bash
python -m benchmarks.qa_frontier.report
```

Pass the same `--data`, `--questions` and `--seed` to both commands.

## Calls needed

One call per model for each distinct prompt: questions × 3 context options ×
models. The 30 SQuAD questions and five models need 450 calls; the six sample
questions need 90. Each provider is spaced to stay under its free-tier limit, and a provider
that reports a rate limit is paused until the next run.

## Data

`squad_dev_30.json` holds 30 questions from the
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) development set:
one question from each of 30 randomly chosen articles, text unchanged. SQuAD is
by Rajpurkar, Zhang, Lopyrev and Liang (2016) and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); this subset is
shared under the same licence. `sample.json` holds six questions written for
this repository.
