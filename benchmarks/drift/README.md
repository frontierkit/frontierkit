# Does the best setup stay the best?

Hosted models change without notice: a vendor swaps hardware, retires a model,
tightens a rate limit. This benchmark asks the same questions of the same
models every day and reports whether the setup you would have chosen on the
first day still holds.

It uses the 30 SQuAD questions and the free models of the
[QA frontier benchmark](../qa_frontier/README.md): each model answers every
question with no context, one paragraph and three paragraphs, and each reply is
scored as it is and cut to its first clause.

## Run it

Set the keys you have (`SAIA_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`,
`OPENROUTER_API_KEY`), then take today's sample:

```bash
python -m benchmarks.drift.collect
```

Every answer, and every failed call with its kind, is appended to
`benchmarks/drift/answers.jsonl` under today's date. If a run is cut short or
rate limited, run it again the same day: it only asks what is still missing.
Models without a key are skipped.

Then read the log:

```bash
python -m benchmarks.drift.report
```

For each day it prints the best setup, its quality as measured and corrected
for having been picked from many, how many setups are too close to it to call,
and the failed calls per model. A change is reported only when the earlier
choice is no longer the best nor too close to call.

## Run it every day

One sample is 90 calls per model and takes about twenty minutes on the free
tiers. Schedule the collect command once a day on a machine that stays on, with
the keys in that machine's environment:

- Linux or macOS: a `cron` entry.
- Windows: a Task Scheduler task.

Missing a day loses that day's sample; it does not harm the others.

## Options

- `--label NAME` names the sample. The default is the date in UTC. Use a label
  such as `2026-11-21-evening` to take more than one sample a day.
- `--questions N` uses the first N questions, for a quick trial.
- `--log PATH` writes to, or reads from, another file.
- `--ask-keys` prompts for missing keys without saving them.
