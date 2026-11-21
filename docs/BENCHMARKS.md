# Benchmarks

Three sets of measurements ship with the project. The first two run on free
models and measure answers. The third measures the search strategies alone.

## Which setup answers best

[`benchmarks/qa_frontier`](https://github.com/frontierkit/frontierkit/tree/main/benchmarks/qa_frontier)
compares pipelines of three steps on reading-comprehension questions from
SQuAD: how much reading material the model gets (none, the best-matching
paragraph, or the best three), which free model answers, and whether the reply
is kept whole or cut to its first clause. Answers are scored with the SQuAD
token F1 against the reference answers.

Thirty questions and two free models on SAIA, with the reply kept whole:

| Setup | Quality | Likely range | Time per answer | Against the top row |
|---|---|---|---|---|
| Gemma 4 31B, three passages | **0.95** | 0.74 to 1.00 | 362 ms | |
| Gemma 4 31B, one passage | 0.90 | 0.69 to 1.00 | 285 ms | too close to call |
| Llama 3.1 8B, three passages | 0.34 | 0.12 to 0.48 | 247 ms | worse |
| Llama 3.1 8B, one passage | 0.30 | 0.11 to 0.50 | 259 ms | worse |
| Gemma 4 31B, no passages | 0.14 | 0.01 to 0.35 | 257 ms | worse |
| Llama 3.1 8B, no passages | 0.01 | 0.00 to 0.22 | 225 ms | worse |

The first two setups cannot be told apart on thirty questions, and both clearly
beat the rest. The ranges and verdicts come from `mean_interval` and
`paired_difference`; see [How sure is the result](USAGE.md#how-sure-is-the-result).

In an earlier run on twelve SQuAD questions, the same Gemma model scored 0.95
on two vendors but answered in 230 ms on one and 14,516 ms on the other. Without reading
material it scored 0.10, and a smaller model that was no faster scored 0.33.

Run it yourself with any of the free keys. Every reply is kept, so collection
can stop and resume around rate limits, and scoring runs offline:

```bash
python -m benchmarks.qa_frontier.collect --budget 100
python -m benchmarks.qa_frontier.report
python -m benchmarks.qa_frontier.replay
```

The last command replays the kept replies to compare ways of finding the best
pipelines within a budget of calls: trying every pipeline in order, trying them
at random, and `run_closed_loop`. The closed loop currently finds the best
pipeline in about as many calls as random picking.

## Does the best setup stay the best

[`benchmarks/drift`](https://github.com/frontierkit/frontierkit/tree/main/benchmarks/drift)
asks the same thirty questions of the same free models every day and reports
whether the setup chosen on the first day still holds. Hosted models change
without notice, and this measures how often that changes the choice.

```bash
python -m benchmarks.drift.collect
python -m benchmarks.drift.report
```

For each day the report prints the best setup, its quality as measured and
corrected for having been picked from many, how many setups are too close to
it to call, and the failed calls per model. It needs a sample a day over
several weeks, so no results are shown here.

## Search on declared numbers

The rest of this page is about a different question: how well the search
strategies recover the best pipelines from declared cost, latency and
reliability alone, before anything is run. Nothing here measures answers.

### Setup

The search space is a five-step pipeline (retrieval, generation, verification,
refinement, postprocessing) with four components per step, so 4**5 = 1,024
pipelines. Its exact frontier holds 27 of them.

Strategies compared: `HeuristicSynthesizer` (greedy), `RandomSynthesizer`,
`BeamSynthesizer` (a beam on one weighted score),
`ParetoPartialBeamSynthesizer` and `ProgressiveParetoBeamSynthesizer`.

Each strategy may evaluate a limited number of candidates: 20, 40, 80, 120 or
180. Beam searches were also run at widths 3, 5, 7, 10 and 15.

Two figures are reported:

- **Coverage**: the share of the exact frontier the strategy recovered.
- **Hypervolume loss**: how much of the space covered by the exact frontier the
  recovered one misses. Zero is a full recovery.

### Measured tables

The tables below are generated, not hand-written. They were measured by
`scripts/regenerate_benchmarks.py` on version 0.6.0 (Python 3.10.4). Random
baselines use seeds 42 to 46; hypervolume estimates use 200,000 Monte Carlo
samples with seed 42. Re-run them with:

```bash
python scripts/regenerate_benchmarks.py
```

#### Coverage by evaluation budget (beam width 7, 5-stage demo registry)

Coverage of the exact Pareto frontier under a bounded evaluation budget; `Random` cells are mean +/- std over 5 seeds (42-46).

| Algorithm | 20 | 40 | 80 | 120 | 180 |
|---|---|---|---|---|---|
| Greedy | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| Random | 0.04 +/- 0.03 | 0.11 +/- 0.04 | 0.20 +/- 0.06 | 0.31 +/- 0.03 | 0.39 +/- 0.11 |
| ScalarBeam | 0.15 | 0.19 | 0.19 | 0.19 | 0.19 |
| ParetoPartialBeam | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| ProgressiveParetoBeam | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 |

#### Mean hypervolume loss by evaluation budget (beam width 7)

Monte Carlo hypervolume loss against the exact frontier (200000 samples, seed 42); `Random` cells are means over 5 seeds.

| Algorithm | 20 | 40 | 80 | 120 | 180 |
|---|---|---|---|---|---|
| Greedy | 115.5990 | 115.5990 | 115.5990 | 115.5990 | 115.5990 |
| Random | 43.2480 | 28.4835 | 20.1942 | 18.9936 | 16.7583 |
| ScalarBeam | 75.7605 | 68.4000 | 68.4000 | 68.4000 | 68.4000 |
| ParetoPartialBeam | 6.9060 | 6.9060 | 6.9060 | 6.9060 | 6.9060 |
| ProgressiveParetoBeam | 38.2215 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Coverage by beam width (evaluation budget 120)

Coverage of the exact Pareto frontier against a fixed evaluation budget.

| Algorithm | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|
| ScalarBeam | 0.19 | 0.19 | 0.19 | 0.26 | 0.26 |
| ParetoPartialBeam | 0.22 | 0.26 | 0.33 | 0.63 | 0.85 |
| ProgressiveParetoBeam | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |

#### Mean hypervolume loss by beam width (evaluation budget 120)

Monte Carlo hypervolume loss against the exact frontier (200000 samples, seed 42).

| Algorithm | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|
| ScalarBeam | 83.9265 | 83.9265 | 68.4000 | 68.4000 | 68.4000 |
| ParetoPartialBeam | 23.2560 | 13.2660 | 6.9060 | 0.1005 | 0.0135 |
| ProgressiveParetoBeam | 1.9695 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### What the tables show

- A beam on one weighted score prunes too early. It drops partial pipelines
  that would have ended on the frontier, and no later step can bring them back.
- Keeping the partial pipelines nothing beats at each step recovers more, and
  always keeping the cheapest, fastest and most reliable ones protects the
  corners of the frontier.
- Widening the beam as the pipeline grows recovers the whole frontier from a
  budget of 40 candidates at width 7, and from width 5 at a budget of 120. The
  plain beam stays at 19% in both cases.
- Random search improves slowly with budget and reaches 39% at 180 candidates.

These results concern declared numbers. Which pipeline gives the best answers
has to be measured, which is what the first two benchmarks do.
