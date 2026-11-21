# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **The website has a new identity.** A new logo: three setups nothing beats, joined by the frontier, with one beaten setup below. The home page (`overrides/home.html`) is rebuilt around a frontier chart that moves continuously, the 30-question results drawn as range bars with a verdict for each setup, and the same content as the README. Motion stops for visitors who prefer reduced motion
- The website has a light theme as well as the dark one, on the home page and the docs pages, with a switch in the top bar. The choice is remembered and carries over between the home page and the docs
- The "How it works" page is rewritten for the library as it is today, with a flowchart from your questions and options through execution, scoring and statistics to a decision that can be checked again. It described the earlier architecture-search project
- The Benchmarks page now leads with the two benchmarks on real models, including the 30-question results with their ranges, and keeps the search-strategy tables as a separate section
- **The changelog is a page of the website.** It shows this file and follows it on every build, and the site is rebuilt whenever this file changes. Links to the changelog from the README, the home page and the package metadata go there
- The contributing guide, the citation title, the package description and keywords, and a few docstrings describe the project as it is today
- README rewritten around what sets the library apart: a headline table from the 30-question benchmark with a likely range for every score and a verdict against the best setup, a "What it does" section covering margins of error, the answer log, the decision record, `verify` and named failures, two commands that run without an API key, and a snippet that shows the paired verdict and saving a decision

## [0.13.0] - 2026-11-21

### Changed
- **The library is now called FrontierKit.** Install it with `pip install frontierkit` and import `frontierkit` where you imported `chatcortex`; nothing else about the API changed. The commands are now `frontierkit` and `frontierkit-registry`. The repository is at https://github.com/frontierkit/frontierkit and the website at https://frontierkit.github.io/frontierkit/. Releases up to 0.12.0 were published as `chatcortex`, and the entries below keep that name
- `chatcortex` 0.13.0 on PyPI is a thin package that installs `frontierkit`, keeps `import chatcortex` and both old commands working, and warns that the name has changed. It will get no further releases
- A decision record stores the library version under `frontierkit`; records written by 0.12.0 still load

### Added
- **Check that a choice still holds.** `verify_decision(decision, pipelines, executor, eval_set)` runs the chosen pipeline again on the same examples and compares the fresh run with the record. It answers `valid`, `quality_dropped`, `slower`, `unavailable` (the service refused, timed out or lost the model) or `not_comparable` (other examples, or a step that now runs on something else). Examples are asked one at a time and the check stops early once the matter is settled; a `valid` result says whether the change was shown to be within the tolerance or merely not disproved, and how big a drop could have gone unnoticed
- **`frontierkit verify decision.json --setup compare.py`** does the same from the command line, with a different exit code for each answer, and appends every check to `decision.history.jsonl`. Made for a scheduled job
- `PipelineEvaluation.latencies` holds the recorded time of each example, and a decision record now keeps the chosen pipeline's score and time on every example
- `examples/verify_decision.py` records a choice and checks it three times, offline

## [0.12.0] - 2026-11-21

### Added
- **Answer log.** `LoggedBackend(backend, "answers.jsonl")` writes every step of a run to an append-only file: what was asked, what answered it (endpoint, model and request options, or a version you give your own function), when, and what came back, failures included. A step already answered is read back instead of being called again, so a repeated comparison makes no model calls and adding one pipeline only runs its new steps. `label` keeps separate samples of the same step, for example one per day
- **Best of many, reported honestly.** `best_of({name: scores})` returns the winner with its score as measured, a corrected score that takes out the luck of having been picked from many, a range that stays valid after the pick, and the contenders it cannot be told apart from
- **Decision record.** `record_decision(evaluations, eval_set, backend=backend).save("decision.json")` writes down which pipeline was chosen, what each step ran on, the range of every score, each pipeline compared with the chosen one, the failures by kind, and the steps all pipelines shared. `load_decision` reads it back
- `OpenAIBackend.identity()` and `RoutedBackend.identity()` say what a component runs on, without the key or any header
- `examples/pick_models.py` shows the likely range of every quality, and says "too close to call" with the list of setups it cannot separate, where it used to name a single winner
- **Daily benchmark** in `benchmarks/drift/`: asks the same questions of the same free models each day, and reports whether the first day's best setup still holds
- **Failures say what went wrong.** A failed step is recorded with a kind: `rate_limited`, `timeout`, `unavailable`, `not_found` (a retired or misspelt model), `auth`, `empty_output`, `step_failed` or `error`. `PipelineEvaluation.errors` gives the kind for each example, `error_counts` totals them, and `quality_when_answered` is the mean score over the runs that finished, so an outage is no longer indistinguishable from a wrong answer. `ExecutionResult.error`, `ProbeResult.error` and the telemetry records carry the same kind; `StepResult(..., error=...)` lets a step name its own; `classify_error(exc)` maps any exception to a kind

### Changed
- When a backend raises, the executor records how long the call took. It recorded 0 ms before, which made a setup that timed out look fast

## [0.11.0] - 2026-11-21

### Added
- `PipelineEvaluation.scores` and `PipelineEvaluation.outputs` keep the score and the returned output of every example, in eval-set order (0 and `None` for a failed run), so a run can be inspected example by example or graded again with another scorer without new calls. `to_dict()` is unchanged
- Scorers for pipelines that extract triples: `triple_precision`, `triple_recall` and `triple_f1` compare a collection of extracted triples with a reference collection. A triple matches only when every field matches after normalisation, duplicates count once, and extra fields such as a modality make the match stricter
- **Figures over the whole eval set**: `EvalSet(..., metrics={name: metric(outputs, expecteds)})` computes each metric once per pipeline over all examples and reports it in `PipelineEvaluation.metrics` (and in `to_dict()` when any are defined). Pipelines are still ranked by `quality`. Ready-made: `pooled_triple_precision`, `pooled_triple_recall` and `pooled_triple_f1` count matches over all examples together, and `label_macro_f1` averages the F1 of each label held in one field of a triple, over extraction and labelling together or, with `matched_only=True`, over labelling alone
- **Margins of error**, in the new `chatcortex.stats`: `mean_interval(scores)` gives the range a pipeline's quality plausibly lies in; `paired_difference(first, second)` compares two pipelines example by example and returns a verdict that is "too close to call" when the range includes zero; `metric_interval(outputs, expecteds, metric)` does the same for a whole-set metric by resampling examples. The default `"betting"` method is valid for any number of examples and under repeated checking (Waudby-Smith and Ramdas, 2023); `"bootstrap"` is narrower but approximate. No new dependency

### Changed
- Text normalisation in the scorers is cached, which makes resampling a triple metric several times faster. Results are unchanged

## [0.10.2] - 2026-11-19

### Added
- A project website at https://siddharth1012.github.io/chatCortex/, built from `docs/` with MkDocs: a home page, the get-started guide, a new "Using ChatCortex from Python" page (`docs/USAGE.md`), the architecture notes and the benchmarks. A `docs` extra installs the site tooling
- The website has a new dark design: a custom home page (`overrides/home.html`) with measured results, a contributions section and restrained motion that is off for visitors who prefer reduced motion, and matching styling for the docs pages (`docs/stylesheets/site.css`)

### Changed
- Package description and keywords on PyPI now say what the library does today; the homepage link points to the website, and README links are absolute so they work on PyPI
- README rewritten around what a newcomer can do in five minutes, with a measured result up front; the per-feature detail moved to `docs/USAGE.md`

## [0.10.1] - 2026-11-19

### Added
- **Guide: pick the best models for your questions, for free** (`docs/GUIDE.md`) with a runnable script, `examples/pick_models.py`: it tries every combination of prompt style, context and free model on a CSV of your questions, scores the answers, and recommends a setup. Calls are spaced per provider for free-tier limits; models that fail a test call are skipped with the vendor's error and the model names the key can use
- `enumerate_pipelines(registry, task, limit, seed)` lists every pipeline a task allows (a reproducible sample above `limit`), and `count_pipelines` counts them; the closed loop's exploration now uses it
- Ready-made scorers for `EvalSet`: `exact_match`, `contains_answer` and `token_f1` (SQuAD-style word overlap), each accepting one reference answer or a list

## [0.10.0] - 2026-11-19

### Added
- **Measured quality as a search objective** — `ComponentMetadata.quality_score` (optional, `None` until measured) holds an estimated answer quality in `[0, 1]`; `AgentGraph.aggregate_quality()` multiplies it across a pipeline, counting unmeasured components as 1.0; `ArchitectureCandidate.total_quality` carries it, and Pareto dominance now maximises it alongside cost, latency and reliability. With nothing measured every candidate has quality 1.0, so synthesis results are unchanged (the benchmark tables regenerate identically)
- `calibrate_quality(registry, evaluations, prior_strength, prior_quality=1.0)` estimates each component's quality from scored pipeline runs, crediting each of a pipeline's `k` components with `quality ** (1/k)` and weighing by run count against an optimistic prior
- `ExhaustiveSynthesizer(registry, keep_ties=True)` and `ParetoSet(keep_ties=True)` keep every metric-identical frontier member instead of one of them
- A measured `quality_score` is saved and loaded with registry files; files without it are unchanged
- `run_closed_loop(..., explore=N)` also scores, each round, the `N` untried pipelines with the highest predicted quality (then lowest predicted latency), so the loop keeps looking after its frontier is fully scored. Requires an eval set; off by default. `LoopIteration.explored_candidates` reports how many were added
- `benchmarks/qa_frontier/` — a question-answering benchmark over 30 pipelines (3 context options × free models × 2 postprocess steps), scored with the SQuAD token F1. `collect` gathers model replies into a resumable response cache with per-provider spacing for free-tier limits; `report` scores every pipeline offline and prints the true quality frontier; `replay` compares how much of that frontier grid search, random search and ChatCortex's closed loop find within a call budget, without new calls. Ships with 30 questions from the SQuAD v1.1 development set (CC BY-SA 4.0, attributed) and six sample questions; any SQuAD-format file can be used
- **OpenAI-compatible endpoints** — `OpenAIBackend(base_url=..., api_key=...)` points the backend at any vendor or local server that accepts chat-completions requests, with no new dependency. `request_options` adds keyword arguments such as `max_tokens` or `temperature` to every request
- **`RoutedBackend({component name: backend}, default=...)`** sends each component to its own backend, so one search can mix vendors and local functions; `unrouted(registry)` lists components without a route, and running one raises `BackendConfigurationError`
- `examples/compare_endpoints.py` — scores providers on the same three questions and reports measured quality and latency; reads `SAIA_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY`, and prints a setup card when none is set. Every default model is free; paid models (the Claude row, or an OpenRouter model without `:free`) run only with `ALLOW_PAID_MODELS=1` and report measured cost per question from token usage

### Changed
- `run_closed_loop(..., eval_set=...)` now steers the search with measured quality: after each round it calls `calibrate_quality` on every pipeline scored so far (prior strength `prior_strength`, or 1 when unset). A pipeline scored in an earlier round is not run again; its evaluation is reused, `executed_candidates` counts only newly run pipelines, and `frontier_objectives` include `quality`. Runs without an eval set behave as before

## [0.9.0] - 2026-11-18

### Added
- **Cost from token usage** — `OpenAIBackend(prices={model: TokenPrice(input_per_million, output_per_million)})` costs each call from the `prompt_tokens` and `completion_tokens` the API reports. A model without a price, or a response without usage, falls back to the component's declared `cost_per_call`; a call that raises records zero cost, as before
- **Scoring pipelines on examples** — `EvalSet(examples, scorer)` pairs example inputs with a `scorer(output, expected)` returning a bool or a number in `[0, 1]`. `evaluate_pipelines(candidates, executor, eval_set)` runs each pipeline on every example and returns a `PipelineEvaluation` per pipeline (measured `quality`, `success_rate`, mean `cost` and `latency_ms`; a run that fails part-way scores 0). `quality_frontier(evaluations)` keeps those not beaten on measured cost, latency and quality. Scoring requires an executor with a backend
- `run_closed_loop(..., eval_set=...)` runs every selected pipeline on every example, feeds all of those runs to calibration, and records per-pipeline results in `LoopIteration.evaluations` (also in `to_dict()`)
- `examples/scored_pipeline.py` — the callable-pipeline components scored on four questions: measured quality 1.00, 0.00 and 0.00 for pipelines declared 0.91, 0.85 and 0.71 reliable (no API keys, runs in CI)
- **Components as Python callables** — `CallableBackend({name: function})` runs each graph node as an ordinary function that receives the previous step's output. Latency is measured; a function that raises or returns `StepResult(success=False)` is a failed step; cost is the declared value unless `StepResult(cost=...)` reports one, and zero for a function that raises. `CallableBackend.missing(registry)` lists components without an implementation, and running one raises `BackendConfigurationError` instead of being recorded as a component failure
- `AgentExecutor.execute(graph, task_input=...)` hands an input to the first step, and `ExecutionResult` gains `output` (the last step's return value) and `success`
- `AgentExecutor`, `CallableBackend`, `StepResult` and `BackendConfigurationError` are exported from the top-level package
- `examples/callable_pipeline.py` — five plain functions, three pipelines on the frontier, each run on a real question (no API keys, runs in CI)
- **Evidence-weighted calibration** — `calibrate(..., prior_strength=n0)` treats the declared value as `n0` pseudo-observations, so a component with `n` records gets weight `n / (n + n0)` on its empirical mean instead of a fixed `alpha`. `run_closed_loop(..., prior_strength=n0)` recalibrates the declared registry from all traces collected so far each round. Defaults are unchanged: without `prior_strength` both functions behave as before

### Changed
- `examples/closed_loop.py` runs 16 iterations with `prior_strength=20` and adds a 20-seed comparison of mean reliability error against the hidden ground truth: declared registry 0.030; fixed `alpha=0.4` 0.066 (beats declared in 0/20 runs); `prior_strength=20` 0.024 (17/20). Measured separately at 4 iterations, `prior_strength=20` averages 0.033 and does not yet beat the declared values

## [0.8.0] - 2026-11-18

### Added
- **Hidden ground truth for the simulator** — `AgentExecutor(ground_truth=registry)` draws simulated success, latency and cost from a separate registry of true values, looked up by component name, instead of from the declared metadata. The synthesizer and calibration never read it, so a closed loop can be tested for convergence on values it was not told. Components missing from the ground truth behave as declared; combining `ground_truth` with `backend` raises `ValueError`

### Changed
- `examples/closed_loop.py` now starts from a registry that is wrong about four components and reports declared vs calibrated vs true values. Measured (4 iterations, alpha=0.4, seed 42): `retrieval_speedster` reliability 0.78 → 0.542 (truth 0.55), `retrieval_budget` 0.82 → 0.908 (0.95), `generation_balanced` latency 180 → 301.9 ms (320), `verification_reliable` cost 0.006 → 0.0086 (0.009). Mean absolute error over all components falls for latency (11.67 → 1.51 ms) and cost (2.5e-4 → 3.2e-5) and **rises for reliability (0.030 → 0.041)**: the fixed-alpha blend adds sampling noise to correctly declared components
- README: the closed-loop demo figures are replaced by the hidden-truth results above

### Fixed
- `CITATION.cff` reported version 0.5.0; it now tracks the current release

### Removed
- Python 3.9 support. 3.9 is end-of-life and no longer available on GitHub's current Ubuntu runners; `requires-python` is now `>=3.10` and CI tests 3.10–3.12

## [0.7.0] - 2026-10-24

### Added
- **The closed loop** — `chatcortex.loop.run_closed_loop(registry, task, synthesizer, executor, iterations, alpha)` drives synthesize → execute → calibrate → re-synthesize as a library API, returning per-iteration `LoopIteration` records (frontier size and objective ranges, probes executed, calibrated registry snapshot; JSON-serializable and deterministic for a seed). The synthesizer is passed as a class or factory — pre-built instances are rejected, because a loop calibrated against a stale registry is not a loop that learns. Executor-composition with an `ExecutionProtocol` backend is covered by tests: measured latency/cost (not declared values) drive calibration
- `examples/closed_loop.py` — the flagship demo (4 iterations, no API keys, runs in CI). Measured run (alpha=0.4, seed 42): frontier breathes 15 → 18 → 14 → 16 architectures as calibrated metadata reshapes the search, and reliability moves from declared priors to empirical values — `retrieval_speedster` 0.78 → 0.53, `retrieval_budget` 0.82 → 0.93, `generation_reliable` 0.98 → 0.997, `verification_balanced` 0.92 → 0.94
- `examples/closed_loop_real.py` — the same loop against real OpenAI chat-completions backends (`chatcortex[real]` + `OPENAI_API_KEY`); without a key it prints a setup card and exits 0 so secretless CI stays green
- **Registry packs** — `save_registry(registry, path)` writes the loader's document shape (no new schema fields) and proves the round-trip by reloading; `validate_registry_file(path)` returns a structured `RegistryReport` (component count, per-row field issues in loader style, duplicate names) without raising, collecting every broken row in one pass. New console script: `chatcortex-registry validate <file>` (exit 0 only when the pack fully validates)
- README: closed-loop headline grounded in the measured demo numbers, a Registry packs section with copy-pasteable snippets, quickstart wiring for both closed-loop examples; GitHub repo description and topics updated to match

### Verification
- `scripts/regenerate_benchmarks.py` re-run on the shipped core: every coverage, hypervolume-loss, and regret row in `docs/benchmark_results.json` reproduces byte-for-byte against 0.6.0 (`git diff` empty) — no search-behavior regression
- Full pytest suite green (143 tests), ruff clean, all four examples run keyless end-to-end plus the keyless real-backend path

## [0.6.0] - 2026-10-24

### Added
- `AgentGraph.edges()` public edge accessor and `AgentGraph.chain_append()` chain builder; the regression gates and the synthesis engine now use the public API instead of `graph._graph`
- `load_registry(path)` — JSON registry loader in `chatcortex.registry.loader`; validates every row and names the failing field on error. `CapabilityRegistry.to_dict()` emits the same document shape for round-trip serialization
- Optional real-execution backend protocol (`ExecutionProtocol` with nested `ProbeResult`, plus a flat `ProbeResult` alias) plus an `openai`-SDK chat-completions adapter behind a new `chatcortex[real]` extra; the zero-dependency core is unchanged
- `AgentExecutor(..., backend=...)` — with a backend set, telemetry records the backend's measured latency/cost/success and the chain stops on the first failure exactly as before
- Trace calibration (`chatcortex.telemetry.calibration.calibrate`) — blends empirical success/latency/cost from TelemetryLogger records into metadata with a documented alpha rule; components without traces keep their priors; a new `CapabilityRegistry.update()` supports the immutable replacement
- Example `examples/registry_from_json.py` runs the full loop (JSON registry → synthesis → execution → calibration) with no API keys

### Changed
- `ParetoPartialBeamSynthesizer` and `ProgressiveParetoBeamSynthesizer` stage truncation replaced with one deterministic Pareto-aware crowding-distance policy in objective space (in `chatcortex.synthesis.engine`), replacing the former cost-first lexicographic sort and the ad-hoc cost-diversity prune. The policy keeps the nondominated partials only, always retains the cost, latency, and reliability extremes, and fills remaining slots by crowding distance. Expected effect, measured on the demo 3-stage task: the reliability corner of the exact frontier `(0.009, 540ms, 0.7787)` — previously lost by both synthesizers at nearly every beam width — is recovered from partial-beam width 4 up (full 6/6 frontier from width 6), and the partial beam's regret at the pinned width drops to zero. Beam widths smaller than the number of nondominated partials may recover a different frontier subset than before (per-stage pruning is stricter about mid-distance partials).
- `ParetoSet` is deterministic: metric-identical candidates are kept once and iteration yields members sorted by objective tuple `(cost, latency, reliability)`. Synthesized frontiers no longer contain duplicate objective rows and are returned in sorted order; the regression-gate coverage floors were tightened to the exact measured values (`4/6`, `5/6`, `1.0`, `1.0`) that the duplicate-tolerant floors previously approximated from below
- `hypervolume_loss` reuses `hypervolume_monte_carlo` instead of duplicating its Monte Carlo sampling loop (two independent seeded runs replace the former shared RNG stream); the unused `compute_hypervolume` wrapper and the undeclared-anywhere `numpy` dependency were removed

## [0.5.0] - 2026-07-29

### Added
- `build_demo_registry()` helper for instant end-to-end quickstarts
- `AgentGraph.iter_components()` for public component iteration
- `examples/quickstart.py` comparing beam, progressive, and exhaustive synthesis
- Full pytest suite covering registry, graph, Pareto logic, synthesizers, and execution
- GitHub Actions CI (ruff + pytest on Python 3.9–3.12)
- Issue and PR templates, code of conduct
- Top-level exports for `ParetoPartialBeamSynthesizer`, `ProgressiveParetoBeamSynthesizer`, `SynthesisBudget`, `CapabilityRegistry`, `ComponentMetadata`

### Fixed
- `TaskSpecification.validate()` did not actually raise on empty capability lists
- `RandomSynthesizer` looped forever when called without a budget; now defaults to 200 evaluations

### Changed
- Demo scripts moved from the package to `examples/`
- Removed research-iteration synthesizers (v2, v3, system-score); their logic is superseded by `ProgressiveParetoBeamSynthesizer` and they remain available in git history
- README rewritten with a working, tested quickstart

## [0.4.0] - 2026-03-08

### Added
- `ProgressiveParetoBeamSynthesizer` — depth-aware progressive beam widening
- Multi-objective search backbone

## [0.3.0]

### Added
- Budget-aware synthesis (`SynthesisBudget`), beam search, scaling experiments

## [0.2.0]

### Added
- Telemetry, evaluation harness, base synthesizers, Pareto computation

## [0.1.0]

### Added
- Component metadata, capability registry, task specification, agent graph
