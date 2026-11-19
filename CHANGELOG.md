# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `benchmarks/qa_frontier/` — a question-answering benchmark over 30 pipelines (3 context options × free models × 2 postprocess steps), scored with the SQuAD token F1. `collect` gathers model replies into a resumable response cache with per-provider spacing for free-tier limits; `report` scores every pipeline offline and prints the true quality frontier. Ships with 30 questions from the SQuAD v1.1 development set (CC BY-SA 4.0, attributed) and six sample questions; any SQuAD-format file can be used
- **OpenAI-compatible endpoints** — `OpenAIBackend(base_url=..., api_key=...)` points the backend at any vendor or local server that accepts chat-completions requests, with no new dependency. `request_options` adds keyword arguments such as `max_tokens` or `temperature` to every request
- **`RoutedBackend({component name: backend}, default=...)`** sends each component to its own backend, so one search can mix vendors and local functions; `unrouted(registry)` lists components without a route, and running one raises `BackendConfigurationError`
- `examples/compare_endpoints.py` — scores providers on the same three questions and reports measured quality and latency; reads `SAIA_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY`, and prints a setup card when none is set. Every default model is free; paid models (the Claude row, or an OpenRouter model without `:free`) run only with `ALLOW_PAID_MODELS=1` and report measured cost per question from token usage

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
