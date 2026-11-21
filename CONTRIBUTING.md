# Contributing to FrontierKit

Thank you for your interest in contributing to FrontierKit.

FrontierKit finds out which model, prompt and pipeline answer your own questions best, and says how sure the result is.

## Development Setup

Clone the repository:

```bash
git clone https://github.com/frontierkit/frontierkit
cd frontierkit
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install development dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Try an example that needs no API key:

```bash
python examples/scored_pipeline.py
```

## Code Style

FrontierKit uses:

* **black** for formatting
* **ruff** for linting

Run:

```bash
black .
ruff check frontierkit tests examples benchmarks compat
```

## Pull Requests

When submitting a PR:

1. Create a feature branch.
2. Ensure tests pass.
3. Write clear commit messages.

Example:

```
Add a scorer for extracted triples
Fix the range for a single example
Show margins of error in the model-picking example
```

### Commit message policy

Commit and tag messages must contain no attribution footers or trailers
— no "Generated with ..." lines, no emoji footer, and no `Co-Authored-By`
lines (including any AI tooling). A commit message ends at its body's
last line.

## What Helps Most

FrontierKit welcomes contributions related to:

* scorers and metrics for more kinds of task
* backends for more model vendors and local servers
* statistics for small sets of examples
* search that needs fewer model calls than trying everything
* benchmarks on real models, and results from running the existing ones

## Code of Conduct

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
