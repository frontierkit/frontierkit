"""The chatcortex compatibility package: the old name loads the new modules."""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPAT = ROOT / "compat" / "chatcortex"


def _run(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that can see both packages."""
    script = "import sys\nsys.path[:0] = [%r, %r]\n" % (str(COMPAT), str(ROOT))
    return subprocess.run(
        [sys.executable, "-W", "error::ImportWarning", "-c", script + textwrap.dedent(code)],
        capture_output=True, text=True, cwd=str(ROOT.parent),
    )


def test_old_name_serves_the_same_modules_and_objects():
    result = _run(
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import chatcortex
        assert [w.category for w in caught] == [DeprecationWarning], caught
        assert "renamed to frontierkit" in str(caught[0].message)

        import frontierkit
        assert chatcortex.__version__ == frontierkit.__version__
        assert chatcortex.EvalSet is frontierkit.EvalSet
        assert set(frontierkit.__all__) <= set(dir(chatcortex))

        # Regular packages, packages without an __init__, and modules.
        import chatcortex.stats
        import chatcortex.stats.intervals as old_intervals
        from chatcortex.evaluation.eval_set import EvalSet
        from chatcortex.adapters import openai_backend
        from chatcortex.registry.cli import main
        import frontierkit.stats.intervals as new_intervals
        assert chatcortex.stats is frontierkit.stats
        assert old_intervals is new_intervals
        assert EvalSet is frontierkit.EvalSet
        assert isinstance(frontierkit.Example("q"), chatcortex.Example)

        # The real modules keep their own identity.
        assert new_intervals.__spec__.name == "frontierkit.stats.intervals"
        assert frontierkit.stats.__spec__.name == "frontierkit.stats"
        assert new_intervals.__name__ == "frontierkit.stats.intervals"

        try:
            import chatcortex.no_such_module
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("a module that does not exist was imported")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_old_name_still_runs_a_comparison_end_to_end():
    result = _run(
        """
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        from chatcortex import (
            AgentExecutor, CallableBackend, CapabilityRegistry, ComponentMetadata,
            EvalSet, Example, TaskSpecification, enumerate_pipelines,
            evaluate_pipelines, exact_match, record_decision,
        )
        from chatcortex.stats import mean_interval

        registry = CapabilityRegistry()
        registry.register(ComponentMetadata(
            name="echo", component_type="tool", capabilities=["answer"],
            cost_per_call=0.0, avg_latency_ms=1.0, reliability_score=1.0,
            privacy_level="internal",
        ))
        task = TaskSpecification(required_capabilities=["answer"])
        executor = AgentExecutor(backend=CallableBackend({"echo": lambda text: text}))
        eval_set = EvalSet([Example("a", expected="a"), Example("b", expected="c")], exact_match)
        evaluations = evaluate_pipelines(enumerate_pipelines(registry, task), executor, eval_set)
        assert evaluations[0].scores == (1.0, 0.0)
        assert mean_interval(evaluations[0].scores).estimate == 0.5
        # Runs a relative import inside the aliased module.
        decision = record_decision(evaluations, eval_set)
        assert decision.data["frontierkit"]
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_compat_metadata_points_at_the_new_package():
    text = (COMPAT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "chatcortex"' in text
    assert '"frontierkit>=0.13.0"' in text
    assert 'chatcortex = "frontierkit.cli:main"' in text
    assert 'chatcortex-registry = "frontierkit.registry.cli:main"' in text
