"""The ``chatcortex`` command.

``chatcortex verify decision.json --setup compare.py`` runs the pipeline a
decision record chose once more and says whether the choice still holds.
It is made for a scheduled job: the exit code tells the job what happened.

==== ==================================================================
code meaning
==== ==================================================================
0    still valid
1    quality dropped
2    the command could not run (bad arguments, missing or broken files)
3    slower
4    unavailable: the service refused, timed out or lost the model
5    not comparable: other examples, or a step that changed
==== ==================================================================

The setup file is ordinary Python. It defines a function, ``setup`` by
default, that returns what the comparison was made with::

    def setup():
        return {"pipelines": pipelines, "executor": executor, "eval_set": eval_set}

Each check is appended to ``<decision>.history.jsonl`` beside the record,
so the file shows when a choice stopped holding.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .decision import load_decision
from .verify import EXIT_CODES, Verification, verify_decision

USAGE_ERROR = 2


class SetupError(Exception):
    """The setup file could not be used."""


def load_setup(target: str) -> Dict[str, Any]:
    """Run ``path/to/file.py[:function]`` and return what it gives.

    Raises:
        SetupError: If the file or function is missing, raises, or does
            not return ``pipelines``, ``executor`` and ``eval_set``.
    """
    path_text, _, function_name = target.partition(":")
    if len(path_text) == 1 and function_name[:1] in ("\\", "/"):
        # A Windows drive letter, not a function name: C:\dir\file.py[:function]
        rest, _, function_name = function_name.partition(":")
        path_text = f"{path_text}:{rest}"
    function_name = function_name or "setup"
    path = Path(path_text)
    if not path.is_file():
        raise SetupError(f"setup file not found: {path}")

    spec = importlib.util.spec_from_file_location(f"chatcortex_setup_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SetupError(f"{path} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    folder = str(path.resolve().parent)
    sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(module)
        function: Optional[Callable[[], Any]] = getattr(module, function_name, None)
        if not callable(function):
            raise SetupError(f"{path} does not define a function named {function_name}")
        given = function()
    except SetupError:
        raise
    except Exception as exc:
        raise SetupError(f"{path} failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if folder in sys.path:
            sys.path.remove(folder)

    if not isinstance(given, dict):
        raise SetupError(
            f"{function_name}() must return a dict with pipelines, executor and eval_set"
        )
    missing = [key for key in ("pipelines", "executor", "eval_set") if key not in given]
    if missing:
        raise SetupError(f"{function_name}() did not return: {', '.join(missing)}")
    return given


def history_path(decision_path: Path) -> Path:
    return decision_path.with_name(decision_path.stem + ".history.jsonl")


def _verify(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    decision_path = Path(args.decision)
    try:
        decision = load_decision(decision_path)
    except FileNotFoundError:
        out(f"error: decision record not found: {decision_path}")
        return USAGE_ERROR
    except (ValueError, json.JSONDecodeError) as exc:
        out(f"error: {decision_path} is not a decision record ({exc})")
        return USAGE_ERROR
    try:
        given = load_setup(args.setup)
        check: Verification = verify_decision(
            decision,
            given["pipelines"],
            given["executor"],
            given["eval_set"],
            tolerance=args.tolerance,
            confidence=args.confidence,
            slower_factor=args.slower_factor,
            max_latency_ms=args.max_latency_ms,
        )
    except (SetupError, ValueError) as exc:
        out(f"error: {exc}")
        return USAGE_ERROR

    if not args.no_history:
        with history_path(decision_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(check.to_dict(), ensure_ascii=False) + "\n")

    if args.json:
        out(json.dumps(check.to_dict(), indent=2, ensure_ascii=False))
    else:
        out(f"{check.status}: {' / '.join(check.components)}")
        out(check.message)
        if check.errors and check.status != "unavailable":
            kinds = ", ".join(f"{count} {kind}" for kind, count in sorted(check.errors.items()))
            out(f"Failed runs, counted as 0: {kinds}.")
    return check.exit_code


def main(argv: Optional[List[str]] = None, out: Callable[[str], None] = print) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="chatcortex", description="Check that a recorded model choice still holds."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify",
        help="run a recorded choice again and say whether it still holds",
        description="Exit codes: "
        + ", ".join(f"{code} {status}" for status, code in EXIT_CODES.items())
        + f", {USAGE_ERROR} could not run.",
    )
    verify.add_argument("decision", help="decision record written by record_decision(...).save()")
    verify.add_argument(
        "--setup",
        required=True,
        metavar="FILE[:FUNCTION]",
        help="Python file whose function (default: setup) returns "
        "{'pipelines': ..., 'executor': ..., 'eval_set': ...}",
    )
    verify.add_argument("--tolerance", type=float, default=0.05,
                        help="accepted drop in mean score (default 0.05)")
    verify.add_argument("--confidence", type=float, default=0.99,
                        help="how sure to be before reporting a change (default 0.99)")
    verify.add_argument("--slower-factor", type=float, default=2.0,
                        help="slow means this many times the recorded median (default 2)")
    verify.add_argument("--max-latency-ms", type=float, default=None,
                        help="also slow above this time per example")
    verify.add_argument("--json", action="store_true", help="print the full result as JSON")
    verify.add_argument("--no-history", action="store_true",
                        help="do not append the result to the history file")
    args = parser.parse_args(argv)
    return _verify(args, out)


if __name__ == "__main__":
    raise SystemExit(main())
