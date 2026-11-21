"""
FrontierKit

Find out which model, prompt and pipeline answer your own questions
best, and how sure you can be.

- Describe your options as components and the steps a pipeline needs.
- Run every pipeline on your examples and score the answers.
- Get a range for every score, and a verdict on whether two pipelines
  can be told apart.
- Keep the pipelines nothing beats on quality, speed and cost.
- Record the choice, and check later that it still holds.
"""

from .decision import Decision, load_decision, record_decision
from .evaluation.corpus_metrics import (
    label_macro_f1,
    pooled_triple_f1,
    pooled_triple_precision,
    pooled_triple_recall,
)
from .evaluation.eval_set import (
    EvalSet,
    Example,
    PipelineEvaluation,
    evaluate_pipelines,
    quality_frontier,
)
from .evaluation.scorers import (
    contains_answer,
    exact_match,
    token_f1,
    triple_f1,
    triple_precision,
    triple_recall,
)
from .execution.callable_backend import CallableBackend, StepResult
from .execution.errors import classify_error
from .execution.executor import AgentExecutor, BackendConfigurationError
from .execution.routed_backend import RoutedBackend
from .graph.agent_graph import AgentGraph
from .loop import LoopIteration, run_closed_loop
from .registry.capability_registry import CapabilityRegistry
from .registry.demo_components import build_demo_registry
from .registry.loader import load_registry
from .registry.metadata import ComponentMetadata
from .registry.package import RegistryReport, save_registry, validate_registry_file
from .stats import (
    BestOf,
    Interval,
    PairedDifference,
    best_of,
    mean_interval,
    metric_interval,
    paired_difference,
)
from .store import AnswerLog, LoggedBackend
from .synthesis.beam_synthesizer import BeamSynthesizer
from .synthesis.budget import SynthesisBudget
from .synthesis.enumeration import count_pipelines, enumerate_pipelines
from .synthesis.exhaustive_synthesizer import ExhaustiveSynthesizer
from .synthesis.heuristic_synthesizer import HeuristicSynthesizer
from .synthesis.pareto_partial_beam_synthesizer import ParetoPartialBeamSynthesizer
from .synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from .synthesis.random_synthesizer import RandomSynthesizer
from .synthesis.task_specification import TaskSpecification
from .verify import Verification, verify_decision

__all__ = [
    "TaskSpecification",
    "HeuristicSynthesizer",
    "RandomSynthesizer",
    "BeamSynthesizer",
    "ExhaustiveSynthesizer",
    "ParetoPartialBeamSynthesizer",
    "ProgressiveParetoBeamSynthesizer",
    "SynthesisBudget",
    "AgentGraph",
    "AgentExecutor",
    "BackendConfigurationError",
    "CallableBackend",
    "RoutedBackend",
    "StepResult",
    "classify_error",
    "EvalSet",
    "Example",
    "PipelineEvaluation",
    "evaluate_pipelines",
    "quality_frontier",
    "exact_match",
    "contains_answer",
    "token_f1",
    "triple_precision",
    "triple_recall",
    "triple_f1",
    "pooled_triple_precision",
    "pooled_triple_recall",
    "pooled_triple_f1",
    "label_macro_f1",
    "Interval",
    "PairedDifference",
    "mean_interval",
    "paired_difference",
    "metric_interval",
    "BestOf",
    "best_of",
    "AnswerLog",
    "LoggedBackend",
    "Decision",
    "record_decision",
    "load_decision",
    "Verification",
    "verify_decision",
    "enumerate_pipelines",
    "count_pipelines",
    "CapabilityRegistry",
    "LoopIteration",
    "ComponentMetadata",
    "build_demo_registry",
    "load_registry",
    "run_closed_loop",
    "save_registry",
    "validate_registry_file",
    "RegistryReport",
]

__version__ = "0.13.0"
