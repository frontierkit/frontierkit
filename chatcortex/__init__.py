"""
ChatCortex

Framework for automated synthesis and optimization of AI agent
architectures under multi-objective constraints.

Core capabilities include:

- Task specification
- Agent architecture synthesis
- Multi-objective optimization
- Pareto frontier analysis
"""

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
    Interval,
    PairedDifference,
    mean_interval,
    metric_interval,
    paired_difference,
)
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

__version__ = "0.10.2"
