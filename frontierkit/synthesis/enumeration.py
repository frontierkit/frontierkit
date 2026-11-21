"""List every pipeline a task allows.

Synthesizers return a frontier: the pipelines that look best on paper.
When you are about to measure pipelines anyway and the space is small,
it is simpler to try all of them. :func:`enumerate_pipelines` lists every
chain the registry can build for a task, one component per required
capability, within the task's cost, latency and privacy limits.
"""

import itertools
import math
import random
from typing import List, Optional

from frontierkit.graph.agent_graph import AgentGraph
from frontierkit.graph.node_ids import node_id_for
from frontierkit.optimization.architecture_candidate import ArchitectureCandidate
from frontierkit.registry.capability_registry import CapabilityRegistry
from frontierkit.synthesis.task_specification import TaskSpecification

DEFAULT_LIMIT = 20_000


def count_pipelines(registry: CapabilityRegistry, task: TaskSpecification) -> int:
    """How many chains the task allows, before cost and latency limits."""
    return math.prod(
        len(registry.get_by_capability(capability, task.privacy_constraint))
        for capability in task.required_capabilities
    )


def enumerate_pipelines(
    registry: CapabilityRegistry,
    task: TaskSpecification,
    limit: Optional[int] = DEFAULT_LIMIT,
    seed: int = 0,
) -> List[ArchitectureCandidate]:
    """Every pipeline for ``task`` that meets its limits.

    Args:
        registry: Components to choose from.
        task: Required capabilities, in order, and hard limits.
        limit: Above this many combinations, return a random sample of
            this size instead of all of them. ``None`` lists everything.
        seed: Seed for that sample, so it is reproducible.

    Returns:
        Candidates in a stable order: registration order within each
        stage when all are listed.
    """
    task.validate()
    options = [
        registry.get_by_capability(capability, task.privacy_constraint)
        for capability in task.required_capabilities
    ]
    size = math.prod(len(o) for o in options)
    if limit is None or size <= limit:
        combos = itertools.product(*options)
    else:
        rng = random.Random(seed)
        combos = (tuple(rng.choice(o) for o in options) for _ in range(limit))

    candidates = []
    for combo in combos:
        graph = AgentGraph()
        for stage, metadata in enumerate(combo):
            graph.chain_append(node_id_for(metadata.name, stage), metadata)
        candidate = ArchitectureCandidate(
            graph=graph,
            total_cost=graph.total_cost(),
            total_latency=graph.total_latency(),
            total_reliability=graph.aggregate_reliability(),
            total_quality=graph.aggregate_quality(),
        )
        if task.max_cost is not None and candidate.total_cost > task.max_cost:
            continue
        if task.max_latency is not None and candidate.total_latency > task.max_latency:
            continue
        candidates.append(candidate)
    return candidates
