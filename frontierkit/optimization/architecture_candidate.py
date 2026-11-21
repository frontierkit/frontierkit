from dataclasses import dataclass

from frontierkit.graph.agent_graph import AgentGraph


@dataclass(frozen=True)
class ArchitectureCandidate:
    """
    Represents a fully constructed agent architecture
    with cached deterministic metrics for optimization
    """

    graph: AgentGraph
    total_cost: float
    total_latency: float
    total_reliability: float
    # Predicted answer quality, from measured component estimates; 1.0
    # when nothing has been measured, which leaves the other three
    # objectives to decide on their own.
    total_quality: float = 1.0

    def metrics(self):
        return {
            "cost": self.total_cost,
            "latency": self.total_latency,
            "reliability": self.total_reliability,
            "quality": self.total_quality,
        }
