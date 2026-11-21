from typing import List

from frontierkit.optimization.architecture_candidate import ArchitectureCandidate
from frontierkit.synthesis.base import Synthesizer
from frontierkit.synthesis.engine import ChainSearchEngine, crowding_truncation
from frontierkit.synthesis.task_specification import TaskSpecification


class ParetoPartialBeamSynthesizer(Synthesizer):
    """
    Beam search that performs multi-objective Pareto pruning
    at each intermediate stage.
    """

    def __init__(self, registry, beam_width: int = 3):
        super().__init__(registry)
        self.beam_width = beam_width

    def synthesize(
        self,
        task: TaskSpecification,
        budget=None,
    ) -> List[ArchitectureCandidate]:
        """Per-stage Pareto filtering plus Pareto-aware crowding truncation
        on intermediate stages; the final stage keeps the whole filtered set."""
        return ChainSearchEngine(self.registry).run(
            task,
            budget,
            truncation=crowding_truncation(lambda stage_idx: self.beam_width),
        )
