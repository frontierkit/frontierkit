from typing import List, Optional

from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.synthesis.base import Synthesizer
from chatcortex.synthesis.budget import SynthesisBudget
from chatcortex.synthesis.engine import ChainSearchEngine
from chatcortex.synthesis.task_specification import TaskSpecification


class ExhaustiveSynthesizer(Synthesizer):
    """
    Exact Pareto frontier via full enumeration

    An untruncated configuration of the shared engine: every stage keeps
    all expansions, so the beam becomes the full Cartesian product of
    candidate components per capability.

    ``keep_ties=True`` keeps every metric-identical frontier member
    instead of one of them; use it in a closed loop with an eval set, so
    look-alike alternatives all get measured.
    """

    def __init__(self, registry, keep_ties: bool = False):
        super().__init__(registry)
        self.keep_ties = keep_ties

    def synthesize(
        self,
        task: TaskSpecification,
        budget: Optional[SynthesisBudget] = None,
    ) -> List[ArchitectureCandidate]:
        return ChainSearchEngine(self.registry, keep_ties=self.keep_ties).run(task, budget)
