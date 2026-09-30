"""Inspectable class balance and reusable fit-local observation weights."""

from dataclasses import dataclass
from typing import ClassVar
from ..weighting import ClassWeightPolicy
from .contracts import Artifact, StudyResult


@dataclass
class ClassWeightReporter:
    """Describe class balance and optionally supply weights to training.

    Consume by naming this fitted preparation study in Candidate.weights_from.
    per_fold/train recalculates on actual fit rows, including inner search fits.
    overall is descriptive unless its rows exactly match the consuming fit.
    """
    type: str = "per_fold"
    partition: str = "train"
    mode: str = "balanced"
    power: float = 1.0
    class_weights: object = None
    calculator: object = None
    target: str | None = None
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        result = ClassWeightPolicy(self.mode, self.power, self.class_weights, self.calculator, self.target).compute(context)
        return StudyResult("Class balance and fitting weights",
                           [Artifact("table", result.classes, "Class balance")],
                           {"class_balance": result.classes},
                           ["Weights average one. Training consumes them only when this study is selected as weights_from.",
                            "Validation, calibration and test rows remain unweighted."], weights=result)
