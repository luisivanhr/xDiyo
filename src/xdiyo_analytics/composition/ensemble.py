from dataclasses import dataclass, field
from .contracts import OutputRef
from .graph import ModelNode, ReducerNode, PredictionGraphSpec
from .training import TrainingPlan
from .adapter import CompositeModelAdapter


@dataclass(frozen=True)
class ModelEnsemble:
    """Fixed same-target blend. Child adapters own their preprocessing."""
    models: dict
    schema: object
    reducer: object
    output: str = 'predict'
    retain_children: bool = True
    training: TrainingPlan = field(default_factory=TrainingPlan)

    def build(self):
        if 'blend' in self.models:
            raise ValueError('blend is reserved for the ensemble reducer.')
        nodes = {name: value if isinstance(value,ModelNode) else ModelNode(value,{self.output:self.schema}) for name,value in self.models.items()}
        nodes['blend'] = ReducerNode(self.reducer,tuple(OutputRef(n,self.output) for n in self.models),self.schema)
        outputs = {self.output:OutputRef('blend')}
        if self.retain_children:
            outputs.update({f'{name}/{self.output}':OutputRef(name,self.output) for name in self.models})
        return CompositeModelAdapter(PredictionGraphSpec(nodes,outputs),self.training)
