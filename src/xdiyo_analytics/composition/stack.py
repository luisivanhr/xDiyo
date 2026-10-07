from dataclasses import dataclass, field
from .contracts import OutputRef, TargetSpec
from .graph import ModelNode, PredictionGraphSpec
from .training import TrainingPlan
from .adapter import CompositeModelAdapter


@dataclass(frozen=True)
class ModelStack:
    """One learned meta layer, fitted on chronological out-of-fold predictions."""
    base_models: dict
    meta_model: object
    meta_features: object
    schema: object
    training: TrainingPlan
    output: str = 'predict'
    target_spec: TargetSpec = field(default_factory=TargetSpec)

    def build(self):
        if 'meta' in self.base_models or any(not isinstance(n,ModelNode) or n.inputs for n in self.base_models.values()):
            raise ValueError('Stack bases must be named input-free ModelNodes; meta is reserved.')
        refs = tuple(OutputRef(name,output) for name,node in self.base_models.items() for output in node.schemas)
        nodes = dict(self.base_models,meta=ModelNode(self.meta_model,{self.output:self.schema},refs,self.meta_features))
        return CompositeModelAdapter(PredictionGraphSpec(nodes,{self.output:OutputRef('meta',self.output)}),self.training,self.target_spec)
