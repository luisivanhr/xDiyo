"""Static acyclic prediction graphs with one supported learned meta layer."""
from dataclasses import dataclass
from .contracts import OutputSchema, OutputRef


@dataclass(frozen=True)
class ModelNode:
    model: object
    schemas: dict
    inputs: tuple = ()
    features: object = None
    calibration: object = None
    frozen: bool = False
    artifact_vintage: object = None
    trained_through: object = None
    artifact_id: str | None = None
    weighting: object = None


@dataclass(frozen=True)
class TransformNode:
    transform: object
    inputs: tuple
    schema: OutputSchema


@dataclass(frozen=True)
class ReducerNode:
    reducer: object
    inputs: tuple
    schema: OutputSchema


@dataclass(frozen=True)
class PredictionGraphSpec:
    nodes: dict
    outputs: dict
    version: int = 1

    def order(self):
        if self.version != 1 or not self.nodes or not self.outputs:
            raise ValueError('Graph version 1 requires nodes and explicit public outputs.')
        if any(not isinstance(n, str) or not n or '/' in n for n in self.nodes):
            raise ValueError('Graph names must be nonempty without namespace separator /.')
        ordered, active, depth = [], set(), {}
        def visit(name):
            if name in active:
                raise ValueError('Prediction graph contains a cycle.')
            if name in ordered:
                return
            if name not in self.nodes:
                raise ValueError(f'Unresolved node {name!r}.')
            node = self.nodes[name]
            if not isinstance(node, (ModelNode, TransformNode, ReducerNode)):
                raise TypeError('Unsupported prediction node.')
            active.add(name)
            upstream = []
            for ref in node.inputs:
                if not isinstance(ref, OutputRef):
                    raise TypeError('Graph edges require OutputRef.')
                visit(ref.node)
                if ref.output not in self.schemas(ref.node):
                    raise ValueError('Unresolved output reference.')
                upstream.append(depth[ref.node])
            depth[name] = max(upstream, default=0) + int(isinstance(node, ModelNode))
            if depth[name] > 2:
                raise ValueError('Unsupported learned depth: only bases plus one meta layer are cross-fitted.')
            if isinstance(node, ModelNode) and node.inputs and node.features is None:
                raise ValueError('Meta models need explicit OutputFeatures.')
            if isinstance(node, ModelNode) and (not node.schemas or any(not isinstance(s, OutputSchema) for s in node.schemas.values())):
                raise ValueError('Each model output requires an OutputSchema.')
            if isinstance(node, ReducerNode):
                source_schemas = [self.schemas(r.node)[r.output] for r in node.inputs]
                if not source_schemas or any(s != source_schemas[0] for s in source_schemas):
                    raise ValueError('Reducer edges require identical schemas before fitting.')
                from .reducers import Mean, HardVote, DistributionMixture
                if isinstance(node.reducer, (Mean, DistributionMixture)):
                    Mean(node.reducer.weights).validate_weights(len(node.inputs))
                if isinstance(node.reducer, DistributionMixture) and (source_schemas[0].family != 'categorical_pmf' or node.schema != source_schemas[0]):
                    raise ValueError('Distribution mixtures require a complete common PMF schema.')
                if isinstance(node.reducer, Mean) and (source_schemas[0].kind != node.reducer.kind or node.schema != source_schemas[0]):
                    raise ValueError('Mean output/input schema mismatch.')
                if isinstance(node.reducer, HardVote) and (source_schemas[0].kind != 'labels' or node.schema.kind != ('vote_fraction' if node.reducer.fractions else 'labels')):
                    raise ValueError('Hard votes must remain labels or explicitly typed vote fractions.')
            if isinstance(node, TransformNode):
                from .transforms import OutputFeatures
                if not isinstance(node.transform, OutputFeatures):
                    raise ValueError('Graph transforms support stateless OutputFeatures; put learned transforms inside child pipelines.')
            active.remove(name)
            ordered.append(name)
        for name in self.nodes:
            visit(name)
        for name, ref in self.outputs.items():
            if not name or not isinstance(ref, OutputRef) or ref.node not in self.nodes or ref.output not in self.schemas(ref.node):
                raise ValueError('Ambiguous or unresolved public output.')
            if name == 'predict_proba' and self.schemas(ref.node)[ref.output].kind != 'probability':
                raise ValueError('predict_proba must be probabilities, never hard-vote fractions.')
        return ordered

    def schemas(self, name):
        node = self.nodes[name]
        return node.schemas if isinstance(node, ModelNode) else {'predict': node.schema}
