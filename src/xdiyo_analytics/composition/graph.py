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
