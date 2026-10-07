"""Shared fitted executor for ensembles and one-layer chronological stacks."""
from dataclasses import dataclass, field, replace
from copy import deepcopy
import numpy as np
import pandas as pd
from .contracts import PredictionBundle, TargetSpec
from .graph import ModelNode, TransformNode, ReducerNode
from .training import TrainingPlan, fit_node, subset, prediction_context


@dataclass
class CompositeModelAdapter:
    graph: object
    training_plan: TrainingPlan = field(default_factory=TrainingPlan)
    target_spec: TargetSpec = field(default_factory=TargetSpec)
    supports_sample_weight = True

    @property
    def requires_inner_preparation(self):
        return any(isinstance(n,ModelNode) and n.inputs for n in self.graph.nodes.values())

    def fit(self, context):
        self.order_ = self.graph.order()
        self.training_plan.validate()
        if not context.X.index.is_unique or not context.y.index.equals(context.X.index) or not context.metadata.index.equals(context.X.index):
            raise ValueError('Composite fit requires exact distinct original row identities.')
        if context.validation is not None:
            raise ValueError('Configure validation/calibration inside children, not as outer composite monitoring.')
        if self.target_spec.kind != 'ordinary':
            raise ValueError('Use ResidualModel or an explicit learned target adapter for nonordinary targets.')
        self.models_, self.oof_, self.audit_ = {}, {}, []
        learned = [n for n in self.order_ if isinstance(self.graph.nodes[n],ModelNode) and self.graph.nodes[n].inputs]
        if learned and self.training_plan.mode != 'chronological':
            raise ValueError('Meta models require a chronological TrainingPlan.')
        if learned and context.sample_weight is not None:
            raise ValueError('Chronological stacks require fold-local child weights; external learned weights may leak OOF labels.')
        base = [n for n in self.order_ if n not in learned]
        # Nodes downstream of meta outputs run only after fitting; not OOF inputs.
        dependencies = set()
        def ancestors(name):
            for ref in self.graph.nodes[name].inputs:
                dependencies.add(ref.node)
                ancestors(ref.node)
        for name in learned:
            ancestors(name)
        base = [n for n in self.order_ if n in dependencies] if learned else base
        fits = 0
        if learned:
            collected = {}
            covered = set()
            for inner, (train,test,boundary) in enumerate(self.training_plan.splits(context)):
                if covered.intersection(context.X.index[test]):
                    raise ValueError('Repeated OOF identities are unsupported.')
                covered.update(context.X.index[test])
                training, prediction = subset(context,train), subset(context,test,prediction=True)
                training.fold_metadata['fit_at'] = boundary
                models = {}
                for name in base:
                    node = self.graph.nodes[name]
                    if isinstance(node,ModelNode):
                        fits += int(not node.frozen)
                        if fits > self.training_plan.max_fits:
                            raise ValueError('Composition fitting budget exceeded.')
                        models[name] = fit_node(node,training)
                bundle = self._execute(prediction,models,base)
                for key,frame in bundle.frames.items():
                    collected.setdefault(key,[]).append(frame)
                self.audit_.append(dict(inner_fold=inner,train_keys=context.X.index[train].tolist(),
                                        test_keys=context.X.index[test].tolist(),fit_at=str(boundary)))
            self.oof_ = {key:pd.concat(frames) for key,frames in collected.items()}
            keys = [key for key in context.X.index if key in covered]
            if len(keys) < len(context.X) and self.training_plan.uncovered == 'error':
                raise ValueError('Uncovered OOF warm-up rows; choose explicit drop policy.')
            _,_,available = self.training_plan.times(context)
            boundary = context.fold_metadata.get('fit_at')
            if boundary is None:
                raise ValueError('Chronological stacks require outer fit_at for meta labels and deployment refit.')
            eligible = available.notna() & available.le(pd.to_datetime(boundary,utc=True))
            keys = [k for k in keys if eligible.loc[k]]
            if not keys:
                raise ValueError('No available meta labels at the outer fitting boundary.')
            meta_context = subset(context,context.X.index.get_indexer(keys))
            for name in learned:
                node = self.graph.nodes[name]
                frames = [self.oof_[(ref.node,ref.output)].loc[keys] for ref in node.inputs]
                inputs = node.features.transform(frames,prediction_context(meta_context))
                fits += int(not node.frozen)
                if fits > self.training_plan.max_fits:
                    raise ValueError('Composition fitting budget exceeded.')
                self.models_[name] = fit_node(node,replace(meta_context,X=inputs))
            deployment = subset(context,np.flatnonzero(eligible.to_numpy()))
        else:
            keys = []
            deployment = context
        for name in self.order_:
            node = self.graph.nodes[name]
            if isinstance(node,ModelNode) and not node.inputs:
                fits += int(not node.frozen)
                if fits > self.training_plan.max_fits:
                    raise ValueError('Composition fitting budget exceeded.')
                self.models_[name] = fit_node(node,deployment)
        self.feature_columns_ = tuple(context.X.columns)
        self.fit_at_ = context.fold_metadata.get('fit_at')
        self.training_summary_ = dict(composition_version=1, fits=fits, inner_folds=self.audit_,
            oof_rows=len(keys), uncovered_rows=len(context.X)-len(keys) if learned else 0,
            deployment=self.training_plan.deployment, trained_through=context.fold_metadata.get('fit_at'),
            oof_refit_mismatch='Bases refitted on eligible outer training; meta-model remains fitted on OOF features.' if learned else None,
            child_summaries={name:deepcopy(getattr(m,'training_summary_',{})) for name,m in self.models_.items()})

    def _execute(self, context, models, order):
        self.training_plan.prediction_times(context)
        context = prediction_context(context)
        frames, schemas = {}, {}
        for name in order:
            node = self.graph.nodes[name]
            incoming = [frames[(r.node,r.output)] for r in node.inputs]
            if isinstance(node,ModelNode):
                if node.frozen:
                    issue = pd.to_datetime(context.metadata[self.training_plan.issue_column or self.training_plan.time_column],utc=True)
                    if pd.to_datetime(node.artifact_vintage,utc=True) > issue.min() or pd.to_datetime(node.trained_through,utc=True) >= issue.min():
                        raise ValueError('Frozen model vintage/training cutoff follows prediction issue time.')
                X = node.features.transform(incoming,context) if node.inputs else context.X
                outputs = models[name].predict(replace(context,X=X))
                if not set(node.schemas) <= set(outputs):
                    raise ValueError(f'Missing child output from {name}.')
            elif isinstance(node,TransformNode):
                outputs = {'predict':node.transform.transform(incoming,context)}
            else:
                outputs = {'predict':node.reducer.reduce(incoming,[schemas[(r.node,r.output)] for r in node.inputs])}
            for output,schema in self.graph.schemas(name).items():
                frames[(name,output)] = schema.validate(outputs[output],context)
                schemas[(name,output)] = schema
        return PredictionBundle(frames,schemas,{'fold_id':context.fold_id,'fold_metadata':deepcopy(context.fold_metadata)})

    def predict(self, context):
        if not hasattr(self,'models_'):
            raise ValueError('Composite model is not fitted.')
        if tuple(context.X.columns) != self.feature_columns_:
            raise ValueError('Composite prediction feature order differs from fit.')
        _, issue = self.training_plan.prediction_times(context)
        if self.requires_inner_preparation:
            if issue.isna().any() or (issue < pd.to_datetime(self.fit_at_,utc=True)).any():
                raise ValueError('Prediction issue time precedes the composite deployment fitting cutoff.')
        return self._execute(context,self.models_,self.order_).public(self.graph.outputs)

    def configure_device(self, device):
        if device not in {'cpu','auto'}:
            raise ValueError('Composition currently supports CPU execution only.')
        return 'cpu'
