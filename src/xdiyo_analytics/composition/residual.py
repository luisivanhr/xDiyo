"""Two explicitly distinct correction schedules; outer evaluation stays untouched."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .training import TrainingPlan, fit_node, subset, prediction_context
from .contracts import TargetSpec
from .graph import ModelNode


@dataclass
class ResidualModel:
    """Regression residuals or binary logistic pseudoresiduals with logit reconstruction.

    algorithmic_residual fits a correction to in-sample training residuals.
    honest_error_meta fits only to chronological OOF residuals. It refits the base
    on eligible development rows and freezes the OOF-trained correction.
    """
    base: ModelNode
    correction: object
    training: TrainingPlan
    target_spec: TargetSpec
    learning_rate: float = 1.
    clip: float = 1e-8
    supports_sample_weight = True

    @property
    def requires_inner_preparation(self):
        return self.training.mode=='honest_error_meta'

    def fit(self,context):
        from ..training.contracts import PredictionContext
        self.training.validate()
        if context.validation is not None:
            raise ValueError('Configure monitoring/calibration inside the residual base.')
        if self.requires_inner_preparation and context.sample_weight is not None:
            raise ValueError('Honest OOF correction requires child-local weights, not outer learned weights.')
        if self.training.mode not in {'algorithmic_residual','honest_error_meta'} or self.target_spec.kind not in {'residual','oof_error'}:
            raise ValueError('ResidualModel needs an explicit algorithmic_residual or honest_error_meta training mode and residual/oof_error target.')
        expected = 'residual' if self.training.mode=='algorithmic_residual' else 'oof_error'
        if self.target_spec.kind != expected:
            raise ValueError('TargetSpec and residual schedule disagree.')
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1 or not 0 < self.clip < .5:
            raise ValueError('Invalid correction learning rate or clipping.')
        if context.y.shape[1] != 1 or self.base.inputs or self.base.frozen:
            raise ValueError('Residual correction currently requires one target and a refit-capable input-free base.')
        self.target_ = context.y.columns[0]
        self.output_ = 'predict_proba' if self.target_spec.loss=='log_loss' else 'predict'
        if self.output_ not in self.base.schemas:
            raise ValueError('Residual base lacks the required declared output.')
        schema = self.base.schemas[self.output_]
        if schema.target != self.target_ or schema.link != 'identity':
            raise ValueError('Residual base target and link must match the correction target in original units.')
        if self.output_ == 'predict' and (schema.kind != 'regression' or schema.units in {'probability', 'logit', 'logits'}):
            raise ValueError('Regression residuals require regression outputs in original target units.')
        if self.output_ == 'predict_proba' and schema.units != 'probability':
            raise ValueError('Binary residuals require probability units.')
        self.classes_ = tuple(schema.classes)
        if self.output_=='predict_proba' and (schema.kind!='probability' or len(self.classes_)!=2):
            raise ValueError('Classification correction supports binary probability bases only.')
        self.audit_=[]
        splits = list(self.training.splits(context)) if self.training.mode=='honest_error_meta' else []
        if len(splits)+2 > self.training.max_fits:
            raise ValueError('Composition fitting budget exceeded.')
        if self.training.mode=='honest_error_meta':
            outputs=[]
            for train,test,boundary in splits:
                fitting=subset(context,train)
                fitting.fold_metadata['fit_at']=boundary
                model=fit_node(self.base,fitting)
                prediction=subset(context,test,prediction=True)
                frame=model.predict(prediction)[self.output_]
                schema.validate(frame,prediction)
                outputs.append(frame)
                self.audit_.append(dict(train_keys=context.X.index[train].tolist(),test_keys=context.X.index[test].tolist(),fit_at=str(boundary)))
            predicted=pd.concat(outputs)
            if not predicted.index.is_unique:
                raise ValueError('Repeated OOF correction identities.')
            if self.training.uncovered=='error' and len(predicted)!=len(context.X):
                raise ValueError('Uncovered correction OOF rows.')
            _,_,available=self.training.times(context)
            if 'fit_at' not in context.fold_metadata:
                raise ValueError('Honest correction requires an outer fit_at.')
            eligible=available.notna() & available.le(pd.to_datetime(context.fold_metadata['fit_at'],utc=True))
            keys=[k for k in context.X.index if k in predicted.index and eligible.loc[k]]
            if not keys:
                raise ValueError('No available OOF correction labels at outer fit_at.')
            correction_context=subset(context,context.X.index.get_indexer(keys))
            predicted=predicted.loc[keys]
            self.base_=fit_node(self.base,subset(context,np.flatnonzero(eligible.to_numpy())))
        else:
            self.base_=fit_node(self.base,context)
            prediction=subset(context,np.arange(len(context.X)),prediction=True)
            predicted=self.base_.predict(prediction)[self.output_]
            schema.validate(predicted,prediction)
            correction_context=context
        self.oof_predictions_=predicted.copy() if self.training.mode=='honest_error_meta' else None
        if self.output_=='predict_proba':
            labels=correction_context.y.iloc[:,0]
            if not labels.isin(self.classes_).all():
                raise ValueError('Unknown correction class.')
            # Negative gradient of binary log loss on logit scale, never logit(y).
            residual=labels.eq(self.classes_[1]).astype(float)-predicted.iloc[:,1]
        else:
            residual=correction_context.y.iloc[:,0]-predicted.iloc[:,0]
        target=pd.DataFrame({self.target_:residual},index=correction_context.X.index)
        from .training import fresh
        self.correction_=fresh(self.correction)
        self.correction_.fit(replace(correction_context,y=target,validation=None))
        self.feature_columns_=tuple(context.X.columns)
        self.fit_at_=context.fold_metadata.get('fit_at')
        self.training_summary_=dict(mode=self.training.mode,target=repr(self.target_spec),inner_folds=self.audit_,
                                   fits=len(splits)+2,
                                   correction_rows=len(target),uncovered_rows=len(context.X)-len(target),
                                   deployment='Refit base; freeze correction. OOF/full-refit feature distribution mismatch remains.' if self.audit_ else 'In-sample boosting stage within training only.')

    def predict(self,context):
        if not hasattr(self,'base_') or tuple(context.X.columns)!=self.feature_columns_:
            raise ValueError('Fit correction first and preserve feature ordering.')
        _, issue = self.training.prediction_times(context)
        context = prediction_context(context)
        if self.requires_inner_preparation:
            if issue.isna().any() or (issue<pd.to_datetime(self.fit_at_,utc=True)).any():
                raise ValueError('Prediction issue time precedes correction deployment cutoff.')
        base=self.base_.predict(context)[self.output_]
        self.base.schemas[self.output_].validate(base,context)
        correction=self.correction_.predict(context)['predict']
        if not correction.index.equals(base.index) or list(correction.columns)!=[self.target_] or not np.isfinite(correction.to_numpy(dtype=float)).all():
            raise ValueError('Correction output does not match target/row contract.')
        if self.output_=='predict':
            reconstructed = base+self.learning_rate*correction
            self.base.schemas[self.output_].validate(reconstructed,context)
            return {'predict':reconstructed,'base/predict':base,'correction/predict':correction}
        from scipy.special import expit
        p=base.iloc[:,1].clip(self.clip,1-self.clip)
        corrected=expit(np.log(p)-np.log1p(-p)+self.learning_rate*correction.iloc[:,0])
        proba=pd.DataFrame(np.column_stack([1-corrected,corrected]),index=base.index,columns=base.columns)
        self.base.schemas[self.output_].validate(proba,context)
        labels=np.asarray(self.classes_)[proba.to_numpy().argmax(axis=1)]
        return {'predict_proba':proba,'predict':pd.DataFrame({self.target_:labels},index=base.index),'base/predict_proba':base,'correction/predict':correction}

    def configure_device(self,device):
        if device not in {'cpu','auto'}:
            raise ValueError('Residual composition currently supports CPU execution.')
        return 'cpu'
