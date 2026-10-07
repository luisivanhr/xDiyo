"""Acceptance regressions for the independent 25eb79a branch review."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xdiyo_analytics.composition import (
    OutputSchema, ModelNode, Estimator, ModelStack, OutputFeatures, TrainingPlan,
    TargetSpec, ResidualModel, CompositeModelAdapter, PredictionGraphSpec, OutputRef,
)
from xdiyo_analytics.composition.training import fresh, subset
from xdiyo_analytics.training import EstimatorAdapter, TrainingRunner, save_model, load_model, ExecutionPolicy
from xdiyo_analytics.training.calibration import CalibratedAdapter, ProbabilityCalibrator
from xdiyo_analytics.training.targets import TargetTransformAdapter
from xdiyo_analytics.training.execution import DeviceAdapter
from xdiyo_analytics.training.contracts import FitContext, PredictionContext
from xdiyo_analytics.splits import SplitPlan
from test_model_composition import setup


def context(data):
    return FitContext(data.X.iloc[:12].copy(), data.metadata.iloc[:12].copy(), data.layout,
                      data.match_columns, 0, {'fit_at':data.metadata.kickoff_at.iloc[12]}, y=data.y.iloc[:12].copy())


def stack(base=None, **options):
    schema = OutputSchema('target')
    return ModelStack({'base':ModelNode(base or Estimator(Ridge()), {'predict':schema})},
                      Estimator(Ridge()), OutputFeatures(('base',)), schema,
                      TrainingPlan(mode='chronological', min_train_groups=4, availability_delay='3h',
                                   feature_availability_column='feature_at', **options))


def residual(**options):
    return ResidualModel(ModelNode(Estimator(Ridge()), {'predict':OutputSchema('target')}), Estimator(Ridge()),
                         TrainingPlan(mode='honest_error_meta', min_train_groups=4, availability_delay='3h',
                                      feature_availability_column='feature_at', **options), TargetSpec('oof_error'))


@pytest.mark.parametrize('wrapped', [False, True])
def test_retained_calibrator_is_rejected_instead_of_reusing_future_labels(wrapped):
    data, _ = setup(True)
    ctx = context(data)
    for later_label in ('a', 'b'):
        base = EstimatorAdapter(DummyClassifier(strategy='prior'), ('predict_proba',))
        base.fit(ctx)
        calibrator = ProbabilityCalibrator(method='isotonic')
        p = pd.DataFrame(np.tile([.5,.5],(6,1)), columns=pd.MultiIndex.from_tuples([('target','a'),('target','b')]))
        calibrator.fit(p, pd.DataFrame({'target':[later_label]*5+['b' if later_label=='a' else 'a']}))
        model = CalibratedAdapter(base, calibrator, np.arange(6))
        if wrapped:
            model = DeviceAdapter(model, lambda:('cpu',), lambda *a:None)
        with pytest.raises(ValueError, match='ModelNode.calibration'):
            fresh(model)


@pytest.mark.parametrize('parallel', [False, True])
def test_retained_target_wrapper_reset_is_future_invariant(parallel, tmp_path, monkeypatch):
    data, plan = setup()
    data.metadata['feature_at'] = data.metadata.kickoff_at
    if parallel:
        plan = SplitPlan([deepcopy(plan.folds[0]),deepcopy(plan.folds[0])], plan.n_rows, plan.row_order)
    results = []
    for later_label in (0.,100.):
        old = TargetTransformAdapter(EstimatorAdapter(RandomForestRegressor(n_estimators=3,warm_start=True,random_state=0)), StandardScaler())
        ctx = context(data)
        old.fit(replace(ctx, y=ctx.y*0+later_label))
        spec = stack(old)
        result = TrainingRunner(spec.build,execution=ExecutionPolicy(2) if parallel else None).run(data,plan)
        results.append(result)
        # Fresh fitting must not mutate retained input trees or transformer.
        assert old.predict(subset(ctx, np.arange(12), prediction=True))['predict'].iloc[0,0] == later_label
    for first, second in zip(results[0].folds,results[1].folds):
        pd.testing.assert_frame_equal(first.model.oof_[('base','predict')],second.model.oof_[('base','predict')])
        pd.testing.assert_frame_equal(first.predictions['predict'],second.predictions['predict'])
    path = save_model(results[0],tmp_path/'retained',fold_id=0)
    monkeypatch.setattr(RandomForestRegressor,'fit',lambda *a,**k:pytest.fail('restoration must not fit'))
    pd.testing.assert_frame_equal(load_model(path).predict(data,positions=np.arange(12,16))['predict'],results[0].folds[0].predictions['predict'])


@pytest.mark.parametrize('kind', ['stack', 'residual'])
@pytest.mark.parametrize('bad', ['future', 'missing', 'issue_after_kickoff', 'missing_column'])
def test_outer_and_restored_prediction_timing_is_label_free(kind,bad,tmp_path,monkeypatch):
    data, plan = setup()
    data.metadata['feature_at'] = data.metadata.kickoff_at
    data.metadata['issue_at'] = data.metadata.kickoff_at
    model = stack(issue_column='issue_at').build() if kind=='stack' else residual(issue_column='issue_at')
    result = TrainingRunner(lambda:deepcopy(model)).run(data,plan)
    path = save_model(result,tmp_path/'timing')
    changed = deepcopy(data)
    if bad=='future':
        changed.metadata.loc[12:,'feature_at'] += pd.Timedelta('1d')
    elif bad=='missing':
        changed.metadata.loc[12:,'feature_at'] = pd.NaT
    elif bad=='missing_column':
        changed.metadata = changed.metadata.drop(columns='feature_at')
    else:
        changed.metadata.loc[12:,'issue_at'] += pd.Timedelta('1h')
    with pytest.raises(ValueError, match='[Ff]eatures|timing|issue/kickoff'):
        TrainingRunner(lambda:deepcopy(model)).run(changed,plan)
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('restoration must not fit'))
    restored = load_model(path)
    with pytest.raises(ValueError, match='[Ff]eatures|timing|issue/kickoff'):
        restored.predict(changed,positions=np.arange(12,16))
    # Direct restored adapter inference only needs prediction metadata, no y.
    ctx = PredictionContext(data.X.iloc[12:],data.metadata.iloc[12:],data.layout,data.match_columns,0)
    ctx.X = ctx.X.rename_axis('row_position')
    ctx.metadata = ctx.metadata.rename_axis('row_position')
    actual = result.folds[0].model.predict(ctx)['predict']
    pd.testing.assert_frame_equal(actual,result.folds[0].predictions['predict'])


class OutcomeBoundaryProbe:
    def fit(self, context):
        self.target_ = context.y.columns[0]

    def predict(self, context):
        assert not hasattr(context,'y')
        assert not {'settlement::target','training_label','actual','label::target','y','net_return_per_unit','gross_return','return_multiplier'} & set(context.metadata)
        assert 'settlement::target' not in context.metadata.attrs
        assert 'training_label' not in context.fold_metadata
        assert 'y' not in context.X.attrs
        assert 'y' not in context.fold_metadata.get('nested',[{}])[0]
        return {'predict':pd.DataFrame({self.target_:0.},index=context.X.index)}


def test_known_outcome_fields_are_absent_in_oof_and_outer_predictions():
    data, plan = setup()
    data.metadata['feature_at'] = data.metadata.kickoff_at
    for key in ('settlement::target','training_label','actual','label::target','y','net_return_per_unit','gross_return','return_multiplier'):
        data.metadata[key] = data.y.target
    data.metadata.attrs['settlement::target'] = data.y.target.tolist()
    plan.folds[0].metadata['training_label'] = data.y.target.tolist()
    plan.folds[0].metadata['nested'] = [{'y':data.y.target.tolist()}]
    data.X.attrs['y'] = data.y.target.tolist()
    result = TrainingRunner(stack(OutcomeBoundaryProbe()).build).run(data,plan)
    assert result.folds[0].model.oof_[('base','predict')].eq(0).all().all()
    assert 'settlement::target' in data.metadata  # caller data is untouched


@pytest.mark.parametrize('schema', [OutputSchema('other_target'), OutputSchema('target',link='log'),
    OutputSchema('target',units='probability'), OutputSchema('target',kind='features')])
def test_incompatible_residual_schema_rejected_before_fit(schema,monkeypatch):
    data,_ = setup()
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('reject before fitting'))
    model = ResidualModel(ModelNode(Estimator(Ridge()),{'predict':schema}),Estimator(Ridge()),
                          TrainingPlan(mode='algorithmic_residual'),TargetSpec('residual'))
    with pytest.raises(ValueError,match='target|original'):
        model.fit(context(data))


class ConstantOutput:
    def __init__(self, values, columns):
        self.values, self.columns = values, columns
    def fit(self, context): pass
    def predict(self, context):
        return {'predict':pd.DataFrame([self.values]*len(context.X),index=context.X.index,columns=self.columns)}


def test_residual_reconstruction_rejects_overflow():
    data,_ = setup()
    model = ResidualModel(ModelNode(Estimator(Ridge()),{'predict':OutputSchema('target')}),Estimator(Ridge()),
                          TrainingPlan(mode='algorithmic_residual'),TargetSpec('residual'))
    model.fit(context(data))
    model.base_ = model.correction_ = ConstantOutput([1e308],['target'])
    with pytest.raises(ValueError,match='finite'):
        model.predict(subset(context(data),[0],prediction=True))


@pytest.mark.parametrize('values,columns', [([-1.,2.],[0,1]),([.2,.2],[0,1]),([.5,.5],[1,0]),
    ([.2,.8],[0,2]),([np.nan,1.],[0,1]),([np.inf,0.],[0,1])])
def test_distribution_public_edge_rejects_invalid_pmf(values,columns):
    data,_ = setup()
    schema = OutputSchema('target',kind='distribution',family='categorical_pmf',support=(0,1),parameters=('mass',),units='probability')
    graph = PredictionGraphSpec({'pmf':ModelNode(ConstantOutput(values,columns),{'predict':schema})},{'predict':OutputRef('pmf')})
    model = CompositeModelAdapter(graph)
    model.fit(context(data))
    with pytest.raises(ValueError,match='PMF'):
        model.predict(subset(context(data),[0],prediction=True))


@pytest.mark.parametrize('options', [dict(family='poisson'),dict(parameters=('mean',)),dict(units='count'),dict(link='log'),dict(support=(0,0)),dict(support=(0,np.nan))])
def test_unsupported_distribution_semantics_fail_at_declaration(options):
    fields=dict(kind='distribution',family='categorical_pmf',support=(0,1),parameters=('mass',),units='probability')
    with pytest.raises(ValueError,match='distribution|categorical_pmf|support'):
        OutputSchema('target',**(fields|options))


def test_distribution_edge_rechecks_restored_semantics_and_accepts_valid_mass():
    data,_=setup()
    prediction=subset(context(data),[0],prediction=True)
    schema=OutputSchema('target',kind='distribution',family='categorical_pmf',support=(0,1),parameters=('mass',),units='probability')
    frame=pd.DataFrame([[.25,.75]],index=prediction.X.index,columns=[0,1])
    pd.testing.assert_frame_equal(schema.validate(frame,prediction),frame)
    # Simulates a retained pre-repair schema whose constructor did not validate.
    object.__setattr__(schema,'family','unsupported')
    with pytest.raises(ValueError,match='family validator'):
        schema.validate(frame,prediction)
