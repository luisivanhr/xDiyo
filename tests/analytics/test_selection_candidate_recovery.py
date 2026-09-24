"""Saved candidate fits can be rescored without repeating model training."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, post
from model_selection_samples import sample, plan, development
from test_football_experiment_search import CountingFactory
from test_model_selection_fit_evidence import GaussianAdapter
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.evaluation import Metric
from xdiyo_analytics.experiments import ExperimentStore, FootballExperiment
from xdiyo_analytics.experiments.recovery import execution_key
from xdiyo_analytics.selection import Candidate, ModelSelection, MetricSelection, ParsimonySelection, WeightedSelection
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, default_recipe, node


def candidates(events):
    return [Candidate(f'Ridge {alpha}', CountingFactory(alpha, events), config={'alpha': alpha})
            for alpha in (.01, 1.)]


@pytest.mark.parametrize('rule', [MetricSelection('mae'), ParsimonySelection('mae'),
                                 WeightedSelection({'mse': .5, 'mae': .5}), MetricSelection('aic')])
def test_missing_decision_evidence_fails_before_any_fit(rule):
    data, events = sample(), []
    with pytest.raises(ValueError, match='(?i)(metric|evidence|aic)'):
        ModelSelection(candidates(events), metrics='mse', decision=rule).run(
            data, plan(data), development_positions=development(data))
    assert events == []


def test_named_metric_key_is_accepted_as_decision_evidence():
    data, events = sample(), []
    result = ModelSelection(candidates(events), metrics=[Metric('mse', key='squared_error')],
        decision=MetricSelection('squared_error')).run(data, plan(data), development_positions=development(data))
    assert len(events) == 4
    assert 'raw::squared_error' in result.comparison


def test_experiment_checks_missing_evidence_before_preanalysis_or_fitting(tmp_path, monkeypatch):
    bundle, events = prepared(holdout=True), []
    def forbidden(*args, **kwargs):
        raise AssertionError('Pre-analysis ran before validating decision evidence')
    monkeypatch.setattr(PreTrainingAnalysis, 'run', forbidden)
    search = ModelSelection(candidates(events), metrics='mse', decision=MetricSelection('mae'))
    with pytest.raises(ValueError, match='(?i)(metric|evidence)'):
        FootballExperiment('Early validation', output_dir=tmp_path).run(
            bundle, model_selection=search, selection_plan=plan(bundle.dataset))
    assert events == []


class GaussianFactory:
    def __init__(self, linear, events):
        self.linear, self.events = linear, events

    def cache_key(self):
        return {'linear': self.linear, 'adapter': 'Gaussian known variance'}

    def __call__(self):
        owner = self
        class CountedGaussian(GaussianAdapter):
            def fit(self, context):
                owner.events.append(owner.linear)
                return super().fit(context)
        return CountedGaussian(owner.linear)


def test_rescoring_preserves_saved_model_only_information_criteria(tmp_path):
    data, events = sample(), []
    store = ExperimentStore(tmp_path, 'Preserved fitting evidence')
    items = [Candidate(str(linear), GaussianFactory(linear, events), config={'linear': linear})
             for linear in [False, True]]
    first = ModelSelection(items, metrics='mse', information_criteria=True, decision=MetricSelection('aic')).run(
        data, plan(data), development_positions=development(data), experiment=store, resume=True)
    revised = ModelSelection(items, metrics='mae', information_criteria=True, decision=MetricSelection('bic')).run(
        data, plan(data), development_positions=development(data), experiment=store, resume=True)
    assert len(events) == 4
    for old, new in zip(first.trials, revised.trials):
        assert old.saved_run_id == new.saved_run_id
        assert old.complexity == new.complexity
        old_fit = next(s.result.tables['fit_statistics'] for s in old.report.studies if s.name == 'fit_evidence')
        new_fit = next(s.result.tables['fit_statistics'] for s in new.report.studies if s.name == 'fit_evidence')
        pd.testing.assert_frame_equal(old_fit, new_fit)
        assert {'aic', 'bic', 'mae'} <= {m['metric'] for m in new.record['metrics']}


def test_changed_decision_and_standard_metrics_reuse_fits_without_overwriting_trials(tmp_path):
    data, events = sample(), []
    store = ExperimentStore(tmp_path, 'Rescore saved candidates')
    first = ModelSelection(candidates(events), metrics='mse').run(data, plan(data),
        development_positions=development(data), experiment=store, resume=True, recovery_namespace='first view')
    assert len(events) == 4
    original = {trial.saved_run_id: (store.path / 'runs' / trial.saved_run_id / 'run.json').read_bytes()
                for trial in first.trials}
    revised = ModelSelection(candidates(events), metrics=['mse', 'mae'], pooling='mean',
        decision=WeightedSelection({'mse': .25, 'mae': .75})).run(data, plan(data),
        development_positions=development(data), experiment=store, resume=True, recovery_namespace='new view')
    assert len(events) == 4
    assert [trial.saved_run_id for trial in revised.trials] == list(original)
    assert all(fold.model is None for trial in revised.trials for fold in trial.training.folds)
    assert {'raw::mse', 'raw::mae'} <= set(revised.comparison)
    for trial in revised.trials:
        assert {'mse', 'mae'} <= {metric['metric'] for metric in trial.record['metrics']}
        assert original[trial.saved_run_id] == (store.path / 'runs' / trial.saved_run_id / 'run.json').read_bytes()
    assert len(store.read_runs(role='trial')) == 2


@pytest.mark.parametrize('change', ['X', 'y', 'split', 'parameters'])
def test_changed_fitting_inputs_do_not_reuse_candidates(tmp_path, change):
    data, events = sample(), []
    splits = plan(data)
    store = ExperimentStore(tmp_path, 'Fitting boundaries')
    first = ModelSelection(candidates(events), metrics='mse').run(data, splits,
        development_positions=development(data), experiment=store, resume=True)
    altered, revised = deepcopy(data), candidates(events)
    if change == 'X':
        altered.X.iloc[0, 0] += .5
    elif change == 'y':
        altered.y.iloc[0, 0] += .5
    elif change == 'split':
        splits = deepcopy(splits)
        splits.folds[0] = replace(splits.folds[0], train=splits.folds[0].train[:-1])
    else:
        revised = [Candidate('Ridge new', CountingFactory(.4, events), config={'alpha': .4})]
    result = ModelSelection(revised, metrics='mse').run(altered, splits,
        development_positions=development(altered), experiment=store, resume=True)
    assert len(events) == 4 + 2 * len(revised)
    assert not ({trial.saved_run_id for trial in first.trials} & {trial.saved_run_id for trial in result.trials})


def test_forced_fresh_search_refits_candidates_and_preserves_recoverable_trials(tmp_path):
    bundle, events = prepared(holdout=True), []
    experiment = FootballExperiment('Force fresh search', output_dir=tmp_path)
    search = ModelSelection(candidates(events), metrics='mse')
    first = experiment.run(bundle, model_selection=search, selection_plan=plan(bundle.dataset), post_analysis=post())
    assert len(events) == 5
    second = experiment.run(bundle, model_selection=search, selection_plan=plan(bundle.dataset), post_analysis=post(), reuse=False)
    assert len(events) == 10
    assert first.record['run_group'] != second.record['run_group']
    assert {t.saved_run_id for t in first.selection.trials}.isdisjoint(t.saved_run_id for t in second.selection.trials)
    for trial in second.selection.trials:
        loaded = experiment.store.load_run(trial.saved_run_id)
        assert loaded['training'] is not None


def test_ui_factory_identity_ignores_only_nonfitting_recipe_fields():
    catalog = catalog_for_ui()
    recipe = default_recipe()
    baseline = execution_key(ModelFactory(recipe, catalog))
    changed = deepcopy(recipe)
    changed['name'] = 'Another display name'
    changed['search'] = {'options': {'decision': node('selection.MetricSelection', metric='mae')}}
    changed['pre_reporters'] = {}
    changed['post_reporters'] = {'Changed': node('reporting.PerformanceReporter', type='overall', metrics=['mae'])}
    changed['output_dir'] = 'a-different-output-view'
    assert execution_key(ModelFactory(changed, catalog)) == baseline
    changed['preprocessors'] = [node('sklearn.preprocessing.RobustScaler')]
    assert execution_key(ModelFactory(changed, catalog)) != baseline
    changed = deepcopy(recipe)
    changed['model']['params']['alpha'] = 5.
    assert execution_key(ModelFactory(changed, catalog)) != baseline
    changed = deepcopy(recipe)
    changed['target_transformer'] = node('targets.StandardScaler')
    assert execution_key(ModelFactory(changed, catalog)) != baseline
