"""Opt-in label distributions share feature scope while retaining table identity."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

from reporting_samples import context, sample, plan
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import FeatureDistributionReporter
from xdiyo_analytics.ui import recipe as recipes
from test_ui_workflow import ui_recipe


def test_distribution_label_picker_uses_discovered_targets_and_opt_in_help():
    from xdiyo_analytics.ui.inventory import inventory
    fields = {field['name']: field for field in inventory()['components']['reporting.FeatureDistributionReporter']['fields']}
    labels = fields['targets']
    assert labels['title'] == 'Labels'
    assert labels['kind'] == 'multiselect' and labels['discovery'] == 'targets'
    assert labels['default'] is None
    assert 'omit' in labels['help'].lower()


@pytest.mark.parametrize('kind', ['overall', 'per_fold'])
def test_feature_label_distributions_use_identical_selected_training_rows(kind):
    data = sample(shuffle=True)
    splits = plan(data)
    reporter = FeatureDistributionReporter(type=kind, partition='train', features=['linear'],
                                           targets=['first'], bins=3, bandwidth=.4, grid_size=11)
    report = PreTrainingAnalysis({'distributions': reporter}).run(data, split_plan=splits)
    for study in report.studies:
        result = study.result
        rows = study.row_positions
        expected = data.y.iloc[rows]['first'].to_numpy()
        summary = result.tables['label_summary'].iloc[0]
        assert summary.label == 'first' and summary.rows == len(rows)
        assert result.tables['summary'].iloc[0].rows == len(rows)
        assert result.tables['label_histogram']['count'].sum() == len(expected)
        np.testing.assert_array_equal(result.tables['label_histogram']['count'], np.histogram(expected, bins=3)[0])
        curve = result.tables['label_kde']
        np.testing.assert_allclose(curve.density, gaussian_kde(expected, bw_method=.4)(curve.x))
        assert [item.title for item in result.artifacts] == ['linear', 'Label · first']


def test_same_named_feature_and_label_stay_separate_and_opt_in():
    ctx = context({'same': [1, 2, 3, 4]}, {'same': [10, 20, 30, 40]})
    before = deepcopy(ctx)
    plain = FeatureDistributionReporter(type='overall', partition='all').run(ctx)
    labels = FeatureDistributionReporter(type='overall', partition='all', targets=['same']).run(ctx)
    assert set(plain.tables) == {'summary', 'histogram', 'kde'}
    for name, table in plain.tables.items():
        pd.testing.assert_frame_equal(labels.tables[name], table)
    assert 'feature' in labels.tables['summary'] and 'label' not in labels.tables['summary']
    assert 'label' in labels.tables['label_summary'] and 'feature' not in labels.tables['label_summary']
    assert labels.tables['histogram'].right.max() == 4
    assert labels.tables['label_histogram'].right.max() == 40
    assert [item.title for item in labels.artifacts] == ['same', 'Label · same']
    pd.testing.assert_frame_equal(ctx.X, before.X)
    pd.testing.assert_frame_equal(ctx.y, before.y)


@pytest.mark.parametrize('values,finite,status', [([3, 3, np.nan, np.inf], 2, 'insufficient_spread'),
                                               ([np.nan, np.inf, -np.inf, 'bad'], 0, 'no_finite_values'),
                                               ([3, np.nan], 1, 'insufficient_spread')])
def test_label_only_context_and_nonfinite_constant_labels(values, finite, status):
    ctx = context({'temporary': [0] * len(values)}, {'observed': values})
    ctx.X = ctx.X.iloc[:, :0]
    result = FeatureDistributionReporter(type='overall', partition='all', targets=['observed']).run(ctx)
    assert result.tables['summary'].empty
    summary = result.tables['label_summary'].iloc[0]
    assert summary.finite == finite and summary.excluded == len(values) - finite
    assert summary.kde_status == status
    assert result.tables['label_kde'].empty
    assert [item.title for item in result.artifacts] == ['Label · observed']


@pytest.mark.parametrize('section,title', [('pre_reporters', 'Pre-training'), ('fitted_reporters', 'Pre-training'), ('post_reporters', 'Post-training')])
@pytest.mark.parametrize('field,display', [('features', 'Features'), ('targets', 'Labels')])
def test_empty_saved_reporter_shape_errors_before_preparation(ui_recipe, monkeypatch, section, title, field, display):
    ui_recipe[section] = {'Distribution': recipes.node('reporting.FeatureDistributionReporter', type='overall', partition='train', **{field: []})}
    before = deepcopy(ui_recipe)
    monkeypatch.setattr(recipes, 'prepare_recipe', lambda *a, **kw: pytest.fail('Empty selection reached preparation'))
    with pytest.raises(ValueError, match=f'{title} analysis → Distribution → {display}'):
        recipes.run_recipe(ui_recipe)
    assert ui_recipe == before


@pytest.mark.parametrize('params', [{}, {'features': None, 'targets': None}, {'features': ['home::corners_mean'], 'targets': ['corners']}])
def test_valid_or_disabled_selections_pass_guard_unchanged(ui_recipe, monkeypatch, params):
    ui_recipe['pre_reporters'] = {'Distribution': recipes.node('reporting.FeatureDistributionReporter', type='overall', partition='train', **params)}
    before = deepcopy(ui_recipe)
    class PreparationReached(Exception):
        pass
    def stop(value, **kwargs):
        assert value['pre_reporters'] == before['pre_reporters']
        raise PreparationReached
    monkeypatch.setattr(recipes, 'prepare_recipe', stop)
    with pytest.raises(PreparationReached):
        recipes.run_recipe(ui_recipe)
    assert ui_recipe == before
