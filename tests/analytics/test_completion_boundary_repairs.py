"""Regressions for the three bounded API findings in the 8207e29 re-audit."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from test_ratings import make_history
from split_samples import dataset, rows_for, cases
from xdiyo_analytics.ratings import build_ratings, RatingRun, GlickoTransition
from xdiyo_analytics.labels import create_labels, Outcome
from xdiyo_analytics.training import ProbabilityCalibrator, TrainingRunner
from xdiyo_analytics.training.calibration_split import fold_boundary
from xdiyo_analytics.splits import Fold, SplitPlan


@pytest.mark.parametrize('transition', [None, GlickoTransition()])
@pytest.mark.parametrize('statistic', [False, True])
def test_glicko_mixed_release_boundary_retains_prior_only_view(tmp_path, transition, statistic):
    from test_ratings import CORNER
    history = make_history([dict(day=0), dict(day=2, result='L'), dict(day=3, status='notstarted')])
    boundary = history.kickoff_at.iloc[2]
    history['result_available_at'] = pd.Series(pd.NaT, index=history.index, dtype='datetime64[ns, UTC]')
    history.loc[:1, 'result_available_at'] = boundary
    options = dict(transition=transition, stat=CORNER if statistic else None)
    run = build_ratings(history, **options)
    query = history.iloc[2:4]
    prior = build_ratings(history.iloc[:2], **options)
    pd.testing.assert_frame_equal(run.features(query), prior.features(query))
    before = pd.Series(boundary-pd.Timedelta('1ns'), index=query.index)
    np.testing.assert_allclose(run.features(query, cutoffs=before),
                               prior.features(query, cutoffs=before))
    mutated = history.copy()
    mutated.loc[2:3, 'result'] = ['W', 'L']
    for col in history.attrs['stat_columns']:
        mutated.loc[2:3, col] = 999
    pd.testing.assert_frame_equal(run.features(query), build_ratings(mutated, **options).features(query))
    # The full batch must remain a single simultaneous update after the boundary.
    all_known = history.copy()
    all_known.loc[2:3, 'kickoff_at'] = boundary-pd.Timedelta('1h')
    release = pd.Series(boundary, index=history.index)
    release.iloc[4:] = pd.NaT
    expected = build_ratings(all_known, available_at=release, **options)
    pd.testing.assert_frame_equal(run.features(history.iloc[4:]), expected.features(history.iloc[4:]))
    run.save(tmp_path / 'ratings')
    pd.testing.assert_frame_equal(RatingRun.load(tmp_path / 'ratings').features(query), run.features(query))


@pytest.mark.parametrize('perspective', ['home', 'away'])
@pytest.mark.parametrize('bounds', [(None, '2025-02-01'), ('2025-02-02', '2025-02-01'), (None, None)])
def test_match_labels_preserve_later_bound_from_either_side(perspective, bounds):
    history = make_history([dict(day=0), dict(day=2)]).iloc[[3, 0, 2, 1]].copy()
    history.index = [8, 8, 4, 2]
    history['result_available_at'] = pd.to_datetime([None, bounds[0], None, bounds[1]], utc=True)
    original = deepcopy(history)
    labels = create_labels(history, {'result': Outcome(perspective=perspective), 'team': Outcome()})
    expected = pd.to_datetime(pd.Series(bounds), utc=True).max()
    actual = labels['result'].metadata.result_available_at.iloc[0]
    assert actual == expected if pd.notna(expected) else pd.isna(actual)
    assert pd.isna(labels['result'].metadata.result_available_at.iloc[1])
    pd.testing.assert_series_equal(labels['team'].metadata.result_available_at, history.result_available_at)
    pd.testing.assert_frame_equal(history, original)


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('explicit', [False, True])
def test_probability_calibration_purges_bounds_at_inner_and_outer_boundary(layout, explicit):
    data = dataset(12, layout=layout, shuffle=True)
    data.y = pd.DataFrame({'outcome': (data.metadata.case % 2).to_numpy()}, index=data.X.index)
    data.metadata['result_available_at'] = pd.Series(pd.NaT, index=data.metadata.index, dtype='datetime64[ns, UTC]')
    issue = data.metadata.loc[data.metadata.case.eq(10), 'kickoff_at'].iloc[0]
    # An early match completes after the calibration tail starts; a tail match
    # completes after model issue. Neither may leak into its respective fit.
    data.metadata.loc[data.metadata.case.eq(1), 'result_available_at'] = issue-pd.Timedelta('1h')
    data.metadata.loc[data.metadata.case.eq(9), 'result_available_at'] = issue+pd.Timedelta('1h')
    options = dict(availability_delay='3h', prediction_lead='0h') if explicit else {}
    cal = ProbabilityCalibrator(fraction=.3, **options)
    train, test = rows_for(data, range(10)), rows_for(data, [10, 11])
    fold = Fold(train, test, test)
    meta = fold_boundary(cal, data, fold)
    fitting, reserved, audit = cal.partition(data, train, meta)
    assert cases(data, fitting) == {0, 2, 3, 4, 5, 6}
    assert cases(data, reserved) == {7, 8}
    assert {r['reason'] for r in audit['purged']} == {
        'base_label_unavailable_at_calibration_cutoff', 'calibration_label_unavailable_at_issue'}
    with pytest.raises(ValueError, match='issue_at'):
        cal.partition(data, train)
    # Exercise the native runner, including isolated base fit and calibrator fit.
    from test_probability_calibration import ScopedProbabilities
    fitted, calibrated = [], []
    class SpyModel(ScopedProbabilities):
        def fit(self, context):
            fitted.append(set(context.metadata.case))
            return super().fit(context)
    original_fit = ProbabilityCalibrator.fit
    with pytest.MonkeyPatch.context() as patch:
        def fit(self, probabilities, y):
            calibrated.append(set(data.metadata.loc[y.index, 'case']))
            return original_fit(self, probabilities, y)
        patch.setattr(ProbabilityCalibrator, 'fit', fit)
        TrainingRunner(SpyModel, calibration=cal).run(data, SplitPlan([fold], len(data.X), np.arange(len(data.X))))
    assert fitted == [{0, 2, 3, 4, 5, 6}]
    assert calibrated == [{7, 8}]


def test_null_review_metadata_preserves_legacy_probability_partition():
    data = dataset(10)
    cal = ProbabilityCalibrator(fraction=.3)
    expected = cal.partition(data, np.arange(10))
    data.metadata['result_available_at'] = pd.NaT
    actual = cal.partition(data, np.arange(10))
    for a, b in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(a, b)
    assert actual[2] == expected[2] == {}


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_probability_bound_equality_and_unknown_explicit_release(layout):
    data = dataset(12, layout=layout, shuffle=True)
    data.metadata['release'] = data.metadata.kickoff_at + pd.Timedelta('3h')
    data.metadata['result_available_at'] = pd.Series(pd.NaT, index=data.metadata.index, dtype='datetime64[ns, UTC]')
    first = data.metadata.loc[data.metadata.case.eq(7), 'kickoff_at'].iloc[0]
    data.metadata.loc[data.metadata.case.eq(1), 'result_available_at'] = first
    data.metadata.loc[data.metadata.case.eq(2), 'result_available_at'] = first+pd.Timedelta('1ns')
    data.metadata.loc[data.metadata.case.eq(3), 'release'] = pd.NaT
    data.metadata.loc[data.metadata.case.eq(3), 'result_available_at'] = first
    cal = ProbabilityCalibrator(fraction=.3, availability_column='release', prediction_lead='0h')
    train, test = rows_for(data, range(10)), rows_for(data, [10, 11])
    boundary = fold_boundary(cal, data, Fold(train, test, test))
    fitting, reserved, _ = cal.partition(data, train, boundary)
    assert cases(data, fitting) == {0, 1, 4, 5, 6}
    assert cases(data, reserved) == {7, 8, 9}
