"""Native diagnostic extraction and explicit unsupported feature naming."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge
from xdiyo_analytics.training.inspection import estimator_history, estimator_coefficients


def test_native_histories_keep_metric_partition_iteration_and_summary():
    class Fitted:
        loss_curve_ = [4., 2., 1.]
        evals_result_ = {'validation': {'mae': [3., 2.]}, 'training': {'rmse': [2., 1., .5]}}
        n_iter_ = np.array([3, 2])
        best_iteration_ = np.int64(1)
    history, summary = estimator_history(Fitted(), 11)
    assert history.fold_id.eq(11).all() and history.attempt.eq(0).all()
    assert history.metric.tolist() == ['train_loss']*3 + ['validation.mae']*2 + ['training.rmse']*3
    np.testing.assert_allclose(history.value, [4, 2, 1, 3, 2, 2, 1, .5])
    assert history.step.tolist() == [1, 2, 3, 1, 2, 1, 2, 3]
    assert summary == {'history_source': 'native_estimator', 'termination_reason': 'estimator_managed',
                       'n_iter': [3, 2], 'best_iteration': 1}


def test_pipeline_without_transformed_names_does_not_guess_coefficients():
    X = pd.DataFrame({'a': [0., 1., 2., 3.], 'b': [1., 0., 1., 0.]})
    model = make_pipeline(FunctionTransformer(lambda frame: np.asarray(frame)[:, ::-1]), Ridge()).fit(X, [1, 2, 3, 4])
    with pytest.raises(AttributeError): estimator_coefficients(model, ['a', 'b'], ['target'])
