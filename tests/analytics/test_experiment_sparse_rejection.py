"""Sparse artifacts must fail explicitly before a final run is published."""
from dataclasses import dataclass

import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from model_selection_samples import FixedAdapter
from test_post_training_experiments import computed_run
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import ExperimentStore, FootballExperiment
from xdiyo_analytics.experiments.recovery import pack
from xdiyo_analytics.experiments.store import _table_document
from xdiyo_analytics.reporting import PredictionReporter, StudyResult
from xdiyo_analytics.selection import Candidate


def sparse_series(dtype):
    values = [False, True, False] if dtype == "bool" else [0, 1, 0]
    fill = False if dtype == "bool" else 0
    return pd.Series(values, dtype=pd.SparseDtype(dtype, fill_value=fill), name="value")


@pytest.mark.parametrize("dtype", ["float64", "int64", "bool", "object"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("artifact", ["report", "history"])
def test_sparse_numerical_artifacts_are_rejected_before_publication(tmp_path, dtype, empty, artifact):
    training, report = computed_run()
    table = sparse_series(dtype).to_frame()
    if empty:
        table = table.iloc[:0]
    if artifact == "history":
        training.folds[0].training_history = table
    else:
        report.studies[0].result.tables["sparse"] = table
    store = ExperimentStore(tmp_path, "Sparse numerical")
    with pytest.raises(TypeError, match="[Ss]parse.*dense"):
        store.save_run(training, report, name="Sparse", config={})
    assert store.read_runs() == []
    assert not list((store.path / "runs").glob("*/run.json"))


@pytest.mark.parametrize("container", ["series", "frame", "index", "columns", "attrs"])
def test_sparse_recovery_and_table_axes_fail_explicitly(container):
    values = sparse_series("float64")
    if container == "series":
        value = values
    elif container == "frame":
        value = values.to_frame()
    elif container == "index":
        value = pd.DataFrame({"value": [1, 2, 3]}, index=pd.Index(values.array))
    elif container == "columns":
        labels = pd.arrays.SparseArray(["a", "b"], fill_value="")
        value = pd.DataFrame([[1, 2]], columns=pd.Index(labels))
    else:
        value = pd.DataFrame({"value": [1, 2, 3]})
        value.attrs["sparse"] = values
    with pytest.raises(TypeError, match="[Ss]parse.*dense"):
        pack(value)
    if container in {"frame", "index", "columns"}:
        with pytest.raises(TypeError, match="[Ss]parse.*dense"):
            _table_document(value)


@dataclass(kw_only=True)
class SparseReporter(PredictionReporter):
    def run(self, context):
        return StudyResult("Sparse", tables={"values": sparse_series("float64").to_frame()})


class SparseHistoryAdapter(FixedAdapter):
    def fit(self, context):
        super().fit(context)
        self.training_history_ = sparse_series("float64").to_frame()
        return self


@pytest.mark.parametrize("artifact", ["report", "history", "output"])
def test_full_experiment_rejects_sparse_artifact_without_a_published_final(tmp_path, artifact):
    experiment = FootballExperiment("Sparse final", output_dir=tmp_path)
    preparation = prepared(holdout=True)
    if artifact == "output":
        preparation.outputs["sparse"] = sparse_series("float64")
    model = Candidate("Sparse history", SparseHistoryAdapter) if artifact == "history" else ridge()
    post = (PostTrainingAnalysis({"sparse": SparseReporter(type="overall", partition="score")})
            if artifact == "report" else None)
    with pytest.raises(TypeError, match="[Ss]parse.*dense"):
        experiment.run(preparation, model=model, post_analysis=post)
    assert experiment.store.read_runs(role="final") == []
    assert not list((experiment.store.path / "runs").glob("*/run.json"))
