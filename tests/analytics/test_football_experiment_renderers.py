"""Custom rendering survives publication and explicit data-only recovery."""
import pytest

from football_experiment_samples import prepared, ridge
from test_football_experiment_search import CountingFactory
from xdiyo_analytics.analysis import PostTrainingAnalysis, PreTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment, ExperimentStore
from xdiyo_analytics.reporting import Artifact, StudyResult
from xdiyo_analytics.selection import Candidate


class CustomReporter:
    type = "overall"
    partition = "score"
    supported_types = ("overall",)

    def cache_key(self):
        return {"type": self.type, "partition": self.partition}

    def run(self, context):
        return StudyResult("Custom study", [Artifact("badge", {"value": 17}, "Custom badge")])


class BadgeRenderer:
    def __init__(self, label="BADGE_A"):
        self.label = label
        self.calls = 0

    def cache_key(self):
        return {"label": self.label}

    def __call__(self, artifact):
        self.calls += 1
        return f"<strong>{self.label}: {artifact.data['value']}</strong>"


def report():
    return PostTrainingAnalysis({"Custom": CustomReporter()})


def test_custom_renderer_publication_display_reuse_and_explicit_reload(tmp_path, monkeypatch):
    renderer = BadgeRenderer()
    mapping = {"badge": renderer}
    fits = []
    candidate = Candidate("Counted", CountingFactory(.1, fits))
    experiment = FootballExperiment("Custom renderers", output_dir=tmp_path)
    data = prepared(holdout=True)
    result = experiment.run(data, model=candidate, post_analysis=report(), renderers=mapping)
    assert len(fits) == 1 and renderer.calls == 1
    saved_html = (result.path / "report.html").read_text(encoding="utf-8")
    assert "BADGE_A: 17" in saved_html
    mapping.clear()  # The result retains its own renderer mapping.
    assert "BADGE_A: 17" in result.to_html()
    assert "BADGE_A: 17" in result.to_notebook()._repr_html_()
    assert "BADGE_A: 17" in result._repr_html_()
    displays = []
    monkeypatch.setattr("IPython.display.display", displays.append)
    result.show()
    assert "BADGE_A: 17" in displays[0]._repr_html_()
    assert "BADGE_OVERRIDE: 17" in result.to_html(renderers={"badge": BadgeRenderer("BADGE_OVERRIDE")})
    calls = renderer.calls
    reused = experiment.run(data, model=candidate, post_analysis=report(), renderers={"badge": renderer})
    assert reused.reused and reused.record["run_id"] == result.record["run_id"]
    assert len(fits) == 1 and renderer.calls == calls
    assert "BADGE_A: 17" in reused.to_html()
    loaded = experiment.load(result.record["run_id"], renderers={"badge": renderer})
    assert "BADGE_A: 17" in loaded.to_html() and len(fits) == 1
    plain = experiment.load(result.record["run_id"])
    assert plain.post_report.studies[0].result.artifacts[0].data == {"value": 17}
    with pytest.raises(ValueError, match="No renderer"):
        plain.to_html()
    assert "BADGE_A: 17" in plain.to_html(renderers={"badge": renderer})
    assert "BadgeRenderer" not in (result.path / "recovery.json").read_text(encoding="utf-8")
    renderer.label = "BADGE_B"
    changed = experiment.run(data, model=candidate, post_analysis=report(), renderers={"badge": renderer})
    assert not changed.reused and changed.record["run_id"] != result.record["run_id"]
    assert "BADGE_B: 17" in (changed.path / "report.html").read_text(encoding="utf-8")
    assert (result.path / "report.html").read_text(encoding="utf-8") == saved_html


@pytest.mark.parametrize("scope", ["pre", "post", "experiment"])
def test_renderers_reach_each_published_report_scope(tmp_path, scope):
    custom = CustomReporter()
    custom.partition = "experiment" if scope == "experiment" else "all" if scope == "pre" else "score"
    kwargs = {"pre_analysis": PreTrainingAnalysis({"Custom": custom})} if scope == "pre" else {
        "post_analysis": PostTrainingAnalysis({"Custom": custom})}
    result = FootballExperiment(scope, output_dir=tmp_path).run(
        prepared(holdout=True), model=ridge(), renderers={"badge": BadgeRenderer()}, **kwargs)
    assert "BADGE_A: 17" in (result.path / "report.html").read_text(encoding="utf-8")
    assert result.report.studies[0].result.artifacts[0].kind == "badge"


def test_direct_store_supports_custom_renderers_and_failure_stays_unpublished(tmp_path):
    experiment = FootballExperiment("Source", output_dir=tmp_path)
    result = experiment.run(prepared(holdout=True), model=ridge())
    custom = report().run(result.training)
    store = ExperimentStore(tmp_path, "Direct")
    record = store.save_run(result.training, custom, name="Custom", config={}, save_html=True,
                            renderers={"badge": BadgeRenderer()})
    assert "BADGE_A: 17" in (store.path / "runs" / record["run_id"] / "report.html").read_text(encoding="utf-8")
    def broken(artifact):
        raise RuntimeError("renderer failed")
    failing = FootballExperiment("Failure", output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="renderer failed"):
        failing.run(prepared(holdout=True), model=ridge(), post_analysis=report(), renderers={"badge": broken})
    assert failing.store.read_runs() == []
    assert list((failing.store.path / "runs").glob("*/run.json")) == []
