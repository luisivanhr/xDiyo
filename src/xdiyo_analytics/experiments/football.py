"""A small public workflow joining preparation, selection, training and reports."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import PreTrainingAnalysis, PostTrainingAnalysis
from ..datasets import ModelDataset
from ..reporting import AnalysisReport, Artifact, StudyResult, StudyRun
from ..selection import Candidate, ModelSelection
from ..selection.core import _fit_candidate, _FreshModels
from ..splits import Fold, SplitPlan
from ..training import TrainingRunner, refit_model
from .recovery import execution_key, signature
from .store import ExperimentStore


@dataclass
class PreparedExperiment:
    """Inspectable model inputs and folds; outputs may retain histories/ratings.

    A preparation callable performs loading, history/rating construction, feature
    and label evaluation, assembly and splitting using existing public functions.
    outputs must be data-only if they are to be retained in recovery bundles.
    """
    dataset: ModelDataset
    split_plan: SplitPlan
    outputs: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.dataset, ModelDataset) or not isinstance(self.split_plan, SplitPlan):
            raise TypeError("PreparedExperiment needs ModelDataset and SplitPlan.")
        if self.split_plan.n_rows != len(self.dataset.X) or not self.split_plan.folds:
            raise ValueError("Prepared splits must refer to this dataset and contain folds.")


@dataclass
class RefitPolicy:
    """Optional extra deployment fit, with explicitly selected training rows.

    candidate=None uses the fixed model or single holdout-search winner. Nested
    CV requires an explicit candidate, because its outer folds can select different
    configurations. Fitted feature selection is rerun on refit fitting rows.
    validation/control default to None, independently of evaluation controls.
    FootballExperiment records the resolved refit settings in config['refit'].
    """
    train_positions: object
    candidate: object = None
    validation: object = None
    control: object = None

    def run(self, dataset, candidate, *, execution=None):
        candidate = self.candidate or candidate
        if not isinstance(candidate, Candidate):
            raise ValueError("Refitting needs an explicit Candidate after nested selection.")
        features = candidate.feature_columns
        weights = None
        if candidate.features_from is not None or candidate.weights_from is not None:
            from ..selection.core import _validate_preparation
            _validate_preparation(candidate)
            rows = np.asarray(self.train_positions)
            plan = SplitPlan([Fold(rows, np.array([], dtype=int), np.array([], dtype=int))],
                             len(dataset.X), np.arange(len(dataset.X)))
            runner = TrainingRunner(candidate.model_factory, validation=self.validation, calibration=candidate.calibration)
            report = deepcopy(candidate.pre_analysis).run(dataset, split_plan=runner.selection_plan(dataset, plan))
            if candidate.features_from is not None:
                studies = [study for study in report.studies if study.name == candidate.features_from]
                if len(studies) != 1 or studies[0].result.selection is None:
                    raise ValueError("Refit selector must return one selection on refit fitting rows.")
                features = studies[0].result.selection.columns
            if candidate.weights_from is not None:
                from ..weighting import report_weights
                fitting_rows = runner.selection_plan(dataset, plan).folds[0].train
                weights = report_weights(report, candidate.weights_from, 0, fitting_rows, dataset.layout)
        return refit_model(dataset, candidate.model_factory, train_positions=self.train_positions,
                           feature_columns=features, target_columns=candidate.target_columns,
                           validation=self.validation, control=self.control, observer=candidate.observer, execution=execution,
                           calibration=candidate.calibration, observation_weights=weights)


@dataclass
class SelectionSummary:
    """Reloadable selection evidence; deliberately contains no executable factories."""
    comparison: pd.DataFrame
    winners: dict


@dataclass
class ExperimentResult:
    """Inspectable outputs; notebook display never fits or predicts.

    Recovery restores saved fold/refit models by default. load_models=False opens
    data-only results; older runs may contain no saved models. selection is a
    live selection result on execution and SelectionSummary on explicit reload.
    """
    prepared: PreparedExperiment
    training: object
    pre_report: AnalysisReport
    post_report: AnalysisReport
    selection: object = None
    refit: object = None
    record: dict = field(default_factory=dict)
    path: object = None
    reused: bool = False
    _renderers: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def dataset(self):
        return None if self.prepared is None else self.prepared.dataset

    @property
    def splits(self):
        return None if self.prepared is None else self.prepared.split_plan

    @property
    def report(self):
        if self.prepared is None and not self.pre_report.studies:
            return self.post_report
        # Fitted studies belong to the evaluated candidate(s), not every search trial.
        fitted_report = getattr(self.training, "fitted_report", None) or AnalysisReport()
        # Prefix study keys, preserving distinct exploratory/fitted/post names.
        studies = [replace(study, name=f"{prefix}/{study.name}")
                   for prefix, report in (("pre", self.pre_report), ("fitted", fitted_report), ("post", self.post_report))
                   for study in report.studies]
        if self.selection is not None:
            summary = self.selection if isinstance(self.selection, SelectionSummary) else _summary(self.selection)
            study = StudyResult("Model selection", [Artifact("table", summary.comparison, "Candidate comparison")],
                                {"comparison": summary.comparison},
                                ["Internal selection evidence; final evaluation appears in the post-training studies."])
            studies.insert(len(self.pre_report.studies), StudyRun(
                "selection/comparison", "overall", "selection", None, self.training.layout,
                np.array([], dtype=int), 0, study, scope_label="internal candidate comparisons"))
        return AnalysisReport(studies, self.record.get("name", "Football experiment"))

    def show(self, **kwargs):
        kwargs.setdefault("renderers", self._renderers)
        return self.report.show(**kwargs)

    def to_html(self, path=None, **kwargs):
        kwargs.setdefault("renderers", self._renderers)
        return self.report.to_html(path, **kwargs)

    def to_notebook(self, **kwargs):
        kwargs.setdefault("renderers", self._renderers)
        return self.report.to_notebook(**kwargs)

    def _repr_html_(self):
        return self.to_notebook()._repr_html_()


def _display_name(candidate, fields=None):
    if candidate is None:
        return "Tuned per fold"
    config = {**candidate.config, **candidate.config.get("grid_parameters", {})}
    keys = list(fields) if fields is not None else [key for key, value in config.items()
                                                   if isinstance(value, (str, bool, int, float)) or value is None][:6]
    parts = candidate.name.split(" · ")
    for key in keys:
        value = config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"Display parameter {key!r} is absent from candidate config.")
            value = value[part]
        parameter = f"{key}={value}"
        if parameter not in parts:
            parts.append(parameter)
    return " · ".join(parts)


def _summary(selection):
    if selection is None:
        return None
    if hasattr(selection, "selections"):
        winners = {str(fold): {"name": result.winner.candidate.name, "config": result.winner.candidate.config,
                               "saved_trial_id": result.winner.saved_run_id}
                   for fold, result in selection.selections.items()}
    else:
        winners = {"holdout": {"name": selection.winner.candidate.name, "config": selection.winner.candidate.config,
                               "saved_trial_id": selection.winner.saved_run_id}}
    from ..selection.presentation import comparison_for_display
    parts = selection.selections.values() if hasattr(selection, "selections") else [selection]
    configs = {trial.trial_id: trial.candidate.config for part in parts for trial in part.trials}
    return SelectionSummary(comparison_for_display(selection.comparison, configs), deepcopy(winners))


class _ExperimentRecords:
    """Read-only reporting snapshot, optionally including an unpublished final.

    Exposes record inspection only: the provisional run has no loadable artifacts.
    Copies isolate reporters from each other and from the record being published.
    """
    def __init__(self, store, pending=None):
        self._path = store.path
        self._manifest = deepcopy(store.manifest)
        records = store.read_runs()
        if pending is not None:
            records = [record for record in records if record["run_id"] != pending["run_id"]]
            records.append(pending)
        self._records = deepcopy(sorted(records, key=lambda item: (item["created_at"], item["run_id"])))

    @property
    def path(self):
        return self._path

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def read_runs(self, *, role=None, run_group=None):
        if role not in {None, "final", "trial"}:
            raise ValueError("role must be final, trial or None.")
        return deepcopy([record for record in self._records
                         if (role is None or record.get("role", "final") == role)
                         and (run_group is None or record.get("run_group", record["run_id"]) == run_group)])


class FootballExperiment:
    """One named experiment folder, containing independently named final runs.

    prepare is a callable returning PreparedExperiment. run(prepared) can also
    accept an independently constructed preparation. reuse=True restores an exact
    completed result or recovers completed search trials after interruption;
    reuse=False starts a fresh execution group. No user-supplied digest is needed.

    Identity covers prepared data/order/definitions, folds, model/search settings,
    pre-training reports, refit policy, explicit config and numerical library source.
    Post-training analysis is refreshed independently of fitting. Custom external
    dependencies must be described in config or their object's cache_key().
    The preparation callable still runs for run() to discover current inputs;
    load(run_id) reopens saved outputs without loading source data or executing it.
    Calls against the same experiment must be serialized.
    """
    def __init__(self, name, *, output_dir="experiments", prepare=None, config=None):
        self.name, self._prepare = name, prepare
        self.config = dict(config or {})
        self.store = ExperimentStore(output_dir, name)

    def prepare(self):
        if not callable(self._prepare):
            raise ValueError("Supply a preparation callable or pass PreparedExperiment to run().")
        result = self._prepare()
        if not isinstance(result, PreparedExperiment):
            raise TypeError("Preparation must return PreparedExperiment.")
        return result

    def load(self, run_id, *, load_models=True, model_serializer=None, renderers=None):
        """Reopen a trusted local run without preparation, fitting or prediction.

        Models and fitted scalers are restored when saved. Set load_models=False
        for data-only inspection; legacy runs cannot restore models never saved.
        """
        loaded = self.store.load_run(run_id, load_models=load_models, model_serializer=model_serializer)
        extra = loaded["extra"]
        if extra.get("kind") == "legacy":
            result = ExperimentResult(None, loaded["training"], AnalysisReport(), loaded["report"],
                                      record=loaded["record"], path=self.store.path / "runs" / run_id, reused=True)
        elif extra.get("kind") == "football_experiment":
            result = ExperimentResult(extra["prepared"], loaded["training"], extra["pre_report"], loaded["report"],
                                      extra["selection"], extra["refit"], loaded["record"],
                                      self.store.path / "runs" / run_id, True)
        else:
            raise ValueError("This record is not a FootballExperiment final result.")
        result._renderers = dict(renderers or {})
        return result

    def leaderboard(self, weights, **kwargs):
        """Read current final runs from this experiment, including earlier sessions."""
        from ..reporting import ExperimentLeaderboardReporter
        return PostTrainingAnalysis({"leaderboard": ExperimentLeaderboardReporter(weights=weights, **kwargs)}).run(
            experiment=self.store)

    def run(self, prepared=None, *, model=None, model_selection=None, selection_plan=None,
            development_positions=None, inner_plan_factory=None, pre_analysis=None, post_analysis=None,
            refit_policy=None, checkpoint_policy=None, name=None, name_fields=None, config=None, reuse=True, execution=None,
            save_models=True, model_serializer=None, renderers=None):
        """Execute fixed fitting, holdout search, or optional nested selection.

        model is a Candidate. Alternatively model_selection is ModelSelection:
        selection_plan supplies inner folds on original dataset positions for ONE
        final holdout fold; development defaults to that final fold's train rows.
        inner_plan_factory selects nested CV over every prepared outer fold and
        receives the outer training subset with local indexes, as in run_nested.

        pre_analysis is descriptive. Learned preparation belongs to Candidate's
        pre_analysis/features_from and runs within the actual fitting populations.
        Post-analysis refreshes on reuse without fitting or predicting. Fitted
        models/scalers are saved by default using joblib; custom frameworks can
        supply model_serializer. Load only your own or otherwise trusted artifacts.
        Experiment reporters receive a read-only snapshot before publication.
        Custom renderers are supplied again on load and are not serialized.
        """
        prepared = self.prepare() if prepared is None else prepared
        if not isinstance(prepared, PreparedExperiment):
            raise TypeError("run needs PreparedExperiment.")
        if (model is None) == (model_selection is None):
            raise ValueError("Supply exactly one model Candidate or ModelSelection.")
        if model is not None and not isinstance(model, Candidate):
            raise TypeError("model must be Candidate.")
        if model_selection is not None and not isinstance(model_selection, ModelSelection):
            raise TypeError("model_selection must be ModelSelection.")
        if model is not None and any(value is not None for value in (selection_plan, development_positions, inner_plan_factory)):
            raise ValueError("Selection scopes require model_selection.")
        nested = inner_plan_factory is not None
        if nested and refit_policy is not None and not isinstance(refit_policy.candidate, Candidate):
            raise ValueError("Refitting needs an explicit Candidate after nested selection.")
        if model_selection is not None:
            model_selection.validate_evidence()
            if nested and (selection_plan is not None or development_positions is not None):
                raise ValueError("Choose nested inner_plan_factory or a holdout selection_plan.")
            if not nested and (selection_plan is None or len(prepared.split_plan.folds) != 1):
                raise ValueError("Holdout search needs an inner selection_plan and one prepared outer fold.")
            if not nested:
                outer = prepared.split_plan.folds[0]
                development = outer.train if development_positions is None else development_positions
                if np.isin(outer.test, development).any() or not np.isin(outer.train, development).all():
                    raise ValueError("Outer test must be outside selection development; outer train must be inside it.")
            source = model_selection.candidates
            # Nested selection calls a source once per outer fold. Preserve that
            # callable, but snapshot one-shot iterables for recovery identity.
            if not nested or not callable(source):
                model_selection = replace(model_selection, candidates=list(source() if callable(source) else source))
        pre_analysis = pre_analysis or PreTrainingAnalysis()
        post_analysis = post_analysis or PostTrainingAnalysis()
        renderers = dict(renderers or {})
        settings = {"experiment": self.config, "preparation": prepared.config, "run": dict(config or {})}
        if execution is not None:
            from ..training.execution import ExecutionPolicy
            if not isinstance(execution, ExecutionPolicy):
                raise TypeError("execution must be ExecutionPolicy or None.")
            settings["execution"] = execution
        identity_settings = deepcopy(settings)
        # The UI retains the full recipe for inspection/export. Its post-analysis
        # settings must not leak into the identity of the numerical fit.
        recipe = identity_settings.get("preparation", {}).get("recipe")
        if isinstance(recipe, dict):
            recipe.pop("post_reporters", None)
            recipe.pop("artifact_export", None)
            recipe.get("analysis_options", {}).pop("post", None)
            if not recipe.get("analysis_options"):
                recipe.pop("analysis_options", None)
        key = execution_key(prepared.dataset, prepared.split_plan, identity_settings, model, model_selection,
                            selection_plan, development_positions, inner_plan_factory, pre_analysis,
                            refit_policy, checkpoint_policy, name, name_fields, save_models, model_serializer, renderers)
        settings["post_analysis"] = signature(post_analysis)
        if reuse:
            record = self.store.find_completed(key)
            if record is not None:
                result = self.load(record["run_id"], model_serializer=model_serializer, renderers=renderers)
                if record.get("config", {}).get("post_analysis") != settings["post_analysis"]:
                    result.post_report = self._numerical_reports(post_analysis).run(
                        result.training, resources=result.prepared.outputs)
                    self._experiment_reports(result, post_analysis)
                    result.record = self.store.refresh_report(result.record["run_id"], result.post_report,
                        display_report=result.report, renderers=renderers, analysis_signature=settings["post_analysis"])
                self._experiment_reports(result, post_analysis)
                return result
        group = self.store.open_run(name or "Football run", key, reuse=reuse)
        dataset, plan = prepared.dataset, prepared.split_plan
        pre_report = pre_analysis.run(dataset, split_plan=plan)
        selection, chosen = None, model
        if model is not None:
            fitting = model
            if checkpoint_policy is not None:
                fitting = replace(model, model_factory=checkpoint_policy.wrap(
                    model.model_factory, self.store.path / "checkpoints", f"{group}:evaluation"))
            training = _fit_candidate(dataset, plan, fitting, _FreshModels(), execution)
        elif nested:
            selection = model_selection.run_nested(dataset, plan, inner_plan_factory, experiment=self.store,
                                                   resume=reuse, checkpoint_policy=checkpoint_policy, recovery_namespace=group,
                                                   name=f"{self.name}: {group}", execution=execution)
            training = selection.training
        else:
            development = plan.folds[0].train if development_positions is None else development_positions
            selection = model_selection.run(dataset, selection_plan, development_positions=development,
                                             experiment=self.store, run_group=group, resume=reuse,
                                             checkpoint_policy=checkpoint_policy, execution=execution)
            chosen = selection.winner.candidate
            training = selection.evaluate(dataset, plan.folds[0], checkpoint_policy=checkpoint_policy,
                                          checkpoint_directory=self.store.path / "checkpoints", checkpoint_namespace=group)
        numerical = self._numerical_reports(post_analysis)
        post_report = numerical.run(training, resources=prepared.outputs)
        refit = None
        settings["refit"] = None
        if refit_policy is not None:
            policy = refit_policy
            refit_candidate = policy.candidate or chosen
            if isinstance(refit_candidate, Candidate):
                # Describe the resolved deployment candidate before checkpoint
                # wrapping, using policy controls rather than evaluation controls.
                settings["refit"] = {
                    "candidate": {"name": refit_candidate.name, "config": deepcopy(refit_candidate.config)},
                    "policy": signature(type(policy)),
                    "train_positions": np.asarray(policy.train_positions).tolist(),
                    "validation": signature(policy.validation), "control": signature(policy.control),
                    "fitting": signature({field: getattr(refit_candidate, field) for field in
                                          ("model_factory", "feature_columns", "target_columns",
                                           "features_from", "pre_analysis")}),
                }
            if checkpoint_policy is not None and refit_candidate is not None:
                refit_candidate = replace(refit_candidate, model_factory=checkpoint_policy.wrap(
                    refit_candidate.model_factory, self.store.path / "checkpoints", f"{group}:refit"))
                policy = replace(policy, candidate=refit_candidate)
            refit = policy.run(dataset, refit_candidate, execution=execution)
        display_name = name or _display_name(chosen, name_fields)
        summary = _summary(selection)
        settings["model"] = deepcopy(chosen.config) if chosen is not None else None
        settings["selection"] = None if summary is None else summary.winners
        candidates = [chosen] if chosen is not None else [part.winner.candidate for part in selection.selections.values()]
        settings["fitting"] = [signature({field: getattr(candidate, field) for field in
                                          ("feature_columns", "target_columns", "features_from", "weights_from", "pre_analysis",
                                           "validation", "control", "calibration")}) for candidate in candidates]
        settings["definitions"] = signature(dataset.definitions)
        result = ExperimentResult(prepared, training, pre_report, post_report, selection, refit, {"name": display_name})
        result._renderers = renderers
        selected_trial_id = None
        if selection is not None and not nested:
            saved_id = selection.winner.saved_run_id
            # Rescoring may recover a trial from an earlier execution group. Its
            # provenance stays in summary.winners; direct links are group-local.
            if any(record["run_id"] == saved_id and record["status"] == "complete"
                   for record in self.store.read_runs(role="trial", run_group=group)):
                selected_trial_id = saved_id
        def finalize_report(record):
            self._experiment_reports(result, post_analysis, experiment=_ExperimentRecords(self.store, record))
            return result.report

        record = self.store.save_run(training, post_report, name=display_name, config=settings,
                                     save_html=True, renderers=renderers, run_group=group,
                                     selected_trial_id=selected_trial_id,
                                     recovery_key=key, recovery={"kind": "football_experiment", "prepared": prepared,
                                                               "pre_report": pre_report, "selection": summary, "refit": refit},
                                     display_report=result.report, save_models=save_models,
                                     model_serializer=model_serializer, _finalize_report=finalize_report)
        result.record, result.path = record, self.store.path / "runs" / record["run_id"]
        return result

    @staticmethod
    def _numerical_reports(post_analysis):
        return PostTrainingAnalysis({key: reporter for key, reporter in post_analysis.reporters.items()
                                     if reporter.partition != "experiment"}, post_analysis.title,
                                    fold_ids=post_analysis.fold_ids)

    def _experiment_reports(self, result, post_analysis, *, experiment=None):
        reporters = {key: reporter for key, reporter in post_analysis.reporters.items() if reporter.partition == "experiment"}
        if reporters:
            report = PostTrainingAnalysis(reporters, post_analysis.title).run(
                result.training, experiment=experiment if experiment is not None else _ExperimentRecords(self.store))
            result.post_report.studies[:] = [study for study in result.post_report.studies
                                            if study.partition != "experiment"] + report.studies
