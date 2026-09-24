"""Optional search on declared development rows, separated from final evaluation."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
from uuid import uuid4

import numpy as np
import pandas as pd

from ..analysis import PostTrainingAnalysis, PreTrainingAnalysis
from ..datasets import ModelDataset
from ..evaluation import Metric
from ..evaluation.metrics import _sample_hash
from ..experiments import configuration_hash
from ..reporting import AnalysisReport, Artifact, PerformanceReporter, StudyResult, StudyRun
from ..splits import Fold, SplitPlan
from ..training import TrainingResult, TrainingRunner
from ..training.runner import _positions
from .candidates import Candidate
from .criteria import MetricSelection, WeightedSelection, SelectionDecision, information_criteria as calculate_ic


_UNSET = object()


def _development(dataset, positions):
    if not isinstance(dataset, ModelDataset):
        raise TypeError("Model selection requires ModelDataset.")
    if not (dataset.X.index.equals(dataset.y.index) and dataset.X.index.equals(dataset.metadata.index)):
        raise ValueError("Dataset frames must retain their shared index/order.")
    rows = _positions(positions, len(dataset.X), "development_positions")
    codes, _ = pd.factorize(dataset.groups, sort=False)
    if np.isin(codes, codes[rows]).sum() != len(rows):
        raise ValueError("Development population must retain whole matches.")
    frames = [frame.iloc[rows].copy(deep=True).reset_index(drop=True) for frame in (dataset.X, dataset.y, dataset.metadata)]
    return replace(dataset, X=frames[0], y=frames[1], metadata=frames[2], definitions=deepcopy(dataset.definitions)), rows


def _restrict_plan(plan, rows, n):
    if not isinstance(plan, SplitPlan) or plan.n_rows != n or not plan.folds:
        raise ValueError("Supply a nonempty split plan indexed into the original dataset.")
    lookup = np.full(n, -1, dtype=int)
    lookup[rows] = np.arange(len(rows))

    def local(values, name, allow_empty=False):
        values = _positions(values, n, name, allow_empty=allow_empty)
        if (lookup[values] < 0).any():
            raise ValueError("Every selection fold must lie inside development_positions.")
        return lookup[values].copy()

    folds = [Fold(local(f.train, "train"), local(f.test, "test"), local(f.score, "score", True), deepcopy(f.metadata))
             for f in plan.folds]
    order = _positions(plan.row_order, n, "row_order")
    return SplitPlan(folds, len(rows), lookup[order[lookup[order] >= 0]], None), lookup


def _local_validation(validation, lookup):
    if validation is None or callable(getattr(validation, "select", None)):
        return deepcopy(validation)
    if isinstance(validation, dict):
        return {key: _local_validation(value, lookup) for key, value in validation.items()}
    values = _positions(validation, len(lookup), "validation", allow_empty=True)
    if (lookup[values] < 0).any():
        raise ValueError("Explicit validation positions must lie inside development_positions.")
    return lookup[values].copy()


def _fitted_objects(root):
    """Walk declared model/preprocessing links, including nested sklearn composites.

    Do not inspect arbitrary object attributes: custom adapters still own fresh
    hidden framework state. Strings such as 'passthrough' are not fitted objects.
    """
    pending, seen, objects = [root], set(), []
    while pending:
        value = pending.pop()
        if value is None or isinstance(value, (str, bytes)) or id(value) in seen:
            continue
        seen.add(id(value))
        objects.append(value)
        for name in ("estimator", "preprocessor", "backend_", "remainder",
                     "transformer", "transformer_", "regressor", "regressor_"):
            child = getattr(value, name, None)
            if child is not None:
                pending.append(child)
        for name in ("steps", "transformers", "transformers_", "transformer_list"):
            entries = getattr(value, name, ())
            if not isinstance(entries, (tuple, list)):
                continue
            for entry in entries:
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    pending.append(entry[1])
    return objects


def _validate_preparation(candidate):
    """Check declarative preparation scope before trial-failure recording."""
    if candidate.pre_analysis is not None:
        if not isinstance(candidate.pre_analysis, PreTrainingAnalysis):
            raise TypeError("Candidate pre_analysis must be PreTrainingAnalysis.")
        if any(reporter.partition != "train" for reporter in candidate.pre_analysis.reporters.values()):
            raise ValueError("Candidate preparation studies must use partition='train'; run exploration separately.")
    if candidate.features_from is not None:
        if candidate.pre_analysis is None or candidate.features_from not in candidate.pre_analysis.reporters:
            raise ValueError("features_from must name a candidate preparation study.")
        if candidate.feature_columns is not None:
            raise ValueError("Choose feature_columns or features_from, not both.")


class _FreshModels:
    def __init__(self):
        self.objects = []

    def factory(self, candidate):
        model_factory = candidate.model_factory
        def create():
            model = model_factory()
            objects = _fitted_objects(model)
            if any(obj is old for obj in objects for old in self.objects):
                raise ValueError("Candidates must create fresh adapters/estimators across trials and folds.")
            self.objects.extend(objects)
            from ..training import IterativeAdapter
            if isinstance(model, IterativeAdapter):
                original = model.backend_factory

                def backend_factory(seed):
                    backend = original(seed)
                    states = _fitted_objects(backend)
                    if any(item is old for item in states for old in self.objects):
                        raise ValueError("Iterative candidates must create fresh backends and model/preprocessing state.")
                    self.objects.extend(states)
                    return backend
                model.backend_factory = backend_factory
            return model
        return create


def _fit_candidate(dataset, plan, candidate, guard, execution=None):
    _validate_preparation(candidate)
    runner = TrainingRunner(guard.factory(candidate), candidate.feature_columns, candidate.target_columns,
                            candidate.control, candidate.validation, candidate.observer, execution)
    report = None
    if candidate.pre_analysis is not None:
        report = deepcopy(candidate.pre_analysis).run(dataset, split_plan=runner.selection_plan(dataset, plan))
    training = runner.run(dataset, plan, analysis_report=report if candidate.features_from is not None else None,
                          features_from=candidate.features_from)
    # Keep every fitted study, including descriptive ones that are not consumed
    # as a selector. Isolate them from the selected StudyRun kept by each fit.
    training.fitted_report = deepcopy(report)
    return training


def _original_positions(training, positions):
    """Restore public prediction and selection scope indexes after isolated fitting."""
    if training.fitted_report is not None:
        for study in training.fitted_report.studies:
            study.row_positions = positions[study.row_positions].copy()
            if study.scope is not None and "row_position" in study.scope:
                study.scope["row_position"] = positions[study.scope.row_position.to_numpy(dtype=int)]
    for fold in training.folds:
        for name in ("train_positions", "test_positions", "score_positions", "fit_positions", "validation_positions"):
            values = getattr(fold, name)
            if values is not None:
                setattr(fold, name, positions[values].copy())
        for frame in [fold.y_true, fold.metadata, *fold.predictions.values()]:
            frame.index = pd.Index(positions[frame.index.to_numpy(dtype=int)], name="row_position")
        if fold.selection is not None:
            fold.selection.row_positions = positions[fold.selection.row_positions].copy()
            if fold.selection.scope is not None and "row_position" in fold.selection.scope:
                fold.selection.scope["row_position"] = positions[fold.selection.scope.row_position.to_numpy(dtype=int)]
    return training


def _metric_records(report):
    records = []
    for study in report.studies:
        if "metrics" in study.result.tables:
            for metric in study.result.tables["metrics"].to_dict("records"):
                records.append(dict(metric, study=study.name, type=study.type, partition=study.partition,
                                    fold_id=study.fold_id, scope_label=study.scope_label, layout=study.layout))
    return records


def _fitted_evidence(candidate, training, dataset, use_ic):
    complexities, fits, signatures = [], [], []
    for fold in training.folds:
        measures = {} if candidate.complexity is None else dict(candidate.complexity(fold.model))
        if any(not isinstance(key, str) or not key or isinstance(value, bool) or not np.isfinite(value) or value < 0
               for key, value in measures.items()):
            raise ValueError("Complexity callbacks must supply named finite nonnegative numbers.")
        if use_ic:
            provider = candidate.fit_statistics
            if provider is None:
                provider = lambda model: model.fit_statistics()
            stats = provider(fold.model)
            criteria = calculate_ic(stats)
            if "n_parameters" in measures and measures["n_parameters"] != stats.n_parameters:
                raise ValueError("Complexity n_parameters disagrees with fit statistics.")
            measures["n_parameters"] = stats.n_parameters
            fit = fold.fit_positions
            truth = dataset.y[list(fold.target_columns)].iloc[fit].copy()
            meta = dataset.metadata.iloc[fit].copy()
            signature = configuration_hash([_sample_hash(truth[target], meta, pd.Series(True, index=truth.index))
                                            for target in truth])
            signatures.append(dict(fold_id=fold.fold_id, sample_hash=signature,
                                   likelihood_id=stats.likelihood_id, n_observations=int(stats.n_observations)))
            fits.append(dict(fold_id=fold.fold_id, log_likelihood=stats.log_likelihood,
                             n_parameters=stats.n_parameters, n_observations=stats.n_observations,
                             likelihood_id=stats.likelihood_id, **criteria))
        complexities.append(dict(fold_id=fold.fold_id, **measures))
    complexity_table = pd.DataFrame(complexities)
    # A missing measure in any fold stays missing, rather than silently averaging
    # only the folds whose adapters happened to provide it.
    aggregate = {name: float(complexity_table[name].mean(skipna=False))
                 for name in complexity_table if name != "fold_id"}
    fit_table = pd.DataFrame(fits)
    metrics = []
    if use_ic:
        for name in ("aic", "bic"):
            metrics.append(dict(metric=name, calculation=f"mean_fold_{name}", target="joint fitted targets",
                                output="fit_statistics", value=float(fit_table[name].mean()), direction="minimize",
                                n=len(fits), n_total=len(fits), n_missing=0, status="ok",
                                parameters=json.dumps({"aggregation": "arithmetic mean across fits",
                                                       "conventions": [s["likelihood_id"] for s in signatures],
                                                       "sample_counts": [s["n_observations"] for s in signatures]}, sort_keys=True),
                                sample_hash=configuration_hash(signatures)))
    return aggregate, complexity_table, fit_table, pd.DataFrame(metrics)


@dataclass
class TrialResult:
    """Candidate evidence. training indexes refer to the original supplied dataset."""
    trial_id: str
    candidate: Candidate
    record: dict
    training: object = None
    report: object = None
    complexity: dict = field(default_factory=dict)
    saved_run_id: str | None = None
    error: str | None = None


@dataclass
class SelectionResult:
    """Winner and internal trials, with no automatic full-data refit or final save."""
    trials: list[TrialResult]
    decision: SelectionDecision
    development_positions: np.ndarray
    n_rows: int
    run_group: str | None = None
    layout: str = "selection"
    n_matches: int = 0
    execution: object = None

    @property
    def winner(self):
        return next(trial for trial in self.trials if trial.trial_id == self.decision.winner_id)

    @property
    def comparison(self):
        return self.decision.table.copy(deep=True)

    def to_report(self):
        """Reusable optional internal-trial comparison; no diagnostic plot execution."""
        from .presentation import comparison_for_display
        table = comparison_for_display(self.comparison, {trial.trial_id: trial.candidate.config for trial in self.trials})
        study = StudyResult("Model selection", [Artifact("table", table, "Candidate comparison")],
                            {"comparison": table}, ["These are selection results, not an independent final evaluation."])
        scope = pd.DataFrame({"row_position": self.development_positions})
        return AnalysisReport([StudyRun("model_selection", "overall", "selection", None, self.layout,
                                       self.development_positions.copy(), self.n_matches, study, scope,
                                       "declared development population")], "Model selection")

    def evaluate(self, dataset, fold, *, validation=_UNSET, control=_UNSET, checkpoint_policy=None,
                 checkpoint_directory=None, checkpoint_namespace=None, execution=_UNSET):
        """Fit the winner freshly and return predictions for an untouched outer fold.

        This is ordinary outer evaluation training, not an optional extra deployment
        refit. It reruns fitted feature selection and preprocessing. Train must lie
        inside the search development population; test must be wholly outside it.
        Custom controls/validation can be explicitly replaced (including by None).
        For evaluating a different dataset, prepare a new evaluation plan upstream;
        this method uses the original dataset row positions and row count.
        """
        if len(dataset.X) != self.n_rows or not isinstance(fold, Fold):
            raise ValueError("Winner evaluation requires a Fold on the original dataset.")
        train = _positions(fold.train, self.n_rows, "outer train")
        test = _positions(fold.test, self.n_rows, "outer test")
        if not np.isin(train, self.development_positions).all() or np.isin(test, self.development_positions).any():
            raise ValueError("Outer training must be within development; outer test must be untouched by selection.")
        candidate = self.winner.candidate
        if validation is not _UNSET:
            candidate = replace(candidate, validation=validation)
        if control is not _UNSET:
            candidate = replace(candidate, control=control)
        if checkpoint_policy is not None:
            if checkpoint_directory is None or checkpoint_namespace is None:
                raise ValueError("Evaluation checkpointing requires a directory and execution namespace.")
            candidate = replace(candidate, model_factory=checkpoint_policy.wrap(
                candidate.model_factory, checkpoint_directory, f"{checkpoint_namespace}:evaluation"))
        guard = _FreshModels()
        for trial in self.trials:
            if trial.training is not None:
                for fitted in trial.training.folds:
                    guard.objects.extend(_fitted_objects(fitted.model))
        plan = SplitPlan([fold], self.n_rows, np.arange(self.n_rows))
        return _fit_candidate(dataset, plan, candidate, guard, self.execution if execution is _UNSET else execution)


@dataclass
class NestedSelectionResult:
    """Separate winner per outer fold and only outer predictions for reporting.

    There is no universal winner inferred from outer test scores. A deployment
    configuration may later be selected again on its declared development data.
    """
    selections: dict[int, SelectionResult]
    training: TrainingResult

    @property
    def comparison(self):
        return pd.concat([selection.comparison.assign(outer_fold=fold)
                          for fold, selection in self.selections.items()], ignore_index=True)


@dataclass
class ModelSelection:
    """Evaluate finite candidates only inside an explicit development population.

    candidates is an iterable of Candidate (including GridCandidates). Callable
    sources returning fresh iterables are supported for repeated/nested runs.
    No search occurs unless run/run_nested is invoked. metrics are registered
    names or Metric requests, computed per fold and on pooled score predictions;
    decision defaults to the sole metric, or must be supplied for multiple metrics.
    Repeated score rows require explicit pooling as in PostTrainingAnalysis.

    information_criteria=True requests adapter-supplied FitStatistics and records
    arithmetic mean per-fit AIC/BIC, along with every individual fit. It does not
    derive likelihood from prediction losses or pretend CV folds are one fit.
    on_error='raise' propagates candidate failures; 'record' keeps failed trials
    visible and proceeds. Declarative scope/preparation and final decision errors
    raise; exceptions from estimator fitting or numerical callbacks follow on_error.
    Store is optional; saved candidates are trial-role records, never final runs.
    evidence_reporters optionally adds score-scoped reporters whose metrics tables
    use the shared numerical schema (e.g. custom betting evidence). Only explicitly
    supplied studies run; ordinary search does not build post-training plots.
    Custom decision objects implement decide(trials)->SelectionDecision.
    """
    candidates: object
    metrics: object = ()
    decision: object = None
    pooling: str | None = None
    information_criteria: bool = False
    on_error: str = "raise"
    evidence_reporters: dict = field(default_factory=dict)

    def validate_evidence(self):
        """Resolve the decision and catch missing built-in metrics before fitting."""
        metrics = [self.metrics] if isinstance(self.metrics, (str, Metric)) else list(self.metrics)
        rule = self.decision
        if rule is None:
            if len(metrics) != 1:
                raise ValueError("Supply a decision rule unless selecting one metric.")
            request = metrics[0]
            rule = MetricSelection(request if isinstance(request, str) else request.key or request.name)
        available = {request if isinstance(request, str) else request.key or request.name for request in metrics}
        if self.information_criteria:
            available.update(("aic", "bic"))
        required = ({rule.metric} if isinstance(rule, MetricSelection) else
                    {key for key, weight in rule.weights.items() if weight > 0}
                    if isinstance(rule, WeightedSelection) else set())
        # Custom studies may emit metrics that are only known at execution time.
        if not self.evidence_reporters and required - available:
            missing = ", ".join(sorted(required - available))
            raise ValueError(f"Decision metric(s) {missing} are not among calculated metrics {sorted(available)}. "
                             "Add them to Metrics or change the Decision metric. No candidates were fitted.")
        return metrics, rule

    def _prediction_evidence(self, training, metrics):
        reporters = {} if not metrics else {
            "selection_metrics": PerformanceReporter(type="overall", partition="score", pooling=self.pooling, metrics=metrics),
            "fold_metrics": PerformanceReporter(type="per_fold", partition="score", metrics=metrics)}
        return PostTrainingAnalysis(reporters, title="Candidate numerical evidence").run(training)

    def run_nested(self, dataset, outer_plan, inner_plan_factory, *, experiment=None, name="Nested selection",
                   resume=False, checkpoint_policy=None, recovery_namespace=None, execution=None):
        """Optional outer CV; inner_plan_factory(development_dataset) -> SplitPlan.

        The factory receives only this outer fold's training population, indexed
        locally from zero. Its plan is mapped back to original positions before
        search. Selection is repeated independently per outer fold; the winner
        is fitted freshly on that outer train and predicts its untouched test.
        Pass the returned .training to final PostTrainingAnalysis. Per-outer
        search groups contain only trials; no final leaderboard entry is saved
        automatically. The caller can save one final nested evaluation record.
        """
        self.validate_evidence()
        if not isinstance(outer_plan, SplitPlan) or outer_plan.n_rows != len(dataset.X) or not outer_plan.folds:
            raise ValueError("Nested selection requires a nonempty outer plan for this dataset.")
        # Materialize finite one-shot sources once so every outer fold receives
        # the same candidate configurations. Callable sources are invoked afresh.
        source = self.candidates if callable(self.candidates) else list(self.candidates)
        shared_guard = _FreshModels()

        selections, results, fitted_studies = {}, [], []
        for outer_id, fold in enumerate(outer_plan.folds):
            development, positions = _development(dataset, fold.train)
            inner = inner_plan_factory(development)
            if not isinstance(inner, SplitPlan) or inner.n_rows != len(positions):
                raise ValueError("Inner factory must return a plan for its supplied development dataset.")

            def original(values, name, allow_empty=False):
                return positions[_positions(values, len(positions), name, allow_empty=allow_empty)]

            folds = [Fold(original(f.train, "inner train"), original(f.test, "inner test"),
                          original(f.score, "inner score", True), {**deepcopy(f.metadata), "outer_fold_id": outer_id})
                     for f in inner.folds]
            plan = SplitPlan(folds, len(dataset.X), original(inner.row_order, "inner row_order"))
            # Evaluate a callable source exactly once per outer fold. Recovery
            # identities and guarded factories must describe the same candidates.
            identity_candidates = list(source() if callable(source) else source)
            if any(not isinstance(candidate, Candidate) for candidate in identity_candidates):
                raise TypeError("Nested search requires Candidate objects.")
            search = replace(self, candidates=[
                replace(candidate, model_factory=shared_guard.factory(candidate))
                for candidate in identity_candidates])
            selection = search.run(dataset, plan, development_positions=positions,
                                   experiment=experiment, name=f"{name}: outer {outer_id}",
                                   resume=resume, checkpoint_policy=checkpoint_policy,
                                   _identity_candidates=identity_candidates,
                                   recovery_namespace=recovery_namespace, execution=execution)
            outer = Fold(fold.train.copy(), fold.test.copy(), fold.score.copy(),
                         {**deepcopy(fold.metadata), "outer_fold_id": outer_id})
            evaluated = selection.evaluate(dataset, outer, checkpoint_policy=checkpoint_policy,
                                           checkpoint_directory=None if experiment is None else experiment.path / "checkpoints",
                                           checkpoint_namespace=selection.run_group)
            fitted = evaluated.folds[0]
            if evaluated.fitted_report is not None:
                for study in evaluated.fitted_report.studies:
                    fitted_studies.append(replace(study, fold_id=outer_id, type="per_fold",
                                                 scope_label=f"{study.type} within outer fold {outer_id} fitting rows"))
            fitted.fold_id = outer_id
            if "fold_id" in fitted.training_history:
                fitted.training_history["fold_id"] = outer_id
            if fitted.selection is not None and fitted.selection.fold_id is not None:
                fitted.selection.fold_id = outer_id
            selections[outer_id], results = selection, [*results, fitted]
        training = TrainingResult(results, dataset.layout, tuple(dataset.identity_columns),
                                  tuple(dataset.match_columns), dataset.target_perspective, deepcopy(dataset.definitions))
        if fitted_studies:
            training.fitted_report = AnalysisReport(fitted_studies, "Fitted preparation studies")
        return NestedSelectionResult(selections, training)

    def run(self, dataset, split_plan, *, development_positions, experiment=None, run_group=None, name="Model selection",
            resume=False, checkpoint_policy=None, _identity_candidates=None, recovery_namespace=None, execution=None):
        metrics, rule = self.validate_evidence()
        development, rows = _development(dataset, development_positions)
        local_plan, lookup = _restrict_plan(split_plan, rows, len(dataset.X))
        codes, _ = pd.factorize(development.groups, sort=False)
        for fold in local_plan.folds:
            if np.intersect1d(fold.train, fold.test).size or not np.isin(fold.score, fold.test).all():
                raise ValueError("Selection train/test must be disjoint and score must be a test subset.")
            for positions in (fold.train, fold.test, fold.score):
                if np.isin(codes, codes[positions]).sum() != len(positions):
                    raise ValueError("Selection partitions must retain whole matches.")
        if self.on_error not in {"raise", "record"}:
            raise ValueError("on_error must be raise or record.")
        if not metrics and not self.information_criteria and not self.evidence_reporters:
            raise ValueError("Selection needs prediction metrics or explicit information criteria.")
        if set(self.evidence_reporters) & {"selection_metrics", "fold_metrics", "fit_evidence"}:
            raise ValueError("Evidence reporter names overlap reserved selection studies.")
        if any(reporter.partition != "score" for reporter in self.evidence_reporters.values()):
            raise ValueError("Additional selection evidence must use partition='score'.")
        if experiment is None and run_group is not None:
            raise ValueError("run_group requires an experiment store.")
        if (resume or checkpoint_policy is not None) and experiment is None:
            raise ValueError("Search recovery/checkpointing requires an experiment store.")
        source = list(self.candidates() if callable(self.candidates) else self.candidates)
        identity_candidates = source if _identity_candidates is None else list(_identity_candidates)
        search_key = None
        if experiment is not None:
            from ..experiments.recovery import execution_key
            search_key = execution_key(dataset, split_plan, rows, metrics, rule, self.pooling,
                                       self.information_criteria, self.evidence_reporters, recovery_namespace, execution)
        if experiment is not None and run_group is None:
            run_group = (experiment.open_run(name, execution_key(search_key, identity_candidates), reuse=resume)
                         if resume or checkpoint_policy is not None else experiment.start_run(name, config={"development_positions": rows.tolist()}))
        trials, names, guard = [], set(), _FreshModels()
        for candidate_index, candidate in enumerate(source):
            if not isinstance(candidate, Candidate) or candidate.name in names:
                raise ValueError("Supply Candidate objects with distinct names.")
            _validate_preparation(candidate)
            names.add(candidate.name)
            trial_id = str(uuid4())
            config = deepcopy(candidate.config)
            record = dict(run_id=trial_id, name=candidate.name, config=config, config_hash=configuration_hash(config),
                          status="complete", metrics=[])
            trial = TrialResult(trial_id, candidate, record)
            local = replace(candidate, validation=_local_validation(candidate.validation, lookup))
            # Fitting identity excludes how retained predictions are scored/ranked.
            # Include data, scopes, fitted preparation and custom fitted evidence.
            trial_key = (execution_key("candidate-evidence-v2", dataset, split_plan, rows,
                                       identity_candidates[candidate_index], self.information_criteria,
                                       self.evidence_reporters, execution) if experiment is not None else None)
            if resume:
                saved = experiment.find_completed(trial_key, role="trial")
                if saved is not None:
                    loaded = experiment.load_run(saved["run_id"])
                    report = self._prediction_evidence(loaded["training"], metrics)
                    report.studies.extend(study for study in loaded["report"].studies
                                          if study.name not in {"selection_metrics", "fold_metrics"})
                    record = deepcopy(saved)
                    record["metrics"] = _metric_records(report)
                    trials.append(TrialResult(saved["run_id"], candidate, record,
                                              loaded["training"], report, loaded["extra"]["complexity"],
                                              saved["run_id"]))
                    continue
            if checkpoint_policy is not None:
                local = replace(local, model_factory=checkpoint_policy.wrap(
                    local.model_factory, experiment.path / "checkpoints", f"{run_group}:{trial_key}"))
            try:
                training = _fit_candidate(development, local_plan, local, guard, execution)
                trial.training = _original_positions(training, rows)
                reporters = {} if not metrics else {
                    "selection_metrics": PerformanceReporter(type="overall", partition="score", pooling=self.pooling, metrics=metrics),
                    "fold_metrics": PerformanceReporter(type="per_fold", partition="score", metrics=metrics)}
                reporters.update(deepcopy(self.evidence_reporters))
                report = PostTrainingAnalysis(reporters, title="Candidate numerical evidence").run(trial.training)
                aggregate, complexities, fits, ic = _fitted_evidence(candidate, trial.training, dataset, self.information_criteria)
                trial.complexity = aggregate
                evidence = StudyResult("Fitted model evidence", tables={"complexity": complexities, "fit_statistics": fits})
                if len(ic):
                    evidence.tables["metrics"] = ic
                report.studies.append(StudyRun("fit_evidence", "overall", "model", None, dataset.layout,
                                               np.array([], dtype=int), 0, evidence, scope_label="mean across candidate fits"))
                trial.report = report
                trial.record["metrics"] = _metric_records(report)
            except Exception as error:
                trial.error = f"{type(error).__name__}: {error}"
                trial.record.update(status="failed", error=trial.error, metrics=[])
                if experiment is not None:
                    saved = experiment.save_failure(name=candidate.name, config=config, error=trial.error, role="trial", run_group=run_group)
                    trial.saved_run_id = saved["run_id"]
                error.add_note(f"Model selection candidate {candidate.name!r}.")
                if self.on_error == "raise":
                    raise
            else:
                if experiment is not None:
                    saved = experiment.save_run(trial.training, trial.report, name=candidate.name, config=config,
                                                role="trial", run_group=run_group,
                                                recovery={"complexity": trial.complexity},
                                                recovery_key=trial_key)
                    trial.saved_run_id = saved["run_id"]
            trials.append(trial)
        if not trials:
            raise ValueError("Candidate source produced no candidates.")
        decision = rule.decide(trials)
        if not isinstance(decision, SelectionDecision) or not any(
                trial.trial_id == decision.winner_id and trial.record["status"] == "complete" for trial in trials):
            raise ValueError("Decision rule must select a successful trial and return SelectionDecision.")
        if not isinstance(decision.table, pd.DataFrame) or "run_id" not in decision.table:
            raise TypeError("Decision comparison must be a DataFrame containing trial run_id values.")
        decision.table["saved_run_id"] = decision.table.run_id.map({trial.trial_id: trial.saved_run_id for trial in trials})
        decision.table["error"] = decision.table.run_id.map({trial.trial_id: trial.error for trial in trials})
        return SelectionResult(trials, decision, rows, len(dataset.X), run_group, dataset.layout,
                               len(development.metadata[list(dataset.match_columns)].drop_duplicates()), execution)
