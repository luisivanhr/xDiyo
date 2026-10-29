"""Deepthroat-gg: explicit retrospective tuning and separate chronological scoring.

Research interpretation of DeepFoot-KNN (Berrar, Lopes, Dubitzky, 2024).
This is not the paper's jointly optimized DeepFoot-KNN2.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..splits import Fold, SplitPlan
from ..training import EstimatorAdapter, TrainingRunner, save_model
from .deepthroat_features import FEATURES, POLICY, prepare_bundles, windows

CLASS_ORDER = (1, 0, -1)
NEIGHBOR_POLICY = 'native_sklearn_selected_k_leave_one_out_v2'


def joint_score_rmse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 2 or not len(actual):
        raise ValueError('Joint score RMSE requires matching nonempty N by 2 arrays.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Score arrays must be finite.')
    return float(np.sqrt(np.mean(np.sum((actual-predicted)**2, axis=1))))


def rps(actual, probabilities):
    actual, probabilities = np.asarray(actual), np.asarray(probabilities, float)
    if actual.ndim != 1 or not len(actual) or probabilities.shape != (len(actual), 3):
        raise ValueError('RPS requires N outcomes and N by 3 probabilities in H/D/A order.')
    if not np.isin(actual, CLASS_ORDER).all() or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-12):
        raise ValueError('Invalid outcomes or probability distributions.')
    observed = actual[:, None] == np.array(CLASS_ORDER)
    return float(np.mean(np.sum((np.cumsum(probabilities, axis=1)[:, :2]-np.cumsum(observed, axis=1)[:, :2])**2, axis=1)/2))


class OutcomeAdapter(EstimatorAdapter):
    """Native estimator adapter with fixed H/D/A schema; absent classes get zero."""

    def predict(self, context):
        result = super().predict(context)
        if 'predict_proba' in result:
            columns = pd.MultiIndex.from_product([self.target_columns_, CLASS_ORDER], names=['target', 'class'])
            result['predict_proba'] = result['predict_proba'].reindex(columns=columns, fill_value=0.)
        return result


def select_neighbors(X, goals, outcomes, k_values=range(3, 351)):
    """Score the same sklearn predictors used for final fitting/persistence.

    Fit each held-out population once, preserving its original row order, and
    reuse its KD trees across k. Query each actual k: maximum-k prefixes and
    deleting self from a full-population tree do not preserve distance ties.
    The returned diagnostic indices describe the maximum feasible k only.
    """
    ks = windows(k_values, 3)
    X, goals, outcomes = np.asarray(X, float), np.asarray(goals, float), np.asarray(outcomes)
    if X.ndim != 2 or goals.shape != (len(X), 2) or outcomes.shape != (len(X),) or not np.isfinite(X).all() or not np.isfinite(goals).all() or not np.isin(outcomes, CLASS_ORDER).all():
        raise ValueError('Neighbor selection needs aligned finite inputs and H/D/A outcomes.')
    feasible = [k for k in ks if k <= len(X)-1]
    if not feasible:
        raise ValueError('No feasible k: leave-one-out fitting population must have at least three other matches.')
    score_predictions = np.empty((len(feasible), len(X), 2))
    probabilities = np.zeros((len(feasible), len(X), 3))
    indices = np.empty((len(X), max(feasible)), dtype=np.intp)
    rows = np.arange(len(X))
    for held_out in rows:
        train = rows[rows != held_out]
        # With explicit kd_tree, k controls the query, not construction of the
        # fitted tree. Independent per-k refits are the regression oracle.
        regressor = KNeighborsRegressor(n_neighbors=max(feasible), weights='uniform', metric='euclidean', algorithm='kd_tree', n_jobs=1).fit(X[train], goals[train])
        classifier = KNeighborsClassifier(n_neighbors=max(feasible), weights='uniform', metric='euclidean', algorithm='kd_tree', n_jobs=1).fit(X[train], outcomes[train])
        query = X[held_out:held_out+1]
        class_positions = [CLASS_ORDER.index(value) for value in classifier.classes_]
        for j, k in enumerate(feasible):
            regressor.set_params(n_neighbors=k)
            classifier.set_params(n_neighbors=k)
            score_predictions[j, held_out] = regressor.predict(query)[0]
            probabilities[j, held_out, class_positions] = classifier.predict_proba(query)[0]
        indices[held_out] = train[regressor.kneighbors(query, return_distance=False)[0]]
    candidate_index = {k: j for j, k in enumerate(feasible)}
    records = []
    for k in ks:
        if k not in feasible:
            records.append(dict(k=k, status='infeasible', rows=len(X), reason='k exceeds N-1'))
        else:
            records.append(dict(k=k, status='evaluated', rows=len(X),
                joint_rmse=joint_score_rmse(goals, score_predictions[candidate_index[k]]),
                rps=rps(outcomes, probabilities[candidate_index[k]])))
    table = pd.DataFrame(records)
    valid = table[table.status.eq('evaluated')]
    best = {metric: int(valid.loc[valid[metric].idxmin(), 'k']) for metric in ('joint_rmse', 'rps')}
    ties = {metric: valid.loc[valid[metric].eq(valid[metric].min()), 'k'].astype(int).tolist() for metric in best}
    return best, ties, table, indices


@dataclass
class CompetitionResult:
    competition_id: object
    n: int
    k_goals: int
    k_outcomes: int
    n_table: pd.DataFrame
    k_table: pd.DataFrame
    ties: dict
    goals: object
    outcomes: object
    dataset: object
    holdout: dict


def fit_bundles(bundles, *, development_end, holdout_end, k_values=range(3, 351)):
    """Choices see development labels only; held-out features update causally.

    The fixed model is evaluated on later fixtures. Completed earlier held-out
    results may enter later historical features, never n/k selection or fitting.
    """
    end, stop = pd.to_datetime(development_end, utc=True), pd.to_datetime(holdout_end, utc=True)
    if pd.isna(end) or pd.isna(stop) or end >= stop:
        raise ValueError('Require development_end < holdout_end.')
    if not bundles:
        raise ValueError('No feature bundles.')
    first = bundles[min(bundles)][0]
    results = []
    for comp in first.metadata.competition_id.unique():
        records, populations = [], {}
        for n in sorted(bundles):
            data, outcomes = bundles[n]
            md = data.metadata
            valid = md.research_eligible & np.isfinite(data.y).all(axis=1) & np.isfinite(outcomes.y).all(axis=1)
            dev = np.flatnonzero(valid & md.competition_id.eq(comp) & md.prediction_cutoff.le(end) & md.label_available_at.le(end))
            populations[n] = dev
            X, y = data.X.iloc[dev].to_numpy(), data.y.iloc[dev].to_numpy()
            z = (X[:, 0]-X[:, 1])-(X[:, 3]-X[:, 4])+X[:, 0]-X[:, 3]
            difference = y[:, 0]-y[:, 1]
            correlation = float(np.corrcoef(z, difference)[0, 1]) if len(dev) >= 3 and np.std(z) > 0 and np.std(difference) > 0 else np.nan
            records.append(dict(n=n, rows=len(dev), pearson=correlation,
                                status='evaluated' if np.isfinite(correlation) else 'rejected_constant_or_insufficient'))
        nt = pd.DataFrame(records)
        finite = nt[nt.status.eq('evaluated')]
        if finite.empty:
            raise ValueError(f'Competition {comp}: no finite signed Pearson candidate.')
        n = int(finite.loc[finite.pearson.idxmax(), 'n'])
        data, outcomes = bundles[n]
        dev = populations[n]
        best, ties, kt, _ = select_neighbors(data.X.iloc[dev], data.y.iloc[dev], outcomes.y.iloc[dev, 0], k_values)
        ties['n'] = finite.loc[finite.pearson.eq(finite.pearson.max()), 'n'].astype(int).tolist()
        md = data.metadata
        test = np.flatnonzero(md.competition_id.eq(comp) & md.research_eligible & md.prediction_cutoff.gt(end) &
                             md.prediction_cutoff.le(stop) & md.label_available_at.le(stop) & np.isfinite(data.y).all(axis=1))
        if not len(test):
            raise ValueError(f'Competition {comp}: chronological holdout has no eligible released labels.')
        plan = SplitPlan([Fold(dev, test, test, {'research_selection': 'retrospective_LOOCV_on_development'})], len(data.X), np.arange(len(data.X)))
        goal_run = TrainingRunner(lambda: EstimatorAdapter(KNeighborsRegressor(n_neighbors=best['joint_rmse'], weights='uniform', metric='euclidean', algorithm='kd_tree', n_jobs=1))).run(data, plan)
        outcome_run = TrainingRunner(lambda: OutcomeAdapter(KNeighborsClassifier(n_neighbors=best['rps'], weights='uniform', metric='euclidean', algorithm='kd_tree', n_jobs=1), ('predict', 'predict_proba'))).run(outcomes, plan)
        metrics = dict(rows=len(test), joint_rmse=joint_score_rmse(data.y.iloc[test], goal_run.folds[0].predictions['predict']),
                       rps=rps(outcomes.y.iloc[test, 0], outcome_run.folds[0].predictions['predict_proba']))
        results.append(CompetitionResult(comp, n, best['joint_rmse'], best['rps'], nt, kt, ties, goal_run, outcome_run, data, metrics))
    return results


@dataclass(frozen=True)
class DeepthroatGGRecipe:
    """JSON/Python portable research configuration; data/roster are explicit inputs."""
    development_end: str
    holdout_end: str
    n_values: tuple = tuple(range(9, 101))
    k_values: tuple = tuple(range(3, 351))
    available_at: str | None = None
    cutoffs: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'n_values', windows(self.n_values, 9))
        object.__setattr__(self, 'k_values', windows(self.k_values, 3))
        end, stop = pd.to_datetime(self.development_end, utc=True), pd.to_datetime(self.holdout_end, utc=True)
        if pd.isna(end) or pd.isna(stop) or end >= stop:
            raise ValueError('Require development_end < holdout_end.')

    def to_dict(self):
        from dataclasses import asdict
        return {'format': 'deepthroat_gg_v1', **asdict(self)}

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if values.pop('format', None) != 'deepthroat_gg_v1':
            raise ValueError('Unsupported Deepthroat-gg recipe version.')
        return cls(**values)

    def export_python(self):
        return 'from xdiyo_analytics.research.deepthroat_gg import DeepthroatGGRecipe\nrecipe = DeepthroatGGRecipe.from_dict(' + repr(self.to_dict()) + ')\n# result = recipe.run(history, roster)\n'

    def run(self, history, roster):
        bundles = prepare_bundles(history, roster, n_values=self.n_values, cutoffs=self.cutoffs, available_at=self.available_at)
        return fit_bundles(bundles, development_end=self.development_end, holdout_end=self.holdout_end, k_values=self.k_values)


def save_research(results, recipe, history, roster, path):
    """Save native models, selection evidence and exact caller-owned inputs once."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    history.to_parquet(path/'history.parquet', compression='zstd', index=True)
    roster.to_parquet(path/'roster.parquet', compression='zstd', index=True)
    (path/'recipe.json').write_text(json.dumps(recipe.to_dict(), indent=2), encoding='utf-8')
    (path/'recipe.py').write_text(recipe.export_python(), encoding='utf-8')
    manifest = dict(display_name='Deepthroat-gg', source='Berrar, Lopes, Dubitzky (2024), DOI 10.1007/s10994-024-06625-9',
        membership=POLICY, features=FEATURES, classes=CLASS_ORDER, missing_class='zero probability',
        selection='signed Pearson n, then separate retrospective leave-self-out k; ties choose smallest',
        distance='Euclidean', weights='uniform', scaling='none', rounding='none',
        neighbor_policy=NEIGHBOR_POLICY,
        neighbor_ties='native sklearn selected-k queries on each fitting population; original case order preserved',
        availability=recipe.available_at or 'retrospective kickoff proxy with native completion bounds',
        versions={'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','pyarrow','joblib')}}, competitions=[])
    for i, result in enumerate(results):
        folder = path/f'competition-{i}'
        folder.mkdir()
        save_model(result.goals, folder/'goals')
        save_model(result.outcomes, folder/'outcomes')
        result.n_table.to_csv(folder/'n-selection.csv', index=False)
        result.k_table.to_csv(folder/'k-selection.csv', index=False)
        result.dataset.X.to_parquet(folder/'features.parquet', compression='zstd')
        result.dataset.metadata.to_parquet(folder/'metadata.parquet', compression='zstd')
        manifest['competitions'].append(dict(competition_id=str(result.competition_id), folder=folder.name,
            n=result.n, k_goals=result.k_goals, k_outcomes=result.k_outcomes, ties=result.ties, holdout=result.holdout,
            fit_positions=result.goals.folds[0].fit_positions.tolist(), test_positions=result.goals.folds[0].test_positions.tolist()))
    manifest['sha256'] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
    manifest['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    source_root = Path(__file__).parents[1]
    manifest['library_source_sha256'] = {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in source_root.rglob('*.py')}
    (path/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
