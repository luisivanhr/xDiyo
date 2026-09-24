"""Fixture-level inspection of retained observed and predicted labels."""

from dataclasses import dataclass
import json
from numbers import Real
from typing import ClassVar

import numpy as np
import pandas as pd

from ..labels import Above, BetOption, Outcome
from .contracts import Artifact, StudyResult
from .post_training import PredictionReporter
from .studies import _columns
from .teams import TeamCatalog, team_key


def _plain(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _probability_rows(frame, target, index):
    if not frame.index.equals(index) or not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Probability outputs need aligned rows and (target, class) columns.")
    values = frame.xs(target, level=0, axis=1)
    if not values.columns.is_unique or len(values.columns) < 2:
        raise ValueError("Probabilities require distinct class columns (at least two).")
    # A class column may be absent in one pooled fold; retain that row as unavailable.
    rows = []
    for row in values.itertuples(index=False, name=None):
        numbers = np.asarray([np.nan if _plain(v) is None else v for v in row], dtype=float)
        valid = (np.isfinite(numbers).all() and ((numbers >= 0) & (numbers <= 1)).all()
                 and np.isclose(numbers.sum(), 1, atol=1e-6, rtol=0))
        rows.append(dict(zip(values.columns, numbers)) if valid else None)
    return rows


@dataclass(kw_only=True)
class MatchResultReporter(PredictionReporter):
    """Compact fixtures with interactive filters and independently colored labels.

    type is per_fold or overall; partition defaults to score. target is a name,
    list of names, or None for all selected targets (one panel per label). Filters
    league/season/team/round are initial display selections, not scope reduction;
    None means All. Tables retain every input observation and its fold/row IDs.
    Team filtering uses either home or away ID, never ambiguous display names.

    comparison='auto' uses categorical comparison for Outcome/Above/BetOption
    definitions or nonnumeric targets, numeric otherwise. Explicit categorical
    supports numeric class labels from custom definitions. Numeric success is
    abs(prediction-result)<=tolerance (default exact). Error is prediction-result.
    Missing/nonfinite values and bet push/void/missing settlements remain neutral.

    output='predict' displays point predictions. A MultiIndex probability output
    displays probabilities without a class decision by default. decision='argmax'
    opts into a decision (first stored class wins ties); threshold requires a
    positive_class and binary probabilities, and uses >=. probability_output adds
    companion probabilities when present. Nothing fits or calls a model.

    catalog accepts TeamCatalog or an ID mapping. show_badges embeds optional
    local assets once per team, so saved HTML/notebook output works offline.
    Metadata names are used when present, otherwise catalog names or Team <ID>.
    league_column/season_column default to source_* then competition_id/season_id.
    """
    partition: str = "score"
    target: object = None
    output: str = "predict"
    comparison: str = "auto"
    tolerance: float = 0.0
    league: object = None
    season: object = None
    team: object = None
    round: object = None
    catalog: object = None
    show_badges: bool = False
    page_size: int = 25
    league_column: str | None = None
    season_column: str | None = None
    probability_output: str | None = "predict_proba"
    decision: str | None = None
    threshold: float | None = None
    positive_class: object = None
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        if context.partition not in {"test", "score"} or context.layout not in {"match", "team_match"}:
            raise ValueError("Match results require match/team_match test or score predictions.")
        if self.comparison not in {"auto", "numeric", "categorical"}:
            raise ValueError("comparison must be auto, numeric or categorical.")
        if isinstance(self.tolerance, bool) or not isinstance(self.tolerance, Real) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if isinstance(self.page_size, bool) or not isinstance(self.page_size, int) or self.page_size < 1:
            raise ValueError("page_size must be a positive integer.")
        if self.decision not in {None, "argmax"} or (self.decision is not None and self.threshold is not None):
            raise ValueError("Choose decision=None/'argmax' or a binary threshold, not both.")
        if self.threshold is not None and (not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1 or self.positive_class is None):
            raise ValueError("A binary threshold needs a value in [0,1] and positive_class.")
        meta = context.metadata
        if not meta.index.equals(context.y.index):
            raise ValueError("Match metadata and targets must stay aligned.")
        selected = _columns(context.y, self.target, "target")
        prediction_frame = context.predictions[self.output]
        if not prediction_frame.index.equals(context.y.index):
            raise ValueError("Predictions and targets must stay aligned.")
        is_probability = isinstance(prediction_frame.columns, pd.MultiIndex)
        if not is_probability and (self.decision is not None or self.threshold is not None):
            raise ValueError("Decision settings require selecting a probability output.")
        league_col = self.league_column or ("source_league" if "source_league" in meta else "competition_id")
        season_col = self.season_column or ("source_season" if "source_season" in meta else "season_id")
        for field, column in (("league", league_col), ("season", season_col), ("round", "round")):
            if (getattr(self, field) is not None or field in {"league", "season"}) and column not in meta:
                raise KeyError(f"Match-result {field} metadata is unavailable: {column}")
        catalog = self.catalog if isinstance(self.catalog, TeamCatalog) else TeamCatalog(self.catalog or {})
        teams, notes = {}, []
        base = []
        # Column-oriented records retain large integer IDs without iterrows coercion.
        for (fold_id, row_position), record in zip(meta.index, meta.to_dict("records")):
            side = record.get("side") if context.layout == "team_match" else None
            if context.layout == "team_match":
                if side not in {"home", "away"}:
                    raise ValueError("Team-match results need a home/away side.")
                home = record["team_id"] if side == "home" else record["opponent_id"]
                away = record["opponent_id"] if side == "home" else record["team_id"]
            else:
                home, away = record["home_id"], record["away_id"]
            for role, key in (("home", home), ("away", away)):
                if team_key(key) not in teams:
                    name, badge, note = catalog.display(key, badges=self.show_badges)
                    if not catalog.entries.get(team_key(key), {}).get("name"):
                        fallback = record.get(f"{role}_name")
                        if context.layout == "team_match":
                            fallback = record.get("team_name" if side == role else "opponent_name", fallback)
                        if fallback is not None and pd.notna(fallback):
                            name = str(fallback)
                        else:
                            notes.append(f"No display name for team {key}; supply it in TeamCatalog.")
                    teams[team_key(key)] = dict(name=name, badge=badge)
                    if note:
                        notes.append(note)
            identity = {key: record[key] for key in context.match_columns}
            fixture = json.dumps([str(fold_id), *[str(v) for v in identity.values()]], ensure_ascii=False)
            base.append(dict(**identity, fixture=fixture, fold_id=fold_id, row_position=row_position,
                             league=_plain(record.get(league_col)), season=_plain(record.get(season_col)),
                             round=_plain(record.get("round")), stage=_plain(record.get("stage")),
                             home_id=home, away_id=away, home=teams[team_key(home)]["name"],
                             away=teams[team_key(away)]["name"], side=side))
        result = StudyResult("Match results", notes=list(dict.fromkeys(notes)))
        result.notes.append("Filters change the visible fixtures only. Data tables retain the full reporter population; the panel exports filtered rows.")
        result.notes.append("Numeric error = prediction − result. Missing values and push/void settlements are neutral.")
        for target in selected:
            definition = context.definitions.get("label")
            categorical = self.comparison == "categorical" or (self.comparison == "auto" and
                (is_probability or isinstance(definition, (Outcome, Above, BetOption)) or not pd.api.types.is_numeric_dtype(context.y[target])))
            probabilities = _probability_rows(prediction_frame, target, context.y.index) if is_probability else None
            if is_probability and self.threshold is not None:
                classes = prediction_frame.xs(target, level=0, axis=1).columns
                if len(classes) != 2 or self.positive_class not in classes:
                    raise ValueError("Threshold decisions require two classes including positive_class.")
                negative = next(key for key in classes if key != self.positive_class)
            if not is_probability and self.probability_output in context.predictions:
                probabilities = _probability_rows(context.predictions[self.probability_output], target, context.y.index)
            predictions = [None]*len(base) if is_probability else prediction_frame[target].tolist()
            if is_probability and (self.decision is not None or self.threshold is not None):
                for i, values in enumerate(probabilities):
                    if values is None:
                        continue
                    if self.threshold is not None:
                        predictions[i] = self.positive_class if values[self.positive_class] >= self.threshold else negative
                    else:
                        predictions[i] = max(values, key=values.get)
            settlements = meta.get(f"settlement::{target}", pd.Series([None]*len(meta), index=meta.index)).tolist()
            rows = []
            for i, (observed, predicted, settlement) in enumerate(zip(context.y[target].tolist(), predictions, settlements)):
                observed, predicted, settlement = _plain(observed), _plain(predicted), _plain(settlement)
                status, error = "unavailable", None
                if settlement in {"push", "void", "missing"}:
                    status = settlement
                elif observed is not None and predicted is not None:
                    if categorical:
                        status = "correct" if observed == predicted else "incorrect"
                    else:
                        if not isinstance(observed, Real) or not isinstance(predicted, Real):
                            raise TypeError("Numeric comparisons require numeric labels and predictions.")
                        error = float(predicted)-float(observed)
                        status = "within tolerance" if abs(error) <= self.tolerance else "outside tolerance"
                elif is_probability and probabilities[i] is not None and observed is not None:
                    status = "no decision"
                row = dict(base[i], target=target, result=observed, prediction=predicted, error=_plain(error),
                           status=status, settlement=settlement,
                           probabilities=(json.dumps({str(k): float(v) for k, v in probabilities[i].items()}, ensure_ascii=False)
                                          if probabilities is not None and probabilities[i] is not None else None))
                rows.append(row)
            columns = list(base[0]) if base else [*context.match_columns, "fixture", "fold_id", "row_position", "league", "season", "round", "stage", "home_id", "away_id", "home", "away", "side"]
            table = pd.DataFrame(rows, columns=[*columns, "target", "result", "prediction", "error", "status", "settlement", "probabilities"])
            if len(table) and table.duplicated(["fixture", "side"]).any():
                raise ValueError("A fold must have at most one observation per fixture/side.")
            for _, group in table.groupby("fixture", sort=False):
                if any(group[column].nunique(dropna=False) != 1 for column in ("home_id", "away_id", "league", "season", "round", "stage")):
                    raise ValueError("Paired fixture rows have conflicting teams or competition context.")
            result.tables[f"matches::{target}"] = table
            result.artifacts.append(Artifact("match_results", table, f"{target}: results and predictions", dict(
                teams=teams, layout=context.layout, page_size=self.page_size, numeric=not categorical,
                probabilities=probabilities is not None, tolerance=float(self.tolerance),
                initial={key: None if getattr(self, key) is None else team_key(getattr(self, key)) for key in ("league", "season", "team", "round")})))
        return result
