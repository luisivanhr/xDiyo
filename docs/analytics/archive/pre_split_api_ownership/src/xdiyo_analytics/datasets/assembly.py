"""Identity-based feature/target alignment and explicit model row layouts."""

from dataclasses import dataclass, field

from ..labels import LabelData


@dataclass
class ModelDataset:
    """Aligned data for splitting, reporting and fitting (none performed here).

    X/y/metadata share one RangeIndex and the same row order. layout declares
    'match' or 'team_match'; identity_columns identify individual observations,
    match_columns identify paired groups. groups returns match-identity tuples,
    not ranker group sizes. Metadata includes settlement::<target> when present.
    target_perspective retains the selected LabelData perspective; definitions
    records the supplied feature/label definitions for later reports.
    """
    X: object
    y: object
    metadata: object
    layout: str
    identity_columns: tuple[str, ...]
    match_columns: tuple[str, ...]
    target_perspective: str
    definitions: dict = field(default_factory=dict)

    @property
    def groups(self):
        """Exact match identities, repeated for the two team rows when applicable."""
        import pandas as pd
        keys = list(zip(*(self.metadata[name].tolist() for name in self.match_columns)))
        return pd.Series(keys, index=self.metadata.index, dtype=object, name="match_group")


def assemble_dataset(features, label, *, layout, feature_columns=None,
                     target_columns=None, drop_missing_targets=False):
    """Assemble one selected LabelData and identity-keyed team features.

    features is evaluate_features(..., keyed=True), or a DataFrame with explicit
    identifier columns: all label match keys, team_id and side. An ordinary index
    or matching row count never substitutes for identifiers. Custom keyed frames
    may use a named MultiIndex containing those same fields. Extra feature rows
    are allowed; every label row (both sides in match layout) must find its keys.

    layout must equal label.unit. 'team_match' retains per-team features and label
    order. 'match' aligns home/away independently, puts home::<feature> followed by
    away::<feature> columns beside each other, and follows label match order. It
    does not aggregate team labels or replicate match labels across teams.

    feature_columns and target_columns select names before pivot/filtering; None
    selects all feature/target columns, excluding explicit identifier columns.
    Feature values, including missing values, pass through with no imputation,
    scaling, deduplication or fitting. Missing feature *records* raise an error.

    drop_missing_targets=False retains prediction fixtures with missing y. True
    removes an entire match when any selected target is missing for either team,
    preserving pairs. LabelData numeric outputs represent nonfinite outcomes as
    missing; manually supplied infinities are not automatically recoded here.
    Finite push/void encodings remain selected values, not implicit drop signals.
    Settlement categories accompany selected targets in metadata, never in X.

    Output frames share a fresh RangeIndex; IDs keep their original exact dtypes.
    No source mutation, file I/O, chronological splitting or automatic sorting.
    """
    import pandas as pd
    from copy import deepcopy

    if not isinstance(label, LabelData):
        raise TypeError("Select one LabelData from create_labels() before assembly.")
    if layout not in {"match", "team_match"} or layout != label.unit:
        raise ValueError("layout must be match/team_match and equal the selected label.unit.")
    if not isinstance(drop_missing_targets, bool):
        raise TypeError("drop_missing_targets must be a boolean.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame with explicit match/team identifiers.")
    for name, frame in (("features", features), ("label.y", label.y), ("label.metadata", label.metadata)):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError(f"{name} must be a DataFrame with unique columns.")
    if not label.y.index.equals(label.metadata.index):
        raise ValueError("LabelData y and metadata must retain their shared row order/index.")
    identities = tuple(label.identity_columns)
    if not identities or len(set(identities)) != len(identities) or "event_id" not in identities:
        raise ValueError("Label identity_columns must be distinct and include event_id.")
    if ("team_id" in identities) != (layout == "team_match") or "side" in identities:
        raise ValueError("Label identities must describe the declared match/team-match unit.")
    match_keys = tuple(name for name in identities if name != "team_id")
    required = [*match_keys, "team_id", "side"]
    if isinstance(features.index, pd.MultiIndex) and set(required) <= set(features.index.names):
        if len(set(features.index.names)) != len(features.index.names):
            raise ValueError("Feature index level names must be distinct.")
        feature_ids = features.index.to_frame(index=False)[required]
        candidates = list(features.columns)
    elif set(required) <= set(features.columns):
        feature_ids = features[required].reset_index(drop=True)
        candidates = [name for name in features.columns if name not in required]
    else:
        raise ValueError("Features need explicit match keys, team_id and side; use evaluate_features(..., keyed=True).")
    if feature_ids.isna().any().any() or not feature_ids["side"].isin(["home", "away"]).all():
        raise ValueError("Feature identities must be nonmissing with side home/away.")
    if feature_ids.duplicated([*match_keys, "team_id"]).any() or feature_ids.duplicated([*match_keys, "side"]).any():
        raise ValueError("Feature rows must be unique per match/team and match/side.")

    def select_columns(requested, available, name):
        chosen = list(available) if requested is None else [requested] if isinstance(requested, str) else list(requested)
        if not chosen or any(not isinstance(column, str) or not column for column in chosen) or len(set(chosen)) != len(chosen):
            raise ValueError(f"{name} must select distinct, nonempty string column names.")
        missing = set(chosen) - set(available)
        if missing:
            raise KeyError(f"Unknown {name}: {sorted(missing)}")
        return chosen

    feature_names = select_columns(feature_columns, candidates, "feature_columns")
    target_names = select_columns(target_columns, label.y.columns, "target_columns")
    meta = label.metadata.copy(deep=True).reset_index(drop=True)
    meta.attrs = {}
    missing = set(identities) - set(meta.columns)
    if missing:
        raise KeyError(f"Label metadata lacks identity columns: {sorted(missing)}")
    if meta[list(identities)].isna().any().any() or meta.duplicated(list(identities)).any():
        raise ValueError("Label metadata must have unique, nonmissing observation identities.")
    y = label.y[target_names].copy(deep=True).reset_index(drop=True)
    y.attrs = {}
    values = features[feature_names].copy(deep=True).reset_index(drop=True)
    values.attrs = {}

    def locate(keys, queries):
        source = pd.MultiIndex.from_frame(feature_ids[list(keys)])
        wanted = pd.MultiIndex.from_frame(queries[list(keys)])
        positions = source.get_indexer(wanted)
        if (positions < 0).any():
            raise ValueError("Selected label observations are missing feature records; missing feature values are allowed, missing identifiers are not.")
        return positions

    if layout == "team_match":
        if not {"side", "opponent_id"} <= set(meta.columns):
            raise KeyError("Team label metadata needs side and opponent_id.")
        if not meta["side"].isin(["home", "away"]).all() or meta["opponent_id"].isna().any():
            raise ValueError("Team labels need home/away sides and nonmissing opponent IDs.")
        positions = locate(identities, meta)
        if feature_ids["side"].iloc[positions].tolist() != meta["side"].tolist():
            raise ValueError("Feature and label team perspectives disagree.")
        # Verify complete pairs before filtering so grouping remains meaningful.
        if meta.duplicated([*match_keys, "side"]).any() or not meta.groupby(list(match_keys), dropna=False).size().eq(2).all():
            raise ValueError("Team-match datasets require both team rows per match.")
        pairs = {}
        for key, team, opponent in zip(
            zip(*(meta[name].tolist() for name in match_keys)), meta["team_id"].tolist(), meta["opponent_id"].tolist(),
        ):
            if team == opponent:
                raise ValueError("Team label identities cannot describe a team playing itself.")
            previous = pairs.setdefault(key, (team, opponent))
            if previous != (team, opponent) and previous != (opponent, team):
                raise ValueError("Paired label opponent IDs disagree.")
        X = values.iloc[positions].reset_index(drop=True)
    else:
        if not {"home_id", "away_id"} <= set(meta.columns):
            raise KeyError("Match label metadata needs home_id and away_id.")
        if meta[["home_id", "away_id"]].isna().any().any() or meta["home_id"].eq(meta["away_id"]).any():
            raise ValueError("Match labels need distinct, nonmissing home/away IDs.")
        halves = []
        for side in ("home", "away"):
            query = meta[list(match_keys)].copy()
            query["side"] = side
            positions = locate([*match_keys, "side"], query)
            if feature_ids["team_id"].iloc[positions].tolist() != meta[f"{side}_id"].tolist():
                raise ValueError("Feature and label home/away team IDs disagree.")
            part = values.iloc[positions].reset_index(drop=True)
            part.columns = [f"{side}::{name}" for name in feature_names]
            halves.append(part)
        X = pd.concat(halves, axis=1)

    if label.settlement is not None:
        if not isinstance(label.settlement, pd.DataFrame) or not label.settlement.columns.is_unique or not label.settlement.index.equals(label.y.index):
            raise ValueError("Settlement must be a DataFrame aligned with LabelData.y.")
        for name in target_names:
            column = f"settlement::{name}"
            if column in meta:
                raise ValueError(f"Settlement metadata column already exists: {column}")
            if name not in label.settlement:
                raise KeyError(f"Settlement is missing target column: {name}")
            meta[column] = label.settlement[name].reset_index(drop=True).copy()
    if drop_missing_targets:
        missing_y = y.isna().any(axis=1)
        match_index = pd.MultiIndex.from_frame(meta[list(match_keys)])
        keep = ~match_index.isin(match_index[missing_y.to_numpy()])
        X, y, meta = [frame.loc[keep].reset_index(drop=True) for frame in (X, y, meta)]
    return ModelDataset(
        X=X, y=y, metadata=meta, layout=layout, identity_columns=identities,
        match_columns=match_keys, target_perspective=label.perspective,
        definitions={"features": deepcopy(features.attrs.get("features", {})),
                     "label": deepcopy(label.definition)},
    )
