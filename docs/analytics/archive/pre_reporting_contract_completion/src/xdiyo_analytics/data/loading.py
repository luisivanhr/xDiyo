"""Load experiment tables; selection and basic checks stay behind the API."""

from dataclasses import dataclass
import hashlib
from numbers import Integral
from pathlib import Path
import re
from typing import TYPE_CHECKING
import warnings

from ._source import _open_source

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class SeasonData:
    """Requested DataFrames plus a small source record. Tables remain separate."""

    tables: dict[str, "pd.DataFrame"]
    provenance: dict

    def __getitem__(self, name: str) -> "pd.DataFrame":
        return self.tables[name]

    @property
    def matches(self):
        return self.tables.get("matches")

    @property
    def statistics(self):
        return self.tables.get("statistics")

    @property
    def pregame(self):
        return self.tables.get("pregame")

    @property
    def shots(self):
        return self.tables.get("shots")


def _read_table(source, name, verify_hashes):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = source.table_path(name)
    info = source.manifest["tables"][name]
    with path.open("rb") as stream:
        stream.seek(0, 2)
        if stream.tell() != info["bytes"]:
            raise ValueError(f"File size does not match the export: {name}")
        stream.seek(0)
        reader = stream
        if verify_hashes:
            payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != str(info.get("sha256", "")).lower():
                raise ValueError(f"File fingerprint does not match the export: {name}")
            reader = pa.BufferReader(payload)
        try:
            with pq.ParquetFile(reader) as parquet:
                table = parquet.read()
        except pa.ArrowException as exc:
            raise ValueError(f"Cannot read Parquet table {name}") from exc
    if table.num_rows != info["rows"]:
        raise ValueError(f"Row count does not match the export: {name}")
    if len(set(table.column_names)) != len(table.column_names):
        raise ValueError(f"Repeated column names in {name}")
    return table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)


def _require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _check_matches(frame, scope):
    import pandas as pd

    ids = ("event_id", "home_id", "away_id", "competition_id", "season_id")
    _require(frame, (*ids, "kickoff_utc"), "matches")
    for key in ids:
        if frame[key].isna().any() or (len(frame) and (
            not pd.api.types.is_integer_dtype(frame[key].dtype) or frame[key].le(0).any()
        )):
            raise ValueError(f"matches.{key} must contain positive integer IDs")
    if frame.event_id.duplicated().any() or frame.home_id.eq(frame.away_id).any():
        raise ValueError("Matches need unique event IDs and different home/away teams")
    if not frame.competition_id.eq(scope["league_id"]).all() or not frame.season_id.eq(scope["season_id"]).all():
        raise ValueError("Match rows belong to a different competition or season")
    bad_time = pd.to_datetime(frame.kickoff_utc, unit="s", errors="coerce", utc=True).isna().sum()
    if bad_time:
        warnings.warn(f"{bad_time} matches have missing or unreadable kickoff times; rows retained.",
                    UserWarning, stacklevel=3)


def _check_statistics(frame, matches=None):
    import pandas as pd

    grain = ["event_id", "period", "group_name", "key", "side"]
    _require(frame, [*grain, "value"], "statistics")
    if frame[grain].isna().any().any() or frame.duplicated(grain).any():
        raise ValueError("Statistics have missing identities or duplicate match/period/group/key/side entries")
    if len(frame) and (not pd.api.types.is_integer_dtype(frame.event_id.dtype) or frame.event_id.le(0).any()):
        raise ValueError("Statistics need positive integer match IDs")
    if not frame.side.isin(["home", "away"]).all():
        raise ValueError("Statistics side must be home or away")
    for key in ("period", "group_name", "key"):
        if not frame[key].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"statistics.{key} must contain nonempty labels")
    if matches is None:
        return
    teams = {event: {"home": home, "away": away} for event, home, away in
            matches[["event_id", "home_id", "away_id"]].itertuples(index=False, name=None)}
    if not frame.event_id.isin(teams).all():
        raise ValueError("Statistics reference matches outside the loaded season")
    if "team_id" in frame:
        for event, side, team in frame[["event_id", "side", "team_id"]].itertuples(index=False, name=None):
            if pd.notna(team) and (isinstance(team, bool) or not isinstance(team, Integral) or team != teams[event][side]):
                raise ValueError("A statistics team ID does not match its home/away team")


def load_season(data_root, season_stem, *, tables=("matches",), record_path=None,
                verify_hashes=False) -> SeasonData:
    """Load selected tables from one season version; no joins or normalization.

    Pass a table name or a list/tuple of names. Returns separate DataFrames with
    exact nullable Arrow types. Essential match/statistic checks run internally;
    cross-table references are checked when both tables are requested. Missing
    values remain missing. Unrequested convenience properties return None.

    Optional record_path saves/reuses the existing selection-record format, only
    after a successful load. Records belong outside data_root; existing ones are
    never overwritten. Full-file SHA256 checking is opt-in with verify_hashes=True;
    readable files, sizes, counts and basic structure are always checked.
    Unavailable tables raise KeyError; missing files FileNotFoundError; invalid
    selections or unusable data ValueError. Source exports are never written.
    """
    if isinstance(tables, str):
        tables = [tables]
    if not isinstance(tables, (list, tuple)) or not tables or any(not isinstance(n, str) or not n.strip() for n in tables):
        raise ValueError("Choose a table name or a nonempty list of table names")
    if len(set(tables)) != len(tables):
        raise ValueError("Choose each table only once")
    source = _open_source(data_root, season_stem, record_path)
    for name in tables:
        source.table_path(name)
    frames = {name: _read_table(source, name, verify_hashes) for name in tables}
    if "matches" in frames:
        _check_matches(frames["matches"], source.manifest["scope"])
    if "statistics" in frames:
        _check_statistics(frames["statistics"], frames.get("matches"))
    provenance = {"season": season_stem, "version": source.manifest["version"],
                "manifest_path": str(source.path), "tables": tuple(tables)}
    for name, frame in frames.items():
        frame.attrs["source"] = {**provenance, "table": name}
    source.save()
    return SeasonData(frames, provenance)


def load_season_table(data_root, season_stem, *, table="matches", record_path=None,
                    verify_hashes=False):
    """Compatibility shortcut: return one DataFrame using the same loading path."""
    return load_season(data_root, season_stem, tables=table, record_path=record_path,
                    verify_hashes=verify_hashes)[table]


def load_seasons(data_root, seasons, *, leagues=None, tables=("matches",),
                 record_dir=None, verify_hashes=False) -> SeasonData:
    """Load available league-season publications and combine each table separately.

    seasons is a label or list/tuple such as ['22_23', '23_24', '24_25'].
    leagues=None discovers all available leagues; otherwise supply exact league
    names such as 'Premier_League'. Each requested season and league must have a
    matching publication, but absent league-season combinations are not invented.
    Failed/incomplete publications raise rather than silently disappearing.

    Uses load_season for each partition, in season order then league-name order.
    Adds source_league and source_season to every row; rejects existing columns
    with those names instead of overwriting them. Concatenates matching tables
    with a fresh index, retaining source row order and missing optional columns.
    Does not join tables, create features or remove duplicate rows across seasons.

    Optional record_dir holds one compatible selection record per publication.
    Records pin versions, not discovery membership: new matching publications may
    be discovered on future calls. Each successful partition may save its record
    even if a later partition fails. The returned provenance lists the exact
    population loaded. No source tables are changed; coverage is never required.
    """
    import pandas as pd

    if isinstance(seasons, str):
        seasons = [seasons]
    if (not isinstance(seasons, (list, tuple)) or not seasons
            or any(not isinstance(s, str) or not re.fullmatch(r"[0-9]{2}_[0-9]{2}", s)
                   or (int(s[:2]) + 1) % 100 != int(s[3:]) for s in seasons)):
        raise ValueError("Choose seasons such as ['22_23', '23_24', '24_25']")
    if len(set(seasons)) != len(seasons):
        raise ValueError("Choose each season only once")
    if isinstance(leagues, str):
        leagues = [leagues]
    if leagues is not None and (not isinstance(leagues, (list, tuple)) or not leagues
            or any(not isinstance(n, str) or not n.strip() for n in leagues)
            or len(set(leagues)) != len(leagues)):
        raise ValueError("Choose distinct league names or leave leagues=None for all available")
    if isinstance(tables, str):
        tables = [tables]
    if (not isinstance(tables, (list, tuple)) or not tables
            or any(not isinstance(n, str) or not n.strip() for n in tables)
            or len(set(tables)) != len(tables)):
        raise ValueError("Choose a table name or a nonempty list of distinct table names")

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    records = Path(record_dir).resolve() if record_dir is not None else None
    if records is not None and records.is_relative_to(root):
        raise ValueError("Save selection records outside the source data directory")
    selected = []
    for path in root.glob("*.manifest.json"):
        stem = path.name.removesuffix(".manifest.json")
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        league, start, end = parts
        season = f"{start}_{end}"
        if season in seasons and (leagues is None or league in leagues):
            selected.append((season, league, stem))
    missing_seasons = set(seasons) - {s for s, _, _ in selected}
    missing_leagues = set(leagues or ()) - {league for _, league, _ in selected}
    if missing_seasons or missing_leagues:
        raise ValueError(f"No publications for seasons {sorted(missing_seasons)} or leagues {sorted(missing_leagues)}")
    selected.sort(key=lambda item: (seasons.index(item[0]), item[1]))

    pieces = {name: [] for name in tables}
    sources = []
    for season, league, stem in selected:
        loaded = load_season(root, stem, tables=tables,
                             record_path=records / f"{stem}.json" if records else None,
                             verify_hashes=verify_hashes)
        for name, frame in loaded.tables.items():
            if {"source_league", "source_season"} & set(frame.columns):
                raise ValueError(f"{stem}/{name} already contains reserved source_league/source_season columns")
            frame = frame.copy(deep=False)
            frame.attrs = {}
            frame["source_league"] = pd.Series(league, index=frame.index, dtype="string[pyarrow]")
            frame["source_season"] = pd.Series(season, index=frame.index, dtype="string[pyarrow]")
            pieces[name].append(frame)
        sources.append(loaded.provenance)
    combined = {name: pd.concat(parts, ignore_index=True, sort=False) for name, parts in pieces.items()}
    provenance = {"seasons": tuple(seasons), "leagues": tuple(sorted({l for _, l, _ in selected})),
                  "tables": tuple(tables), "sources": sources}
    return SeasonData(combined, provenance)
