"""Discover published tables and physical schemas without loading observations."""

from pathlib import Path
from typing import TYPE_CHECKING

from ._source import _open_source

if TYPE_CHECKING:
    import pandas as pd


# Descriptions document schema-2 table roles; actual columns come from each file.
TABLE_DESCRIPTIONS = {
    "matches": "Match identity, teams, competition, season, scheduled kickoff and scores.",
    "statistics": "Team statistics by match, period, group, statistic key and side; measures are key values, not separate columns.",
    "shots": "Individual shots: players, classifications, timing, coordinates, expected-goal values and quality indicators.",
    "pregame": "Match-specific team positions in position, plus provider form/context in value_json; not a complete league standings table.",
    "coverage": "Collection status and latest attempt status for match components and sides.",
    "availability": "Reported player availability/absence context, reasons, status and classification.",
    "lineup_teams": "Team lineup context: formation, roster status, phase and confirmation.",
    "lineup_players": "Player lineup entries: identity, playing position, shirt number, phase and substitute flag.",
    "player_statistics": "Player statistics by match/team/phase and key, with numeric and JSON values.",
    "heatmap_points": "Team spatial points with coordinates, weights, kind and source order.",
    "incidents": "Match incidents: type/class, player, side and reported timing.",
    "goal_sequences": "Goal-linked sequences: scorer/team, classification, action count, timing and shot linkage.",
    "goal_actions": "Actions within goal-linked sequences: actors, recipients/evidence, coordinates and completion/assist flags.",
    "passing_edges": "Player-to-player links within goal sequences, with evidence and action ordering.",
    "timing": "Period timing evidence, source values, quality and added-time/elapsed-time candidates.",
    "comments": "Commentary text with reported timing, side, source order and source payload.",
    "team_seasons": "Team-season membership/movement evidence, including promotion/relegation flags when supplied.",
    "legacy_fields": "Preserved legacy keys and JSON values with source context.",
}


def inspect_season(data_root, season_stem, *, record_path=None) -> "pd.DataFrame":
    """List a season's tables, descriptions and physical columns without loading rows.

    Returns a DataFrame indexed by table with rows, column_count, description,
    columns (list), column_types (dict) and column_summary (string). Optional
    record_path reuses an existing saved selection; inspection never creates it.
    Names/types come from file metadata, not measured values or a full hash audit.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = _open_source(data_root, season_stem, record_path, existing_record=True)
    entries = []
    for name, info in sorted(source.manifest["tables"].items()):
        path = source.table_path(name)
        if path.stat().st_size != info["bytes"]:
            raise ValueError(f"File size does not match the export: {name}")
        try:
            with pq.ParquetFile(path) as parquet:
                schema, rows = parquet.schema_arrow, parquet.metadata.num_rows
        except pa.ArrowException as exc:
            raise ValueError(f"Cannot inspect Parquet table {name}") from exc
        if rows != info["rows"]:
            raise ValueError(f"Row count does not match the export: {name}")
        if len(set(schema.names)) != len(schema.names):
            raise ValueError(f"Repeated column names in {name}")
        entries.append({
            "table": name, "rows": rows, "column_count": len(schema),
            "description": TABLE_DESCRIPTIONS.get(name, "No description documented; inspect the columns below."),
            "columns": schema.names,
            "column_types": {field.name: str(field.type) for field in schema},
            "column_summary": ", ".join(f"{field.name}: {field.type}" for field in schema),
        })
    catalogue = pd.DataFrame(entries).set_index("table")
    catalogue.attrs["source"] = {"season": season_stem, "version": source.manifest["version"],
                                  "manifest_path": str(source.path)}
    return catalogue
