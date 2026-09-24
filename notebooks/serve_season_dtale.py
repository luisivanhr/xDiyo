"""Temporary interactive inspection of one season; not an analytics loader."""
from pathlib import Path
import hashlib
import json
import threading

import dtale
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/xDiyo_data"
STEM = "Premier_League_24_25"
STATE = PROJECT / "docs/analytics/dtale_season_session.json"

publication = json.loads((ROOT / f"{STEM}.manifest.json").read_text(encoding="utf-8"))
source = ROOT / publication["season_file"]
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == publication["sha256"], "Season file hash mismatch"
table = pq.ParquetFile(source).read()
assert table.num_rows == publication["matches"]

# D-Tale is a scalar grid. Keep every nested value as complete JSON text;
# converting to text is for display only and never writes the source file.
nested = [field.name for field in table.schema if pa.types.is_list(field.type)]
frame = table.select([name for name in table.column_names if name not in nested]).to_pandas()
for name in nested:
    frame[name] = [
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if value is not None else None
        for value in table[name].to_pylist()
    ]
frame = frame[table.column_names]

view = dtale.show(
    frame,
    name="Premier League 2024 25 all columns",
    host="127.0.0.1",
    open_browser=False,
    notebook=False,
    reaper_on=False,
    allow_cell_edits=False,
    auto_hide_empty_columns=False,
    inplace=False,
)
assert view is not None, "D-Tale did not start"
STATE.write_text(json.dumps({
    "url": view._main_url,
    "source": str(source),
    "sha256": digest,
    "rows": len(frame),
    "columns": list(frame.columns),
    "nested_columns_displayed_as_json": nested,
    "source_modified": False,
}, indent=2), encoding="utf-8")
print(f"Ready: {view._main_url} ({len(frame)} rows, {len(frame.columns)} columns)", flush=True)

# Keep this inspection process alive independently of the notebook kernel.
threading.Event().wait()
