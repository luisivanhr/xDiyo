from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def audit_legacy(directory):
    csv.field_size_limit(20_000_000)
    root=Path(directory).resolve()
    entries=[]
    ids={}
    for path in sorted(root.glob("*.csv")):
        with path.open(encoding="utf-8-sig",newline="") as stream:
            reader=csv.DictReader(stream)
            keys=[r.get("match_id") for r in reader]
            fields=reader.fieldnames
        if not keys or "match_id" not in fields:
            entries.append({"file":path.name,"kind":"other_csv","rows":len(keys),"columns":fields,
                            "bytes":path.stat().st_size,"sha256":hash_file(path)})
            continue
        ids[path.name]=keys
        entries.append({"file":path.name,"kind":"lineups" if path.stem.endswith("_lineups") else "matches",
            "rows":len(keys),"columns":fields,"duplicate_ids":len(keys)-len(set(keys)),
            "missing_ids":sum(not k for k in keys),"bytes":path.stat().st_size,"sha256":hash_file(path)})
    joins=[]
    for name,keys in ids.items():
        if name.endswith("_lineups.csv"): continue
        lineup=ids.get(name.replace(".csv","_lineups.csv"))
        joins.append({"file":name,"lineups_present":lineup is not None,
                      "same_id_set":set(keys)==set(lineup) if lineup is not None else None,
                      "same_order":keys==lineup if lineup is not None else None})
    return {"source":str(root),"files":entries,"joins":joins,
            "total_bytes":sum(r["bytes"] for r in entries),
            "match_rows":sum(r["rows"] for r in entries if r["kind"]=="matches"),
            "lineup_rows":sum(r["rows"] for r in entries if r["kind"]=="lineups")}
