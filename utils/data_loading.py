"""Data-only loading boundary, independent of torch and model construction."""
from pathlib import Path
import re

import pandas as pd

from scraping.sofascore.legacy import load_season

PROJECT_DATA = Path(__file__).resolve().parents[1]/"data"/"xDiyo_data_heatmaps"


def merge_seasons(leagues, home_flags=False, *, data_dir=None, include_lineups=True, team_season_flags=None):
    root=Path(data_dir) if data_dir else Path("xDiyo_data_heatmaps") if Path("xDiyo_data_heatmaps").is_dir() else PROJECT_DATA
    frames=[]
    league_indices={}
    flags = None
    if team_season_flags is not None:
        from scraping.sofascore.movements import read_flags
        flags = read_flags(team_season_flags)
    for source in leagues:
        candidate=Path(source)
        if candidate.is_file():
            path=candidate
        else:
            path=root/candidate
            if not path.is_file():
                path=next((root/(str(source)+ext) for ext in (".parquet",".csv")
                           if (root/(str(source)+ext)).is_file()),root/(str(source)+".csv"))
        frame=load_season(path,include_lineups=include_lineups)
        name=re.sub(r"_\d{2,4}_\d{2,4}$","",path.stem)
        if flags is not None:
            from utils.team_seasons import join_team_season_flags
            if "competition_id" not in frame:
                import json
                from scraping.sofascore.movements import DEFAULT_REGISTRY
                registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
                if name not in registry:
                    raise ValueError(f"No stable competition ID configured for {name}")
                frame["competition_id"] = registry[name]["competition_id"]
            frame = join_team_season_flags(frame, flags)
        if path.suffix==".json" and "competition_id" in frame and not frame.empty:
            name=f"competition_{frame.competition_id.iloc[0]}"
        league_indices.setdefault(name,len(league_indices))
        frame=frame.copy()
        for side in ("home","away"):
            col=side+"_standing"
            if col not in frame:
                frame[col]=float("nan")
            values=pd.to_numeric(frame[col],errors="coerce")
            maximum=values.max()
            frame[col]=values/maximum if pd.notna(maximum) and maximum>0 else float("nan")
        rounds=pd.to_numeric(frame["round"],errors="coerce")
        maximum=rounds.max()
        frame["normalized_round"]=rounds/maximum if pd.notna(maximum) and maximum>0 else float("nan")
        # Preserve the old two-digit feature convention for legacy season labels;
        # read four-digit provider labels correctly instead of truncating to 20.
        season=frame.get("season_year",pd.Series(index=frame.index,dtype="string")).astype("string")
        start=pd.to_numeric(season.str.extract(r"^(\d{4}|\d{2})",expand=False),errors="coerce")
        frame["season_start"]=start.astype("Int64")
        if home_flags:
            frame["home_is_home"],frame["away_is_home"]=1,0
        frame["league_index"]=league_indices[name]
        frame=frame.drop(columns=[c for c in frame if c=="index" or c.startswith("Unnamed:")])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(),league_indices
    merged=pd.concat(frames,ignore_index=True,sort=False)
    # Keep a union schema and missing observations. Imputation belongs to the
    # caller's feature policy, not the collection loader.
    merged.attrs["numeric_availability"]=merged.select_dtypes(include="number").notna().to_dict("list")
    return merged,league_indices
