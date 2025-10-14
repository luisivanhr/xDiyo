from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import pandas as pd


def _collect_league_seasons(league_name: str, data_dir: str) -> List[Tuple[int, int, str]]:
    """Return sorted season files for a given league."""
    seasons: List[Tuple[int, int, str]] = []
    prefix = f"{league_name}_"
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.csv') or file_name.endswith('_lineups.csv'):
            continue
        if not file_name.startswith(prefix):
            continue

        remainder = file_name[len(prefix):-4]
        parts = remainder.split('_')
        if len(parts) != 2:
            continue

        try:
            start_year = int(parts[0])
            end_year = int(parts[1])
        except ValueError:
            continue

        seasons.append((start_year, end_year, os.path.join(data_dir, file_name)))

    seasons.sort(key=lambda entry: entry[0])
    return seasons


def _add_promotion_columns(df: pd.DataFrame, promoted: Sequence[str], demoted: Sequence[str]) -> pd.DataFrame:
    promoted_set = set(promoted)
    demoted_set = set(demoted)

    df = df.copy()
    df['home_got_promoted'] = df['home_team'].isin(promoted_set).astype(int)
    df['away_got_promoted'] = df['away_team'].isin(promoted_set).astype(int)
    df['home_got_demoted'] = df['home_team'].isin(demoted_set).astype(int)
    df['away_got_demoted'] = df['away_team'].isin(demoted_set).astype(int)
    return df


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _save_csv(df: pd.DataFrame, output_dir: str, file_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_path, index=False)


def _unique_team_names(df: pd.DataFrame) -> set[str]:
    teams = pd.concat([df['home_team'], df['away_team']], ignore_index=True)
    unique = teams.dropna().unique()
    return set(unique)


def get_promotions_and_demotions(
    top_league: str,
    bottom_league: Optional[str] = None,
    top_promoted: Optional[Sequence[str]] = None,
    bottom_promoted: Optional[Sequence[str]] = None,
    bottom_demoted: Optional[Sequence[str]] = None,
    data_dir: str = 'xDiyo_data_heatmaps',
    output_dir: str = 'xDiyo_data_promotions'
) -> None:
    """Annotate each season with promotion and demotion dummy columns."""

    top_seasons = _collect_league_seasons(top_league, data_dir)
    if not top_seasons:
        raise ValueError(f'No seasons found for top league {top_league}')

    bottom_seasons: List[Tuple[int, int, str]] = []
    if bottom_league:
        bottom_seasons = _collect_league_seasons(bottom_league, data_dir)
        if not bottom_seasons:
            print(f'No seasons found for bottom league {bottom_league}.')

    top_promoted = list(top_promoted or [])
    bottom_promoted = list(bottom_promoted or [])
    bottom_demoted = list(bottom_demoted or [])

    # Load earliest seasons
    top_before_year, _, top_before_path = top_seasons[0]
    top_before_df = _load_csv(top_before_path)
    print(
        f'Earliest top league season for {top_league}: '
        f'{os.path.basename(top_before_path)} (start_year: {top_before_year})'
    )
    print(
        f'Initial top league promoted teams for {top_before_year}: '
        f'{sorted(top_promoted)}'
    )
    top_before_df = _add_promotion_columns(top_before_df, top_promoted, [])
    _save_csv(top_before_df, output_dir, top_before_path)

    bottom_before_df: Optional[pd.DataFrame] = None
    bottom_before_year: Optional[int] = None
    bottom_idx_before = 0
    if bottom_seasons:
        bottom_before_year, _, bottom_before_path = bottom_seasons[0]
        bottom_before_df = _load_csv(bottom_before_path)
        print(
            f'Earliest bottom league season for {bottom_league}: '
            f'{os.path.basename(bottom_before_path)} (start_year: {bottom_before_year})'
        )
        print(
            f'Initial bottom league promoted teams for {bottom_before_year}: '
            f'{sorted(bottom_promoted)}'
        )
        print(
            f'Initial bottom league demoted teams for {bottom_before_year}: '
            f'{sorted(bottom_demoted)}'
        )
        bottom_before_df = _add_promotion_columns(bottom_before_df, bottom_promoted, bottom_demoted)
        _save_csv(bottom_before_df, output_dir, bottom_before_path)

    top_idx_before = 0

    while top_idx_before + 1 < len(top_seasons):
        top_idx_after = top_idx_before + 1
        top_after_year, _, top_after_path = top_seasons[top_idx_after]
        top_after_df = _load_csv(top_after_path)

        bottom_after_df = None
        bottom_after_year: Optional[int] = None

        if bottom_before_df is not None and bottom_before_year is not None:
            top_year_matches_bottom = top_seasons[top_idx_before][0] == bottom_before_year
            if top_year_matches_bottom and bottom_idx_before + 1 < len(bottom_seasons):
                bottom_after_year, _, bottom_after_path = bottom_seasons[bottom_idx_before + 1]
                bottom_after_df = _load_csv(bottom_after_path)
            elif top_seasons[top_idx_before][0] < bottom_before_year:
                bottom_after_df = None

        # Compute promoted teams for top league
        top_before_names = _unique_team_names(top_before_df)
        top_after_names = _unique_team_names(top_after_df)
        top_promoted_names = sorted(top_after_names - top_before_names)
        print(
            f'Top league promoted teams for season starting {top_after_year}: '
            f'{top_promoted_names}'
        )

        top_after_df = _add_promotion_columns(top_after_df, top_promoted_names, [])
        _save_csv(top_after_df, output_dir, top_after_path)

        if bottom_after_df is not None and bottom_after_year is not None:
            bottom_before_names = _unique_team_names(bottom_before_df)
            bottom_after_names = _unique_team_names(bottom_after_df)
            delta_bottom_names = bottom_after_names - bottom_before_names
            bottom_promoted_names = sorted(delta_bottom_names - top_before_names)
            bottom_demoted_names = sorted(delta_bottom_names & top_before_names)
            print(
                f'Bottom league promoted teams for season starting {bottom_after_year}: '
                f'{bottom_promoted_names}'
            )
            print(
                f'Bottom league demoted teams for season starting {bottom_after_year}: '
                f'{bottom_demoted_names}'
            )

            bottom_after_df = _add_promotion_columns(bottom_after_df, bottom_promoted_names, bottom_demoted_names)
            _save_csv(bottom_after_df, output_dir, bottom_after_path)

            print(
                f'Advancing bottom league from {bottom_before_year} to {bottom_after_year}'
            )
            bottom_before_df = bottom_after_df
            bottom_before_year = bottom_after_year
            bottom_idx_before += 1
        elif bottom_before_df is not None and bottom_before_year is not None:
            print(
                f'Bottom league remains at {bottom_before_year} while processing '
                f'top league season starting {top_after_year}.'
            )

        print(
            f'Advancing top league from {top_seasons[top_idx_before][0]} to {top_after_year}'
        )
        top_before_df = top_after_df
        top_idx_before = top_idx_after