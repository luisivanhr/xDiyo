"""Read-only real-schema and publication parity audit; no fitting or publication."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xdiyo_analytics.data import SeasonData, homogenize_international_statistics, international_statistic_mapping
from xdiyo_analytics.data.statistics_backfill import STATISTICS, PERIODS
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.odds.workbook import _Workbook


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def audit(native_root, workbook, catalogue, output):
    output.mkdir(parents=True,exist_ok=True)
    pointer = native_root/'Premier_League_24_25.manifest.json'
    manifest_path = native_root/json.loads(pointer.read_text())['manifest']
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent/manifest['tables']['statistics']['file']
    assert sha(path) == manifest['tables']['statistics']['sha256']
    native = pd.read_parquet(path,columns=['period','group_name','key','name']).drop_duplicates()
    known = set(native.itertuples(index=False,name=None))
    book_hash = sha(workbook)
    book = _Workbook(workbook)
    try:
        headers = {s:list(book.headers(s).values()) for s in ('Statistics_FT','Statistics_1H','Statistics_2H','Odds')}
    finally:
        book.archive.close()
    rows = []
    for p,n in PERIODS.items():
        sheet = 'Statistics_'+p.upper()
        for field,(key,label,group,_) in STATISTICS.items():
            assert (n,group,key,label) in known, (n,group,key,label)
            for side in ('home','away'):
                source = f'{side}_{field}_{p}'
                assert source in headers[sheet], source
                rows.append(dict(source=source,period=n,group=group,key=key,label=label,
                                 home_perspective=international_statistic_mapping('home')[source],
                                 away_perspective=international_statistic_mapping('away')[source]))
    unmatched = {s:[c for c in cols if c not in international_statistic_mapping('home')] for s,cols in headers.items()}
    for cols in headers.values():
        renamed = homogenize_international_statistics(pd.DataFrame(columns=cols),perspective='home')
        for col in cols:
            if col not in international_statistic_mapping('home'):
                assert col in renamed
    publications = []
    for stat_path in sorted((catalogue/'international_dataset').glob('*/statistics.parquet')):
        match_path = stat_path.parent/'matches.parquet'
        before = {p.name:sha(p) for p in stat_path.parent.glob('*') if p.is_file()}
        stats = pd.read_parquet(stat_path)
        matches = pd.read_parquet(match_path)
        # Missing actual kickoff remains missing; no timing evidence is invented.
        matches['kickoff_utc'] = pd.to_datetime(matches.kickoff_at,utc=True).map(
            lambda x:x.timestamp() if pd.notna(x) else np.nan)
        source = SeasonData({'matches':matches,'statistics':stats},{})
        expected = build_team_history(source)
        wide = matches.copy()
        labels = {v:k for k,v in international_statistic_mapping('home').items()}
        cells = stats.pivot(index='event_id',columns=['side','period','group_name','key'],values='value')
        for (side,period,group,key) in cells.columns:
            role = 'team' if side == 'home' else 'opponent'
            target = f'{role}::{period}::{group}::{key}::value'
            assert target in labels, target
            wide[labels[target]] = cells[(side,period,group,key)].reindex(matches.event_id).to_numpy()
        actual = build_team_history(SeasonData({'matches':wide},{}))
        assert actual.attrs['stat_columns'] == expected.attrs['stat_columns']
        for col in expected:
            a,b = actual[col],expected[col]
            if col == 'side':a,b = a.astype('string'),b.astype('string')
            pd.testing.assert_series_equal(a,b)
        assert before == {p.name:sha(p) for p in stat_path.parent.glob('*') if p.is_file()}
        publications.append(dict(version=stat_path.parent.name,matches=len(matches),statistic_rows=len(stats),
                                 compared_team_rows=len(actual),compared_statistic_columns=len(expected.attrs['stat_columns']),
                                 native_names_already_present=True,history_value_identity_missingness_parity=True,
                                 publication_files_unchanged=True,file_hashes=before))
    assert publications, 'No international publications audited'
    assert sha(workbook) == book_hash
    with (output/'mapping.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    repo = Path(__file__).resolve().parents[1]
    receipt = dict(mapping_count=len(rows),native_statistic_sha256=sha(path),
        native_schema_evidence=native.loc[native.key.isin([v[0] for v in STATISTICS.values()])].to_dict('records'),
        workbook_sha256=book_hash,unmapped_original_headers=unmatched,publications=publications,
        original_inputs_unchanged=True,training_runs=0,
        source_sha256={str(p.relative_to(repo)):sha(p) for p in [Path(__file__),
            repo/'src/xdiyo_analytics/data/international_statistics.py',repo/'src/xdiyo_analytics/data/statistics_backfill.py',
            repo/'src/xdiyo_analytics/histories/team.py']})
    # Keep physical Windows checkout hashes and Git-normalized source hashes
    # distinct: CRLF checkout bytes need not equal the committed LF blob.
    receipt['source_lf_sha256'] = {name:hashlib.sha256((repo/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                   for name in receipt['source_sha256']}
    (output/'schema-verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(mapping_count=len(rows),publications=len(publications),parity=True,training_runs=0)))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('native-root','workbook','catalogue','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.native_root,a.workbook,a.catalogue,a.output)
