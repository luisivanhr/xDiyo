"""Bind the same workbook's labelled opening draw prices, outside model inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def prepare(run, snapshot):
    run=Path(run);snapshot=Path(snapshot)
    manifest=json.loads((snapshot/'manifest.json').read_text())
    expected='a91c03e33820453892708d08baa3a070a8e6ddc74e4ced23296a6ba8542dbf0f'
    if not any(f['sha256']==expected for f in manifest['source_files']):raise ValueError('Wrong workbook odds source.')
    files=[];frames=[]
    for year in ('22_23','23_24','24_25','25_26'):
        for path in sorted(snapshot.glob(f'league=*/season={year}/Odds/quotes-*.parquet')):
            q=pd.read_parquet(path,filters=[('market','==','1x2'),('selection','==','draw'),('quote_type','==','opening')])
            frames.append(q);files.append(dict(path=str(path),sha256=digest(path)))
    quotes=pd.concat(frames,ignore_index=True);quotes['match_id']=quotes.match_id.astype('int64')
    if quotes.match_id.duplicated().any():raise ValueError('Multiple opening draw quotes for workbook fixture.')
    if not quotes.period.eq('FT').all():raise ValueError('Wrong odds period.')
    schedule=pd.read_parquet(run/'data/original13/schedule.parquet')
    schedule=schedule.loc[schedule.issuance_at.ge(pd.Timestamp('2022-07-01',tz='Europe/Berlin'))].copy()
    keys=['competition_id','season_id','event_id']
    output=schedule[keys+['match_id','result_available_at']].merge(quotes[['match_id','decimal_odds','status']],on='match_id',how='left',validate='one_to_one')
    output['draw_opening_odds']=output.decimal_odds.where(output.status.eq('valid')&output.decimal_odds.gt(1),np.nan)
    destination=run/'data/financial_inputs.parquet'
    if destination.exists():raise ValueError('Financial inputs already exist.')
    output[keys+['draw_opening_odds','result_available_at']].to_parquet(destination,index=False)
    receipt=dict(source_workbook_sha256=expected,files=files,rows=len(output),valid_prices=int(output.draw_opening_odds.notna().sum()),
        missing_prices=int(output.draw_opening_odds.isna().sum()),output_sha256=digest(destination),
        quote_timing='vendor opening/prematch label only; quote capture timestamps and bookmaker unavailable',
        financial_scope='original13 only; retrospective simulation, not executable quotes',
        settlement_time='same actual kickoff retrospective proxy as experiment',model_features=False)
    (run/'data/OPENING_ODDS_PROVENANCE.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('snapshot');prepare(**vars(p.parse_args()))
