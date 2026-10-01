"""Create an importable, entirely synthetic odds recipe; never train a model.

Run: python examples/odds_synthetic.py --output <local-output-directory>
Then import <output>/recipe.json in the experiment builder. No paid data is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xdiyo_analytics.odds import extract_odds, build_fixture_crosswalk, save_crosswalk
from xdiyo_analytics.odds.crosswalk import native_fixture_metadata
from xdiyo_analytics.odds.workbook import MARKETS
from xdiyo_analytics.ui import default_recipe
from xdiyo_analytics.ui.recipe import node, export_python, export_notebook


def create_example(output):
    output = Path(output).resolve()
    root = output/'data/xDiyo_data'
    if (output/'recipe.json').exists():
        raise ValueError('Choose a fresh output directory; an example recipe already exists')
    vendor_rows = []
    seasons = ['22_23','23_24','24_25']
    for year, season in zip([2022,2023,2024], seasons):
        stem = 'Synthetic_'+season
        directory = root/f'_tables/competition=17/season={year}/versions/v1'
        directory.mkdir(parents=True, exist_ok=True)
        matches, stats = [], []
        for i in range(8):
            event = year*100+i
            kickoff = pd.Timestamp(f'{year}-09-01', tz='UTC')+pd.Timedelta(days=7*i)
            matches.append(dict(event_id=event,competition_id=17,season_id=year,home_id=101,away_id=202,home_name='Example Home',away_name='Example Away',
                kickoff_utc=int(kickoff.timestamp()),status='finished',round=i+1,is_awarded=False,
                home_score_current=1,away_score_current=0))
            for side,team,value in [('home',101,3+i%4),('away',202,2+i%3)]:
                stats.append(dict(event_id=event,period='ALL',group_name='Match overview',key='cornerKicks',side=side,team_id=team,value=value))
            vendor_rows.append([str(event+1000000),kickoff.strftime('%Y-%m-%d %H:%M:%S'),'England','Premier League',
                                f'{year}/{year+1}','Example Home','Example Away',*['1.9']*8])
        manifest=dict(schema_version='2',parser_version='2',version='v1',
            scope=dict(league_id=17,league_name='Synthetic',season_id=year,season_start=year,season_end=year+1),tables={})
        for name, rows in [('matches',matches),('statistics',stats)]:
            path=directory/(name+'.parquet')
            table=pa.Table.from_pandas(pd.DataFrame(rows),preserve_index=False)
            pq.write_table(table,path)
            manifest['tables'][name]=dict(file=path.name,rows=len(rows),bytes=path.stat().st_size,columns=table.column_names,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        (root/(stem+'.manifest.json')).write_text(json.dumps(dict(schema_version='2',complete=True,matches=8,
            season_file=stem+'.parquet',manifest=(directory/'manifest.json').relative_to(root).as_posix())),encoding='utf-8')
    headers=['match_id','start_datetime','country','league','season','home_team','away_team',*MARKETS['Corners_Closing_Odds']]
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    def column(i):
        result=''
        while i:
            i,r=divmod(i-1,26)
            result=chr(65+r)+result
        return result
    rows=''.join(f'<row r="{n}">'+''.join(f'<c r="{column(i)}{n}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'
        for i,v in enumerate(row,1))+'</row>' for n,row in enumerate([headers,*vendor_rows],1))
    workbook=output/'synthetic_prices.xlsx'
    with ZipFile(workbook,'w') as archive:
        archive.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Corners_Closing_Odds" r:id="r1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr('xl/worksheets/sheet1.xml',f'<worksheet xmlns="{ns}"><sheetData>{rows}</sheetData></worksheet>')
    snapshot=extract_odds([workbook],data_root=root,output_root=output/'data/odds/database',seasons=seasons,
                            league_aliases={('England','Premier League'):'Synthetic'})
    native=native_fixture_metadata(root,seasons=seasons)
    walk=build_fixture_crosswalk(snapshot,native,seasons=seasons)
    crosswalk=save_crosswalk(walk,snapshot=snapshot,output_root=output/'data/odds/mappings')
    recipe=default_recipe(str(root))
    recipe.update(name='Synthetic odds integration',output_dir=str(output/'experiments'))
    recipe['data'].update(leagues=['Synthetic'],tables=['matches','statistics'])
    recipe['model']=node('sklearn.linear_model.LogisticRegression',max_iter=1000)
    recipe['prediction_methods']=['predict','predict_proba']
    odds=node('input.OddsSeries',snapshot=str(output/'data/odds'),crosswalk=str(crosswalk),seasons=seasons,
              market='corners',selection='over',line=9.5,quote_type='closing',settlement_confirmed=True)
    option=node('labels.BetOption',source=recipe['labels']['corners'],selection='over',line=9.5)
    recipe['post_reporters']={
        'Decisions':node('reporting.BetOutcomeReporter',type='overall',partition='test',pooling='last',
                        offers={'Over 9.5':node('evaluation.BetOffer',option=option,odds=odds)},
                        policy=node('evaluation.HighestExpectedProfit')),
        'Accounting':node('reporting.BetPerformanceReporter',type='overall',partition='test',pooling='last',source='Decisions'),
    }
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n',encoding='utf-8')
    (output/'experiment.py').write_text(export_python(recipe),encoding='utf-8')
    (output/'experiment.ipynb').write_text(json.dumps(export_notebook(recipe),indent=2),encoding='utf-8')
    return output/'recipe.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    print(create_example(parser.parse_args().output))
