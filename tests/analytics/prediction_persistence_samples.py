"""Synthetic ordinary publications containing completed and future fixtures together."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import Stat,IsHome,RollingMean,evaluate_features
from xdiyo_analytics.labels import TeamValue,MatchTotal,create_labels
from xdiyo_analytics.datasets import assemble_dataset

EVENT=2**63+100
HOME,AWAY=2**53+3,2**53+5
BASE=1788220800.

def publish(root):
    root=Path(root)
    for league,competition in [('Alpha',17),('Beta',42)]:
        season_id=202627;stem=f'{league}_26_27'
        version=root/f'_tables/competition={competition}/season={season_id}/versions/v1';version.mkdir(parents=True)
        matches=pd.DataFrame({'event_id':pd.Series([EVENT+i for i in range(8)],dtype='uint64'),
            'competition_id':competition,'season_id':season_id,'home_id':HOME,'away_id':AWAY,
            'home_name':league+' Home','away_name':league+' Away',
            'kickoff_utc':pd.Series([BASE,BASE+86400,BASE+3*86400,BASE+2*86400,BASE+2.5*86400,None,BASE+86400,BASE+4*86400],dtype='Float64'),
            'status':pd.Series(['finished','finished','notstarted','notstarted','finished','notstarted','inprogress','postponed'],dtype='string'),
            'round':pd.Series([1,2,5,6,3,7,4,8],dtype='Int64'),
            'is_awarded':pd.Series([False,None,False,False,True,None,False,False],dtype='boolean'),
            'home_score_current':pd.Series([2,1,None,None,3,None,0,None],dtype='Int64'),
            'away_score_current':pd.Series([1,1,None,None,0,None,0,None],dtype='Int64')})
        records=[]
        for i in range(8):
            for side,team,value in [('home',HOME,4.),('away',AWAY,3.)]:
                records.append({'event_id':EVENT+i,'period':'ALL','group_name':'Match overview','key':'cornerKicks',
                    'side':side,'team_id':team,'value':value if i in [0,3,4] else None})
        statistics=pd.DataFrame(records);statistics.event_id=statistics.event_id.astype('uint64')
        pregame=statistics[['event_id','side','team_id']].copy();pregame['position']=1
        shots=statistics[['event_id','side']].copy();shots['shot_id']=np.arange(len(shots))+100
        manifest={'schema_version':'2','parser_version':'2','version':'v1',
            'scope':{'league_id':competition,'league_name':league,'season_id':season_id,'season_start':2026,'season_end':2027},'tables':{}}
        for name,frame in {'matches':matches,'statistics':statistics,'pregame':pregame,'shots':shots,'catalog':pd.DataFrame({'name':['constant']})}.items():
            file=version/f'{name}.parquet';pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),file)
            payload=file.read_bytes();manifest['tables'][name]={'file':file.name,'rows':len(frame),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        (version/'manifest.json').write_text(json.dumps(manifest))
        (root/f'{stem}.manifest.json').write_text(json.dumps({'schema_version':'2','complete':True,'matches':8,
            'season_file':stem+'.parquet','manifest':(version/'manifest.json').relative_to(root).as_posix()}))
    return root

def snapshot(root):return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}

def event_mask(frame,event):
    return pd.Series([pd.notna(value) and value==event for value in frame.event_id.tolist()],index=frame.index,dtype=bool)

def prepared(batch,layout='match',shuffle=True):
    history=build_team_history(batch.data);stat=Stat('ALL','Match overview','cornerKicks')
    features=evaluate_features(history,{'home':IsHome(),'recent':RollingMean(stat,2)},keyed=True)
    label=create_labels(history,{'corners':MatchTotal(stat) if layout=='match' else TeamValue(stat)})['corners']
    data=assemble_dataset(features,label,layout=layout,drop_missing_targets=False)
    if shuffle:
        order=np.random.default_rng(713).permutation(len(data.X))
        for name in ['X','y','metadata']:setattr(data,name,getattr(data,name).iloc[order].reset_index(drop=True))
    return data,history
