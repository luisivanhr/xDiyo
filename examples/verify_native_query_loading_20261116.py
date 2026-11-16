"""Real native identity/query and GAT preparation checks, without training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq
from xdiyo_analytics.data import NativeLibrary
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.ui import default_recipe, prepare_recipe
from xdiyo_analytics.ui.recipe import node


def main():
    root=Path('data/xDiyo_data');library=NativeLibrary(root);seen={};duplicates=[]
    for pub in library.publications:
        source=_open_source(root,pub['stem'])
        frame=pq.read_table(source.table_path('matches'),columns=['competition_id','season_id','event_id']).to_pandas()
        for key in frame.itertuples(index=False,name=None):
            if key in seen:duplicates.append([list(key),seen[key],pub['stem']])
            seen[key]=pub['stem']
    assert not duplicates,duplicates
    named=library.load(query={'competitions':'HNL','seasons':61243},tables=['matches','team_seasons'])
    by_id=library.load(query={'competitions':170,'seasons':61243},tables=['matches','team_seasons'])
    pd.testing.assert_frame_equal(named.matches,by_id.matches)
    runs=[]
    for stem in ('HNL_24_25','Peru_Liga_1_24_24'):
        recipe=default_recipe(str(root))
        recipe['data']=dict(data_root=str(root),query={'publications':stem},tables=['matches','team_seasons'])
        recipe.update(features={'home':node('features.IsHome')},preprocessors=[],target_transformer=None,adapter=None,
            labels={'outcome':node('labels.Outcome',perspective='home')},target='outcome',
            split=node('splits.TemporalSplit',train_size=10,test_size=5,unit='rounds'),
            model=node('graphs.TemporalGraphModel',round_stage_column='countdown_calendar_id',
                config=node('graphs.TemporalGraphConfig',timezone='UTC',round_keys=['competition_id','season_id','round_stage','round'])))
        prepared=prepare_recipe(recipe)
        metadata=prepared.dataset.metadata
        assert metadata.feature_cutoff_at.le(metadata.kickoff_at.dt.normalize()).all()
        assert prepared.outputs['graph_inputs'] is not None
        runs.append(dict(publication=stem,targets=len(metadata),folds=len(prepared.split_plan.folds),
                         phases=sorted(metadata.round_stage.unique().tolist()),fit_performed=False))
    receipt=dict(command=subprocess.list2cmdline([sys.executable,*sys.argv]),unique_fixture_identities=len(seen),
                 duplicate_identities=duplicates,name_id_frame_parity=True,gat_preparations=runs,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output=Path('docs/analytics/native_preflight_20261116/query-loading.json')
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
