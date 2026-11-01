"""Reproduce aggregate input audits and a no-fit frozen-history preparation example."""
import argparse
import importlib.util
import json
from pathlib import Path
import pandas as pd
from xdiyo_analytics.data.catalogue import load_snapshot, select_data, inspect_coverage
from xdiyo_analytics.data.catalogue.importers import inspect_historical_revisions
from xdiyo_analytics.data.catalogue.selection import _bounds
from xdiyo_analytics.ui import prepare_recipe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--catalogue',required=True);parser.add_argument('--v1',required=True);parser.add_argument('--v2',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.catalogue);receipt=json.loads((root/'import-receipt.json').read_text());snapshots=receipt['versions']
    spec=importlib.util.spec_from_file_location('named_example',Path(__file__).resolve().parents[3]/'examples/named_dataset_recipe.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native,_,_=load_snapshot(root,'comprehensive_dataset',snapshots['comprehensive_dataset'])
    fixtures=native['matches'].query('competition_id==17').sort_values(['date','event_id'])
    train=fixtures.iloc[:4].event_id.tolist()
    boundary=pd.Timestamp(fixtures.iloc[3]['date'])+pd.Timedelta(days=7)
    evaluation=fixtures.loc[pd.to_datetime(fixtures.date)>=boundary].iloc[:4].event_id.tolist()
    single=module.make_recipe(root,{'comprehensive_dataset':snapshots['comprehensive_dataset']},train,evaluation,mode='single_dataset')
    expanded=module.make_recipe(root,snapshots,train,evaluation)
    before=prepare_recipe(single);after=prepare_recipe(expanded)
    assert before.dataset.metadata.event_id.tolist()==after.dataset.metadata.event_id.tolist()
    for a,b in zip(before.split_plan.folds,after.split_plan.folds):
        assert (a.train==b.train).all() and (a.test==b.test).all()
    x=before.dataset.X;y=after.dataset.X
    changed=(~(x.eq(y)|(x.isna()&y.isna()))).sum().sum()
    assert len(set(expanded['data_selection']['selection']['history_ids'])-set(single['data_selection']['selection']['history_ids']))>0
    union=select_data(root,snapshots=snapshots,mode='union')
    last_cutoff=max(_bounds(r,'retrospective_reported_goals')[0] for r in fixtures.loc[fixtures.event_id.isin(evaluation)].to_dict('records'))
    baseline_eligible=sum(_bounds(r,'retrospective_reported_goals')[1]<=last_cutoff for r in native['matches'].loc[native['matches'].event_id.isin(single['data_selection']['selection']['history_ids'])].to_dict('records'))
    result=dict(revision_audit=inspect_historical_revisions(args.v1,args.v2),versions=snapshots,
                union_targets=len(union['target_ids']),union_unique=len(set(union['target_ids'])),
                fixed_targets=train+evaluation,changed_feature_cells=int(changed),
                baseline_history_population=len(single['data_selection']['selection']['history_ids']),baseline_history_available_by_last_target=baseline_eligible,extended_history=len(expanded['data_selection']['selection']['history_ids']),
                added_history=len(set(expanded['data_selection']['selection']['history_ids'])-set(single['data_selection']['selection']['history_ids'])),
                frozen_membership_and_split_equal=True,models_fitted=0,
                coverage={n:inspect_coverage(root,n,v) for n,v in snapshots.items()})
    for name,recipe in [('single',single),('extended',expanded)]:
        (root/(name+'-recipe.json')).write_text(json.dumps(recipe,indent=2)+'\n',encoding='utf-8')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='coverage'},indent=2))

if __name__=='__main__':main()
