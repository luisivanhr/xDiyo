"""Native integration gates for canonical selections, with a closed requirement graph."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import numpy as np
import pandas as pd

SAFE={'MatchScore','IsHome','ForAgainst','H2H','Lag','RollingMean','RollingStd','RollingSkewness','RollingZScore','EMA',
      'Constant','Column','Abs','Sum','Product','Difference','Ratio','RollingWeightedMean','Outcome','MatchTotal','TeamValue'}


def feature_requirements(value):
    requirements={'goals'}
    def visit(node):
        if isinstance(node,dict):
            if 'component' in node:
                name=node['component'].split('.')[-1];params=node.get('params',{})
                check(name,params);visit(params)
            else:
                for v in node.values():visit(v)
        elif isinstance(node,(list,tuple)):
            for v in node:visit(v)
        elif is_dataclass(node):
            params={f.name:getattr(node,f.name) for f in fields(node)}
            check(type(node).__name__,params);visit(params)
    def check(name,params):
        if name=='Stat':
            if params.get('period') is None or params.get('field','value')!='value':raise ValueError('Named datasets require explicit statistic period and numeric value')
            requirements.add(f"stat:{params['period']}:{params['key']}")
        elif name=='TeamMovement':
            requirements.add('movements')
        elif name not in SAFE:
            raise ValueError(f'Named dataset dependency {name} needs a verified capability contract; membership, ratings, graph and standings inputs are not inferred')
    visit(value)
    return requirements


def validate_feature_selection(history,features):
    selection=history.attrs.get('data_selection')
    if selection:
        missing=feature_requirements(features)-set(selection['requirements'])
        if missing:raise ValueError(f'Rebuild frozen selection for feature requirements: {sorted(missing)}')


def named_recipe_data(recipe):
    from .selection import resolve_selection,selected_history
    options=recipe['data_selection']
    data=resolve_selection(options['root'],options['selection'])
    required=feature_requirements([recipe.get('features',{}),recipe.get('labels',{})])
    if required-set(data.provenance['data_selection']['requirements']):raise ValueError('Rebuild named selection for the complete feature/label requirements')
    blocked=['ratings','feature_preset','context_features','identity_features']
    if any(recipe.get(k) for k in blocked) or recipe.get('model',{}).get('component') in ('graphs.TemporalGraphModel','neural.HeatmapCNNModel'):
        raise ValueError('Selected named dataset lacks verified membership, schedule or spatial/model dependencies')
    if recipe.get('stat_selection') or recipe.get('history') or recipe.get('feature_options') or recipe.get('cutoff_hours') is not None:
        raise ValueError('Named selections own their frozen history and timing; remove legacy history/cutoff overrides')
    return data,selected_history(data)


def select_targets(dataset,selection):
    mask=dataset.metadata.event_id.isin(selection['target_ids'])
    if set(dataset.metadata.loc[mask,'event_id'])!=set(selection['target_ids']):raise ValueError('Assembly dropped frozen target fixtures')
    dataset.X=dataset.X.loc[mask].reset_index(drop=True);dataset.y=dataset.y.loc[mask].reset_index(drop=True)
    dataset.metadata=dataset.metadata.loc[mask].reset_index(drop=True)
    dataset.definitions['data_selection']=deepcopy(selection)
    return dataset


def frozen_split(dataset,selection):
    from ...splits import Fold,SplitPlan
    if selection.get('fold_membership') is not None:
        def rows(ids):
            return np.asarray([i for event in ids for i in np.flatnonzero(np.asarray([int(e)==event for e in dataset.metadata.event_id]))],dtype=int)
        folds=[Fold(rows(f['train_ids']),rows(f['evaluation_ids']),rows(f['score_ids']),deepcopy(f['metadata'])) for f in selection['fold_membership']]
        return validate_selected_plan(dataset,SplitPlan(folds,len(dataset.metadata),np.arange(len(dataset.metadata))))
    if not selection['train_ids'] and not selection['evaluation_ids']:return None
    train=np.flatnonzero(dataset.metadata.event_id.isin(selection['train_ids']))
    test=np.flatnonzero(dataset.metadata.event_id.isin(selection['evaluation_ids']))
    if not len(train) or not len(test):raise ValueError('Fixed train/evaluation populations must both be nonempty for fitting')
    metadata=dataset.metadata
    if 'result_available_at' not in metadata:raise ValueError('Selected labels need result availability bounds')
    if pd.to_datetime(metadata.iloc[train].result_available_at,utc=True).max()>pd.to_datetime(metadata.iloc[test].kickoff_at,utc=True).min():
        raise ValueError('Training labels are unavailable at the first evaluation cutoff')
    return SplitPlan([Fold(train,test,test.copy(),dict(data_selection=selection['fingerprint']))],len(metadata),np.arange(len(metadata)))


def validate_selected_plan(dataset,plan):
    for fold in plan.folds:
        if len(fold.train) and len(fold.test):
            release=pd.to_datetime(dataset.metadata.iloc[fold.train].result_available_at,utc=True)
            cutoff=pd.to_datetime(dataset.metadata.iloc[fold.test].kickoff_at,utc=True)
            if release.isna().any() or cutoff.isna().any() or release.max()>cutoff.min():
                raise ValueError('Named selection training labels unavailable at evaluation cutoff')
    return plan
