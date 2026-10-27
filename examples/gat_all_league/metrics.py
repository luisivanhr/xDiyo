"""Retained outcome/goal scoring plus normalized ordered three-class RPS."""
import numpy as np
from gat_suite import metrics as base
from gat_cross_league.stable_scoring import install
install()

P_COLUMNS=base.P_COLUMNS
C_COLUMNS=base.C_COLUMNS
ADDITIVE=[*base.ADDITIVE,'rps']
NAMES=[*ADDITIVE,'conditional_winner_log_loss','draw_auc','draw_ap']

def rps(probability,truth):
    p=np.asarray(probability,dtype=float);y=np.asarray(truth,dtype=int)
    if p.ndim!=2 or p.shape!=(len(y),3) or not np.isfinite(p).all() or (p<0).any() or (y<0).any() or (y>2).any():
        raise ValueError('RPS requires aligned finite H/D/A probabilities and class indices')
    np.testing.assert_allclose(p.sum(1),1,atol=3e-7,rtol=0)
    p=p/p.sum(1,keepdims=True)
    return .5*np.square(np.cumsum(p,axis=1)[:,:2]-(y[:,None]<=np.arange(2))).sum(1)

def add_rps(table):
    for view,columns in [('raw',P_COLUMNS),('calibrated',C_COLUMNS)]:
        if set(columns)<=set(table):table[f'{view}_rps']=rps(table[columns],table.y)
    return table

def scores(table,view='raw'):
    columns=P_COLUMNS if view=='raw' else C_COLUMNS
    result=base.outcome_metrics(table[columns].to_numpy(),table.y)
    result['rps']=float(rps(table[columns],table.y).mean())
    return result

def table_scores(table):
    result=base.table_metrics(table)
    for view,columns in [('raw',P_COLUMNS),('calibrated',C_COLUMNS)]:result[view]['rps']=float(rps(table[columns],table.y).mean())
    return result

def summarize(losses,leagues):
    losses=np.asarray(losses,float);leagues=np.asarray(leagues)
    if losses.shape!=leagues.shape or not len(losses) or not np.isfinite(losses).all():raise ValueError('Invalid validation population')
    per_league={str(k):float(losses[leagues==k].mean()) for k in np.unique(leagues)}
    return dict(micro=float(losses.mean()),macro=float(np.mean(list(per_league.values()))),per_league=per_league,count=len(losses))
