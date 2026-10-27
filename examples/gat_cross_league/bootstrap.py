"""Prespecified shared calendar-week resamples, fixed before model scoring."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import core
from compare_gat_temporal import write_json,digest


def fixed_mixture(scores,counts):
    """Fold by league scores with original fixture weights; never average seeds' probabilities."""
    scores=np.asarray(scores,dtype=float);counts=np.asarray(counts,dtype=float)
    total=counts.sum(0)
    weights=np.divide(counts,total,out=np.zeros_like(counts),where=total>0)
    if np.any(~np.isfinite(scores[counts>0])):
        raise ValueError('A supported stratum has no score')
    league=(np.where(counts>0,scores,0)*weights).sum(0)
    league=np.where(total>0,league,np.nan)
    macro=float(league.mean()) if (total>0).all() else None
    micro=float(np.nansum(league*total)/total.sum()) if total.sum() else None
    return dict(per_league=league,macro=macro,micro=micro,all_leagues_supported=bool((total>0).all()))


def week_groups(issuance):
    local=pd.DatetimeIndex(issuance).tz_convert('Europe/Berlin')
    dates=local.normalize().tz_localize(None)
    monday=(dates-pd.to_timedelta(dates.weekday,unit='D')).tz_localize('Europe/Berlin')
    weeks=pd.date_range(monday.min(),monday.max(),freq='W-MON')
    return weeks,weeks.get_indexer(monday)


def calendar_samples(counts, length, rng, replicates=20000, cap=1000000):
    """Reject based solely on positive-support schedule strata, never scores."""
    counts=np.asarray(counts);supported=counts.sum(0)>0
    n=len(counts);blocks=int(np.ceil(n/length));accepted=[];drawn=0;rejected=0
    while sum(len(x) for x in accepted)<replicates and drawn<cap:
        size=min(replicates-sum(len(x) for x in accepted),cap-drawn)
        starts=rng.integers(0,n,size=(size,blocks))
        indices=((starts[:,:,None]+np.arange(length))%n).reshape(size,-1)[:,:n]
        weights=np.zeros((size,n),dtype=np.int16)
        np.add.at(weights,(np.arange(size)[:,None],indices),1)
        valid=((weights.astype(np.int64)@counts[:,supported])>0).all(1)
        accepted.append(weights[valid]);drawn+=size;rejected+=int((~valid).sum())
    all_weights=np.concatenate(accepted) if accepted else np.empty((0,n),dtype=np.int16)
    return (all_weights[:replicates] if len(all_weights)>=replicates else None),dict(
        drawn=drawn,rejected=rejected,rejection_rate=rejected/drawn if drawn else None,
        supported_strata=int(supported.sum()),unavailable_strata=np.flatnonzero(~supported).tolist(),
        requested=replicates,cap=cap,interval_available=len(all_weights)>=replicates)


def prepare(run):
    run=Path(run);p,store=core.load(run);rng=np.random.default_rng(20261027);receipts=[]
    for fold in core.FOLDS:
        frame=core.populations(run,fold)['outer']
        issue=store.issue_times(frame,core.config());weeks,positions=week_groups(issue)
        counts=np.column_stack([np.bincount(positions,weights=frame.competition_id.eq(league),minlength=len(weeks))
                                for league in p['competitions']]).astype(np.int64)
        for _,indices in frame.groupby(list(core.config().round_keys)).indices.items():
            assert len(set(positions[indices]))==1,'A full forecast round was split across weeks'
        folder=run/'shared/bootstrap/calendar'/fold;folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'weeks.json',[x.isoformat() for x in weeks])
        np.save(folder/'fixture_week.npy',positions);np.save(folder/'counts_by_league.npy',counts)
        for length in (2,4,8):
            weights,receipt=calendar_samples(counts,length,rng)
            if weights is not None:np.save(folder/f'blocks_{length}.npy',weights)
            receipt.update(fold=fold,block_weeks=length,week_count=len(weeks),competitions=p['competitions'])
            receipts.append(receipt)
    write_json(run/'shared/checks/secondary_bootstrap.json',receipts)
    return receipts
