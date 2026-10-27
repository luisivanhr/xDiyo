"""Fold-local inner-only configuration choice and a durable outer-scoring gate."""
from pathlib import Path
import json
import numpy as np
from . import core

def choose(records,fold):
    selected=[]
    for name in core.CONFIGS:
        rows=[r for r in records if r['fold']==fold and r['configuration']==name]
        if sorted(r['seed'] for r in rows)!=list(core.SEEDS):raise ValueError('Each configuration needs all three matched inner scores')
        rows=sorted(rows,key=lambda r:r['seed']);values=[float(r['inner_micro_ce']) for r in rows]
        if not np.isfinite(values).all():raise ValueError('Nonfinite inner score')
        selected.append(dict(configuration=name,seeds=list(core.SEEDS),inner_scores=values,mean_inner_micro_ce=float(np.mean(values))))
    winner=min(selected,key=lambda r:(r['mean_inner_micro_ce'],core.CONFIGS.index(r['configuration'])))
    return dict(fold=fold,configuration=winner['configuration'],configurations=selected,tie_order=list(core.CONFIGS))

def seal(run):
    run=Path(run);protocol,_=core.load(run,frozen=True);records=[];identities={}
    for job in protocol['matrix']:
        folder=run/job['path'];result=json.loads((folder/'fit.json').read_text())
        assert result['identity']['monitor']==core.MONITOR
        assert result['summary']['monitor']==core.MONITOR
        records.append(dict(fold=job['fold'],configuration=job['configuration'],seed=job['seed'],inner_micro_ce=result['selection_score']))
        for name in ('fit.json','inner_raw.parquet','model/weights.pt'):
            identities[(folder/name).relative_to(run).as_posix()]=core.digest(folder/name)
    decisions=[choose(records,fold) for fold in core.FOLDS]
    path=run/'SELECTION.json'
    value=dict(protocol_sha256=core.digest(run/'PROTOCOL.json'),monitor=core.MONITOR,fold_choices=decisions,
        source_hashes=identities,latest_fold_recommendation=decisions[-1]['configuration'],
        rule='Own-fold selected raw inner micro CE averaged over all three seeds; exact ties use declared order')
    if path.exists():
        previous=json.loads(path.read_text())
        assert {k:v for k,v in previous.items() if k!='sealed_at'}==value,'Existing choices disagree; do not replace them after outer scoring'
        return previous
    assert not list(run.glob('0*/**/outer_predictions.parquet')),'New outer GAT scores exist before configuration selection'
    value['sealed_at']=__import__('pandas').Timestamp.now(tz='UTC').isoformat()
    core.write_json(path,value);return value

def verify_gate(run):
    run=Path(run);value=json.loads((run/'SELECTION.json').read_text())
    assert value['protocol_sha256']==core.digest(run/'PROTOCOL.json') and value['monitor']==core.MONITOR
    for relative,expected in value['source_hashes'].items():assert core.digest(run/relative)==expected,relative
    return value
