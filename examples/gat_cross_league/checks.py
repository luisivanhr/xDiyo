"""Focused executable checks of the new population and exact-computation risks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from . import core,bootstrap
from .components import ComponentStore
from .initialization import initial_path,feature_names,raw_predictions
from compare_gat_temporal import digest,write_json
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.streaming import GroupPlan,prepare_group
from xdiyo_analytics.graphs.network import TemporalNetwork
from temporal_team_gat import make_context


def component_parity(full,restricted,frame,config,pre,state,device):
    plans=[GroupPlan(s,config,make_context(frame),s.issue_times(frame,config)) for s in (full,restricted)]
    assert plans[0].descriptors[0][0]==plans[1].descriptors[0][0]
    raw=[p.raw(0) for p in plans]
    np.testing.assert_array_equal(raw[0][5],raw[1][5]) # rest, support, neutral and masks upstream
    full_snap,small_snap=raw[0][1],raw[1][1]
    select=np.array([full_snap.teams.index(t) for t in small_snap.teams])
    np.testing.assert_array_equal(full_snap.support[select],small_snap.support)
    np.testing.assert_array_equal(full_snap.gate[select],small_snap.gate)
    positions=full_snap.edges.index.get_indexer(small_snap.edges.index)
    np.testing.assert_array_equal(full_snap.log_weight[positions],small_snap.log_weight)
    values=[];gradients=[];data_x=[]
    for store,group in zip((full,restricted),raw):
        prep=GraphPreprocessor.from_state(config,pre.state())
        _,data,home,away,fixture=prepare_group(group,prep,device,store)
        model=TemporalNetwork(config,len(pre.leagues),pre.edge_dim,len(feature_names(pre)['fixture'])).to(device)
        model.load_state_dict(state)
        logits,details=model(data,home,away,fixture,return_goal_details=True)
        objective=logits.square().mean()+details['rates'].square().mean()+config.prior_penalty*(model.league.weight.square().sum()+model.entry.weight.square().sum())
        objective.backward()
        values.append((logits.detach().cpu(),details['rates'].detach().cpu()))
        gradients.append({n:p.grad.detach().cpu() if p.grad is not None else None for n,p in model.named_parameters()})
        data_x.append(data['x'].detach().cpu())
    torch.testing.assert_close(data_x[0][positions],data_x[1],atol=0,rtol=0) # includes all temporal gap coordinates
    worst=0.
    for a,b in zip(*values):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        worst=max(worst,float((a-b).abs().max()))
    for name,a in gradients[0].items():
        b=gradients[1][name]
        assert (a is None)==(b is None),name
        if a is not None:
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    return dict(device=device,full_edges=len(full_snap.edges),retained_edges=len(small_snap.edges),
                full_nodes=len(full_snap.teams),retained_nodes=len(small_snap.teams),max_output_error=worst,
                all_parameter_gradients_checked=True,rest_support_gaps_exact=True)


def check(run):
    p,full=core.load(run,components=False)
    restricted=ComponentStore(full.history,full.team_seasons,schedule=full.schedule,provenance=full.provenance)
    torch.set_num_threads(1)
    from xdiyo_analytics.graphs.execution import prepare_determinism
    prepare_determinism(core.config(),'cuda:0')
    protected=json.loads((run/'shared/checks/protected_artifacts.json').read_text())
    for path,expected in protected.items():assert digest(core.ROOT/path)==expected,path
    population_checks=[];coverage=[];pairing=[]
    for fold in core.FOLDS:
        previous=None
        for phase,parent_phase in [('fit','expanded'),('inner','inner'),('outer','outer')]:
            local=core.populations(run,fold,'local')[phase]
            pooled=core.populations(run,fold)[phase]
            assert digest(run/'shared/populations'/f'{fold}_local_{phase}.parquet')==digest(core.PARENT/'shared/data'/f'{fold}_{parent_phase}.parquet')
            assert core.keys(local).isin(core.keys(pooled)).all() and not core.keys(pooled).duplicated().any()
            issue=full.issue_times(pooled,core.config())
            available=full.label_availability(pooled,core.config())
            cutoff=min(core.config().history_cutoff(t) for t in issue)
            assert previous is None or previous<=cutoff
            previous=available.max()
            full.assert_targets_unobserved(pooled,issue,core.config())
            subset=pooled.iloc[::17]
            pd.testing.assert_series_equal(full.issue_times(subset,core.config()),issue.iloc[::17])
            population_checks.append(dict(fold=fold,phase=phase,local=len(local),pooled=len(pooled),
                league_count=int(pooled.competition_id.nunique()),duplicate_fixture_keys=0,round_anchors_preserved=True))
            plan=GroupPlan(restricted,core.config(),make_context(pooled),issue)
            numeric,_=plan.fitting_inputs()
            for league,indices in pooled.groupby('competition_id').indices.items():
                row=dict(fold=fold,phase=phase,competition_id=int(league),count=len(indices))
                for name,column in zip(('neutral_venue','home_rest_days','away_rest_days','home_support','away_support'),numeric.T):
                    row[name+'_missing']=int(np.isnan(column[indices]).sum())
                coverage.append(row)
        local=core.populations(run,fold,'local')['fit'];pooled=core.populations(run,fold)['fit']
        local_issue=full.issue_times(local,core.config());pooled_issue=full.issue_times(pooled,core.config())
        common=[local_issue.min(),local_issue.max()]
        local_probe=local[local_issue.isin(common)].reset_index(drop=True)
        pooled_probe=pooled[pooled_issue.isin(common)].reset_index(drop=True)
        indices=core.keys(pooled_probe).get_indexer(core.keys(local_probe))
        for seed in core.SEEDS:
            for name in ('snapshot_B','time_C'):
                cfg=core.config(name)
                pre=GraphPreprocessor.from_state(cfg,json.loads((run/'shared/preprocessing'/f'{fold}.json').read_text())['state'])
                state=torch.load(initial_path(run,dict(configuration=name,fold=fold,seed=seed)),weights_only=True)
                one=raw_predictions(restricted,local_probe,cfg,pre,state)
                two=raw_predictions(restricted,pooled_probe,cfg,pre,state)
                for a,b in zip(one,two):np.testing.assert_allclose(a,b[indices],atol=2e-7,rtol=2e-6)
                pairing.append(dict(fold=fold,seed=seed,configuration=name,common_fixtures=len(indices),
                    max_error=max(float(np.max(np.abs(a-b[indices]))) for a,b in zip(one,two))))
    pd.DataFrame(coverage).to_csv(run/'shared/checks/fixture_coverage.csv',index=False)
    write_json(run/'shared/checks/local_pooled_initial_pairing.json',pairing)
    inventory=pd.read_csv(run/'league_inventory.csv')
    source_versions=json.loads((run/'shared/checks/source_versions.json').read_text())
    source_versions+=p['provenance']['parent']['sources']
    inventory['source_versions']=inventory.league.map(lambda league:'; '.join(
        f'{s["season"]}:{s["version"]}' for s in source_versions if s['season'].rsplit('_',2)[0]==league))
    hist=full.history
    inventory['history_first']=inventory.competition_id.map(hist.groupby('competition_id').kickoff_at.min().astype(str))
    inventory['history_last']=inventory.competition_id.map(hist.groupby('competition_id').kickoff_at.max().astype(str))
    inventory.to_csv(run/'league_inventory.csv',index=False)
    receipts=[]
    # Real-data complete-component checks cover single-country and mixed groups,
    # fitting and both later replay phases, plus CPU and CUDA paths.
    for name in ('snapshot_B','time_C'):
        cfg=core.config(name)
        pre=GraphPreprocessor.from_state(cfg,json.loads((run/'shared/preprocessing/fold_2.json').read_text())['state'])
        state=torch.load(initial_path(run,dict(configuration=name,fold='fold_2',seed=17)),weights_only=True)
        if name=='time_C':
            for key,value in state.items():
                if key.endswith(('.e.weight','.u.weight')):
                    value[:,-14:]=torch.sin(torch.arange(value.shape[0]*14).reshape(value.shape[0],14))*.01
        for phase in ('fit','inner','outer'):
            frame=core.populations(run,'fold_2')[phase]
            issue=full.issue_times(frame,cfg)
            for when in (issue.min(),issue.max()):
                probe=frame[issue.eq(when)].reset_index(drop=True)
                for device in ('cpu','cuda:0'):
                    result=component_parity(full,restricted,probe,cfg,pre,state,device)
                    result.update(configuration=name,phase=phase,issuance=str(when));receipts.append(result)
    write_json(run/'shared/checks/component_parity.json',receipts)
    resampling=bootstrap.prepare(run)
    receipt=dict(status='passed',population_checks=population_checks,component_cases=len(receipts),
        max_component_output_error=max(x['max_output_error'] for x in receipts),protected_files=len(protected),
        bootstrap_rejections=sum(x['rejected'] for x in resampling),
        core_tests='See focused_tests.txt for synthetic gradient, routing, availability, initialization and aggregation tests')
    write_json(run/'shared/checks/correctness.json',receipt)
    # Enable only after the full-context equality checks have succeeded.
    p['resources']['complete_components']=True
    p['resources']['component_policy']='Complete static disconnected components containing target teams; global registry and regularizer unchanged'
    write_json(run/'PROTOCOL.json',p)
    core.status(run,stage='component_equivalence_verified')
    print(json.dumps(receipt),flush=True)
