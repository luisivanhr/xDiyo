from dataclasses import asdict
from pathlib import Path
import copy,json,sys
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from test_gat_cross_league import two_leagues,gradients
from gat_all_league import core,metrics,stages,selection,execution
from gat_all_league.backend import AllLeagueBackend
from temporal_team_gat import make_context
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.training import TrainingControl,EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.training.native_resume import capture_backend
from compare_gat_goals import digest_state


def toy_run(tmp_path,device='cpu'):
    old=two_leagues();schedule=stages.repair_schedule(old.schedule)
    history=stages.attach_schedule(old.history,schedule)
    store=core.ComponentStore(history,old.team_seasons,schedule=schedule)
    fitting=schedule[schedule.event_id.isin([2,3])].reset_index(drop=True)
    inner=schedule[((schedule.competition_id==10)&schedule.event_id.isin([6,7,8]))|
                   ((schedule.competition_id==20)&schedule.event_id.eq(6))].reset_index(drop=True)
    run=tmp_path/'tiny';(run/'shared/preprocessing').mkdir(parents=True)
    prototype=_Backend(core.config('time_C'),store,17,'cpu');prototype.initialize(make_context(fitting))
    core.write_json(run/'shared/preprocessing/fold_1.json',dict(state=prototype.preprocessor.state()))
    job=dict(job_id='test',configuration='time_C',fold='fold_1',seed=17,arm='pooled',path='01_pooled/test')
    initial=core.initial_path(run,job);initial.parent.mkdir(parents=True);torch.save(prototype.model.state_dict(),initial)
    protocol=dict(configurations={'time_C':asdict(core.config('time_C'))},competitions=[10,20],
        resources=dict(device=device,input_cache_bytes=32*2**20,headroom_bytes=2**30))
    context=make_context(fitting);context.validation=make_context(inner)
    backend=AllLeagueBackend(run,protocol,store,job);backend.initialize(context)
    return backend,context,run,protocol,store,job


def test_repeated_round_stages_and_non_belgian_anchors():
    frame=pd.DataFrame(dict(competition_id=[38,38,38,38,36,36],season_id=[1]*6,event_id=range(1,7),
        home_id=[1,3,1,3,10,12],away_id=[2,4,2,4,11,13],tournament_id=[38,38,12656,12656,92066,92067],round=[1,1,1,1,34,34],
        kickoff_at=pd.to_datetime(['2024-01-02T10:00Z','2024-01-03T10:00Z','2024-04-02T10:00Z','2024-04-04T10:00Z','2024-04-01T10:00Z','2024-04-02T10:00Z'])))
    frame['round_issue_at']=frame.groupby(['competition_id','season_id','round']).kickoff_at.transform('min')
    fixed=stages.repair_schedule(frame)
    assert fixed.round_stage.tolist()==[38,38,12656,12656,0,0]
    assert fixed.round_issue_at.iloc[2]==pd.Timestamp('2024-04-02T10:00Z')
    assert fixed.round_issue_at.iloc[2]!=frame.round_issue_at.iloc[2]
    filtered=stages.attach_schedule(frame.iloc[[3]],fixed)
    assert filtered.round_issue_at.iloc[0]==fixed.round_issue_at.iloc[2]
    np.testing.assert_array_equal(frame.round_issue_at.iloc[-2:],fixed.round_issue_at.iloc[-2:])
    with pytest.raises(ValueError,match='Unreviewed'):
        stages.repair_schedule(frame.assign(tournament_id=[38,38,999,12656,92066,92067]))


def test_rps_order_normalization_and_limits():
    np.testing.assert_array_equal(metrics.rps(np.eye(3),[0,1,2]),[0.,0.,0.])
    np.testing.assert_array_equal(metrics.rps(np.eye(3),[2,0,0]),[1.,.5,1.])
    np.testing.assert_allclose(metrics.rps(np.full((3,3),1/3),[0,1,2]),[5/18,1/9,5/18],atol=1e-15)
    p=np.array([[.2,.3,.5],[.7,.2,.1]])
    expected=np.array([.5*(.2**2+.5**2),.5*((.7-1)**2+(.9-1)**2)])
    np.testing.assert_allclose(metrics.rps(p,[2,0]),expected)
    np.testing.assert_allclose(metrics.rps(p,[2,0]),metrics.rps(p[:,::-1],[0,2]))


def test_fixture_weighted_monitor_is_not_league_average():
    result=metrics.summarize([1.,2.,3.,8.],[10,10,10,20])
    assert result['micro']==3.5 and result['macro']==5. and result['per_league']=={'10':2.,'20':8.}

def test_fixed_calendar_resample_denominators_and_original_mixture():
    from gat_all_league.analysis import stratum_samples,combine_strata,contrasts
    loss=np.array([1.,3.,10.,20.,30.]);leagues=np.array([10,10,20,20,20]);week=np.array([0,1,0,1,1])
    weights=np.array([[1.,1.],[2.,1.],[1.,2.]])
    point,sample,count=stratum_samples(loss,leagues,week,weights,[10,20])
    np.testing.assert_array_equal(point,[2.,20.]);np.testing.assert_array_equal(count,[2,3])
    for index,w in enumerate(weights):
        for column,league in enumerate([10,20]):
            take=leagues==league
            np.testing.assert_allclose(sample[index,column],np.average(loss[take],weights=w[week[take]]))
    other=np.array([4.,8.]);second=np.tile(other,(3,1));counts=np.array([[2,3],[4,1]])
    combined,draws,total=combine_strata([point,other],[sample,second],counts)
    np.testing.assert_allclose(combined,[(2*2+4*4)/6,(20*3+8)/4])
    np.testing.assert_allclose(combined@(total/total.sum()),(2*2+20*3+4*4+8)/10)
    np.testing.assert_allclose(draws[0],combined)
    assert len(contrasts())==11

def test_all13_scoring_averages_seed_losses_not_probabilities():
    from gat_all_league.analysis import MODELS,score_models
    models={}
    probabilities=[np.array([.6,.2,.2]),np.array([.2,.5,.3]),np.array([.3,.2,.5])]
    for fold in core.FOLDS:
        for name in MODELS:
            frames=[]
            for seed in range(3 if name in (*core.CONFIGS,'validation_selected') else 1):
                frame=pd.DataFrame({'competition_id':np.repeat(np.arange(13),3),'y':np.tile([0,1,2],13)})
                frame[metrics.P_COLUMNS]=np.tile(probabilities[seed],(len(frame),1))
                frame[metrics.C_COLUMNS]=frame[metrics.P_COLUMNS].to_numpy();frames.append(frame)
            models[fold,name]=frames
    scored=score_models(models,'outer')
    part=scored[(scored.model=='time_C')&(scored.fold=='combined')&(scored.view=='raw')&(scored.aggregation=='micro')]
    expected=np.mean([-np.log(prob).mean() for prob in probabilities])
    np.testing.assert_allclose(part.log_loss.mean(),expected,atol=1e-14)
    assert not np.isclose(expected,-np.log(np.mean(probabilities,axis=0)).mean())
    np.testing.assert_allclose(part.log_loss,part.draw_log_loss+part.winner_contribution,atol=1e-14)

def test_fresh_cuda_all_inner_fit_export_and_selection_gate(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import subprocess,os
    backend,context,run,p,store,job=toy_run(tmp_path,'cpu')
    p.update(provenance=store.provenance,store_version=store.version,matrix=[job])
    p['resources'].update(device='cuda:0',threads=1,rss_limit_gib=8.,cuda_fraction=.8)
    (run/'shared/data').mkdir();(run/'shared/populations').mkdir()
    for name,frame in [('history',store.history),('memberships',store.team_seasons),('schedule',store.schedule)]:
        frame.to_parquet(run/'shared/data'/f'{name}.parquet',index=False)
    for phase,frame in [('fit',context.metadata),('inner',context.validation.metadata),('outer',store.schedule[store.schedule.event_id.eq(9)])]:
        frame.to_parquet(run/'shared/populations'/f'fold_1_{phase}.parquet',index=False)
    core.write_json(run/'PROTOCOL.json',p)
    core.write_json(run/'FREEZE.json',dict(protocol_sha256=core.digest(run/'PROTOCOL.json'),sources={},inputs={}))
    env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(core.ROOT/'examples/run_gat_all_league.py'),'fit','--run',str(run),'--job',job['job_id']],
        capture_output=True,text=True,env=env,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    folder=run/job['path'];receipt=json.loads((folder/'fit.json').read_text())
    assert receipt['reload_exact'] and receipt['public_boundaries_verified'] and not receipt['outer_scored']
    assert not (folder/'outer_predictions.parquet').exists() and not (folder/'calibration.json').exists()
    inner=pd.read_parquet(folder/'inner_raw.parquet')
    np.testing.assert_allclose(receipt['selection_score'],metrics.scores(inner)['log_loss'],rtol=0,atol=1e-14)
    assert len(inner)==4 and set(inner.competition_id)=={10,20}
    curve=pd.read_csv(folder/'training.csv',float_precision='round_trip');execution.verify_progress(curve,receipt['summary'])
    state=torch.load(folder/'attempt_01/last_training.pt',weights_only=True)
    assert state['active']['step']==len(curve) and state['active']['best_step']==receipt['summary']['best_step']
    before=core.digest(folder/'model/weights.pt')
    with pytest.raises(FileNotFoundError):execution.evaluate(run)
    assert before==core.digest(folder/'model/weights.pt')


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_actual_monitor_single_pass_and_foreign_label_gradient_isolation(tmp_path,device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    backend,context,*_=toy_run(tmp_path,device)
    before=digest_state(capture_backend(backend))
    backend.model.eval()
    backend._loss(backend.validation_groups,backend.validation_targets,backend.validation_weights)
    p=backend._inner_probability.copy();y=backend.validation_targets.cpu().numpy().copy()
    expected=-np.log(p[np.arange(len(y)),y]).mean()
    np.testing.assert_allclose(backend._inner_summary['micro'],expected,rtol=1e-6,atol=1e-7)
    assert backend._inner_summary['count']==4 and len(backend._inner_summary['per_league'])==2
    backend.optimizer.zero_grad(set_to_none=True)
    backend._loss(backend.train_groups,backend.train_targets,backend.train_weights,True)
    original_gradients=gradients(backend.model);original_parameters={k:v.clone() for k,v in backend.model.state_dict().items()}
    foreign=np.flatnonzero(backend._validation_leagues==20)
    old_monitor=backend._inner_summary['micro']
    backend.validation_targets[foreign]=(backend.validation_targets[foreign]+1)%3
    backend._loss(backend.validation_groups,backend.validation_targets,backend.validation_weights)
    assert abs(backend._inner_summary['micro']-old_monitor)>1e-6
    for name,value in backend.model.state_dict().items():torch.testing.assert_close(value,original_parameters[name],rtol=0,atol=0)
    for name,value in gradients(backend.model).items():
        if value is not None:torch.testing.assert_close(value,original_gradients[name],rtol=0,atol=0)
    backend.optimizer.zero_grad(set_to_none=True);backend._loss(backend.train_groups,backend.train_targets,backend.train_weights,True)
    for name,value in gradients(backend.model).items():
        if value is not None:torch.testing.assert_close(value,original_gradients[name],rtol=0,atol=0)
    backend.validation_targets.copy_(torch.as_tensor(y,device=backend.device))
    calls=backend._validation_forward_calls
    values=backend.checkpoint_metrics(core.MONITOR,backend.step())
    assert backend._validation_forward_calls-calls==len(backend.validation_groups)
    assert values[core.MONITOR]==values['validation_loss']==backend._inner_summary['micro']
    assert values['all13_inner_raw_hda_ce_macro']==backend._inner_summary['macro']
    assert len(backend.train_targets)==len(context.X)


def test_configuration_choice_is_fold_local_and_inner_only():
    records=[dict(fold=f,configuration=c,seed=s,inner_micro_ce=1+core.CONFIGS.index(c)*.01,outer_ce=-core.CONFIGS.index(c)*100.)
        for f in core.FOLDS for c in core.CONFIGS for s in core.SEEDS]
    expected=selection.choose(records,'fold_1')
    changed=copy.deepcopy(records)
    for item in changed:
        item['outer_ce']=-9999
        if item['fold']=='fold_2':item['inner_micro_ce']=10-core.CONFIGS.index(item['configuration'])
    assert selection.choose(changed,'fold_1')==expected
    assert selection.choose(changed,'fold_2')['configuration']=='time_C'
    for item in changed:item['inner_micro_ce']=1.
    assert selection.choose(changed,'fold_1')['configuration']=='snapshot_B'
    with pytest.raises(ValueError,match='three'):selection.choose(records[:-1],'fold_2')


def test_resume_rejects_old_monitor_and_population_identity(tmp_path):
    folder=tmp_path/'job';attempt=folder/'attempt_01';attempt.mkdir(parents=True)
    path=attempt/'last_training.pt';torch.save(dict(example=True),path)
    identity=dict(monitor=core.MONITOR,population='corrected')
    for changed in [dict(monitor='validation_loss',population='corrected'),dict(monitor=core.MONITOR,population='old')]:
        core.write_json(attempt/'last_training.json',dict(identity=changed,sha256=core.digest(path)))
        with pytest.raises(ValueError,match='identity mismatch'):execution.continuation(folder,identity)


def test_earliest_best_and_patience_ignore_auxiliary_and_bundesliga_metrics(tmp_path):
    backend,context,run,protocol,store,job=toy_run(tmp_path)
    class Controlled(AllLeagueBackend):
        def initialize(self,ctx):super().initialize(ctx);self.iteration=0
        def step(self):
            self.iteration+=1
            with torch.no_grad():next(self.model.parameters()).fill_(self.iteration)
            return {core.MONITOR:[1.,.8,.8,.9][self.iteration-1], 'bundesliga_ce':1/self.iteration,
                    'goal_loss':10/self.iteration,'regularization':1/self.iteration}
        def checkpoint_metrics(self,monitor,values):return values
    states=[]
    selected,history,summary=run_resumable_iterations(lambda seed:Controlled(run,protocol,store,job),context,
        TrainingControl(max_steps=10,seed=17,monitor=core.MONITOR,early_stopping=EarlyStopping(2,0),restore_best=True),
        save=lambda capture,complete:states.append((capture(),complete)))
    assert summary['best_step']==2 and summary['attempts'][0]['steps']==4
    assert next(selected.model.parameters()).flatten()[0].item()==2
    assert states[-2][0]['active']['step']==4
    assert next(iter(states[-2][0]['active']['backend']['model'].values())).flatten()[0].item()==4
    assert states[-1][1] is True


def test_declared_matrix_and_grouping_is_not_feature():
    assert len(core.matrix())==18
    for name in core.CONFIGS:
        cfg=core.config(name)
        assert cfg.round_keys==core.ROUND_KEYS and cfg.layers==2 and cfg.embedding_dim==32
        assert 'round_stage' not in cfg.fixture_features and 'round_stage' not in cfg.statistics
        assert cfg.goal_auxiliary.beta==(.1 if name=='time_C' else 0.)
