"""Behavioral tests for temporal reconstruction, neural math and library contracts."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.preprocessing import MaskedScale
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.training.contracts import FitContext, PredictionContext

HAS_TORCH = importlib.util.find_spec("torch") is not None
neural = pytest.mark.skipif(not HAS_TORCH, reason="optional torch")


def history(events):
    rows = []
    for event, date, home, away, gf, ga, comp, season, round_ in events:
        for team, opponent, side, a, b in [(home, away, "home", gf, ga), (away, home, "away", ga, gf)]:
            rows.append(dict(event_id=event, kickoff_at=pd.Timestamp(date, tz="UTC"),
                team_id=team, opponent_id=opponent, side=side, goals_for=a, goals_against=b,
                competition_id=comp, season_id=season, round=round_, status="finished"))
    return pd.DataFrame(rows)


def sample():
    h = history([(i+1, f"2020-01-{1+3*i:02}", 1+i%2, 2-i%2,
                  2 if i%3 == 0 else 0, 0 if i%3 == 0 else i%3, 10, 900, i+1) for i in range(9)])
    m = pd.DataFrame([dict(team_id=t, competition_id=10, season_id=900,
                          entry_at=pd.Timestamp("2019-07-01", tz="UTC"), movement="retained") for t in (1,2,3)])
    return TemporalEventStore(h, m)


def config(**kwargs):
    return TemporalGraphConfig("Europe/Berlin", embedding_dim=4, hidden_width=8, **kwargs)


def context(store, positions, *, fit=True):
    meta = store.schedule.iloc[positions].copy()
    meta.index = pd.Index([101 + 7*i for i in positions], name="row_position")
    X = pd.DataFrame({"unused": np.zeros(len(meta))}, index=meta.index)
    common = dict(X=X, metadata=meta, layout="match", match_columns=("competition_id","season_id","event_id"), fold_id=0)
    if fit:
        y = pd.DataFrame({"result": np.where(meta.goals_for > meta.goals_against, 1,
                                np.where(meta.goals_for < meta.goals_against, -1, 0))}, index=meta.index)
        return FitContext(**common, y=y)
    return PredictionContext(**common)


def test_cutoff_dst_and_previous_local_day():
    c = config()
    assert c.history_cutoff(pd.Timestamp("2020-03-29T16:00Z")) == pd.Timestamp("2020-03-28T23:00Z")
    assert c.labels_known_at(pd.Series(pd.to_datetime(["2020-03-29T16:00Z"]))).iloc[0] == pd.Timestamp("2020-03-29T22:00Z")
    with pytest.raises(ValueError):
        c.history_cutoff("2020-01-01")


def test_round_identity_frozen_postponement_and_subset():
    s = sample()
    schedule = s.schedule
    schedule.loc[1, "round"] = 1
    schedule.loc[1, "kickoff_at"] = pd.Timestamp("2020-06-01", tz="UTC")
    # Same round number in another competition-season has its own anchor.
    extra = schedule.iloc[[2]].copy()
    extra["competition_id"], extra["season_id"], extra["round"] = 11, 12, 1
    schedule = pd.concat([schedule, extra], ignore_index=True)
    s = TemporalEventStore(s.history, s.team_seasons, schedule=schedule)
    issue = s.issue_times(schedule, config())
    assert issue.iloc[0] == issue.iloc[1] == pd.Timestamp("2020-01-01", tz="UTC")
    assert issue.iloc[-1] != issue.iloc[0]
    assert s.issue_times(schedule.iloc[[1]], config()).iloc[0] == issue.iloc[0]


def test_one_day_ahead_and_explicit_issue():
    s = sample()
    c = config(forecast_policy="one_day_ahead")
    f = s.schedule.iloc[[2]].copy()
    assert s.issue_times(f, c).iloc[0] == f.kickoff_at.iloc[0]-pd.Timedelta(hours=24)
    f["issuance_at"] = pd.Timestamp("2020-01-04", tz="UTC")
    assert s.issue_times(f,c).iloc[0] == f.issuance_at.iloc[0]


def test_duplicates_pair_consistency_repeated_matches_large_ids():
    s = sample()
    assert len(s.snapshot(pd.Timestamp("2020-03-01",tz="UTC"), config()).edges) == 18
    with pytest.raises(ValueError, match="Duplicate"):
        TemporalEventStore(pd.concat([s.history, s.history.iloc[[0]]]), s.team_seasons)
    bad = s.history
    bad.loc[0,"goals_for"] = 99
    with pytest.raises(ValueError, match="Inconsistent"):
        TemporalEventStore(bad, s.team_seasons)
    h, m = s.history, s.team_seasons
    huge = 2**63+19
    for col in ("team_id","opponent_id","event_id"):
        h[col] = pd.array([huge+int(x) for x in h[col]],dtype="UInt64")
    m.team_id = pd.array([huge+int(x) for x in m.team_id],dtype="UInt64")
    exact = TemporalEventStore(h,m)
    assert huge+1 in exact.snapshot(pd.Timestamp("2020-03-01",tz="UTC"),config()).teams
    h["team_id"] = h.team_id.astype(float)
    with pytest.raises(ValueError, match="exact"):
        TemporalEventStore(h,m)


def test_missing_zero_and_statistics_validation():
    scale = MaskedScale().fit(np.array([[0.],[1.],[np.nan]]))
    out = scale.transform(np.array([[0.],[np.nan]]))
    assert out[0,-1] == 0 and out[1,-1] == 1
    assert out[1,0] == 0
    with pytest.raises(ValueError, match="xG"):
        config(statistics=("team::ALL::Expected goals::xg::value",))


def test_season_discount_movement_context_preserves_history():
    s = sample()
    m = s.team_seasons
    newer = pd.DataFrame([dict(team_id=t, competition_id=11 if t==1 else 10, season_id=2,
                              entry_at=pd.Timestamp("2020-02-01",tz="UTC"),
                              movement="promoted" if t==1 else "retained") for t in (1,2)])
    moved = TemporalEventStore(s.history,pd.concat([m,newer]))
    time = pd.Timestamp("2020-02-02",tz="UTC")
    a,b = s.snapshot(time,config()),moved.snapshot(time,config())
    np.testing.assert_allclose(b.support, a.support*.5)
    assert b.league == (11,10) and b.entry == ("promoted","retained")
    assert len(a.edges)==len(b.edges)


@neural
def test_attention_numerical_reference_old_edges_and_empty():
    import torch
    from xdiyo_analytics.graphs.network import grouped_softmax
    scores = torch.tensor([[.2,.4],[-.5,.1],[.3,.8]],dtype=torch.float64)
    receiver = torch.tensor([0,0,2])
    logw = torch.tensor([-10000.,-10001.,-20000.],dtype=torch.float64)
    actual = grouped_softmax(scores,receiver,logw,4)
    expected = torch.cat((torch.softmax(scores[:2]+logw[:2,None],dim=0),torch.ones(1,2)),dim=0)
    torch.testing.assert_close(actual,expected)
    assert grouped_softmax(scores[:0],receiver[:0],logw[:0],4).shape==(0,2)


@neural
def test_values_two_hops_and_no_three_hop_influence():
    import torch
    from xdiyo_analytics.graphs.network import EdgeLayer
    # Directed chain 3 -> 2 -> 1 -> 0. A value change on 3->2 reaches 0 only after 3 hops.
    torch.manual_seed(2)
    layers = [EdgeLayer(2,1,1) for _ in range(3)]
    for layer in layers:
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.fill_(.3)
    receiver, sender = torch.tensor([0,1,2]),torch.tensor([1,2,3])
    h = torch.zeros(4,2)
    x = torch.zeros(3,1)
    altered = x.clone(); altered[2]=1
    def run(features, n):
        state = h
        for layer in layers[:n]:
            state = layer(state,features,receiver,sender,torch.zeros(3),torch.ones(4))
        return state
    a,b=run(x,1),run(altered,1)
    assert not torch.equal(a[2],b[2]) # identical priors, different edge outcomes
    assert torch.equal(a[1],b[1])
    a,b=run(x,2),run(altered,2)
    assert not torch.equal(a[1],b[1])
    assert torch.equal(a[0],b[0])
    assert not torch.equal(run(x,3)[0],run(altered,3)[0])


@neural
def test_fit_gradients_validation_isolation_row_order_restarts_and_save(tmp_path):
    s=sample()
    train=context(s,[1,2,3])
    train.validation=context(s,[5])
    model=TemporalGraphAdapter(config(),s)
    model.fit_controlled(train,TrainingControl(max_steps=3,restarts=1,seed=23))
    b=model.backend_
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in b.model.layers.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in b.model.head.parameters())
    # union of eligible fitting histories ends before Jan 10.
    assert b.preprocessor.edge_scale.mean[0] == pytest.approx(s.history[s.history.kickoff_at < pd.Timestamp("2020-01-09T23:00Z")].goals_for.mean())
    pred=context(s,[8,7],fit=False)
    result=model.predict(pred)["predict_proba"]
    assert result.index.equals(pred.X.index)
    np.testing.assert_allclose(result.sum(axis=1),1,atol=1e-6)
    before=deep_state(b.model)
    model.predict(pred)
    assert before==deep_state(b.model)
    ser=TemporalGraphSerializer(s)
    ser.save(model,tmp_path)
    restored=ser.load(tmp_path)
    pd.testing.assert_frame_equal(result,restored.predict(pred)["predict_proba"])
    queries=pd.DataFrame({"team_id":[1,3,999],"issuance_at":pd.to_datetime(["2020-03-01T12:00Z"]*3)},index=[8,2,9])
    a,z=model.encode_teams(queries),restored.encode_teams(queries)
    pd.testing.assert_frame_equal(a,z)
    assert z.loc[9,"support"]==0 and np.isfinite(z.filter(regex="^z_").to_numpy()).all()
    assert z.loc[2,"league"] == 10 # known cold team contextual prior
    assert z.loc[8,"evidence"]
    with pytest.raises(ValueError,match="exact event_store"):
        TemporalGraphSerializer().load(tmp_path)
    with pytest.raises(ValueError,match="boundary"):
        model.predict(context(s,[1],fit=False))


def deep_state(model):
    return {k:v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items()}


@neural
def test_same_day_future_invariance_and_inactivity():
    s=sample()
    model=TemporalGraphAdapter(config(),s,control=TrainingControl(max_steps=2))
    model.fit(context(s,[1,2,3]))
    cutoff=pd.Timestamp("2020-02-01T12:00Z")
    q=pd.DataFrame({"team_id":[1],"issuance_at":[cutoff]})
    before=model.encode_teams(q)
    extra=history([(100,"2020-02-01T08:00",1,2,99,0,10,900,10),
                   (101,"2020-03-01",1,2,88,0,10,900,11)])
    extended=TemporalEventStore(pd.concat([s.history,extra]),s.team_seasons)
    model.attach_event_store(extended)
    after=model.encode_teams(q)
    np.testing.assert_array_equal(before.filter(regex="^z_"),after.filter(regex="^z_"))
    model.attach_event_store(extended)
    late=q.copy(); late.issuance_at=pd.Timestamp("2050-02-01T12:00Z")
    idle=model.encode_teams(late)
    assert idle.support.iloc[0] < before.support.iloc[0]
    np.testing.assert_allclose(idle.filter(regex="^z_").to_numpy(),
                               idle.filter(regex="^prior_").to_numpy(),atol=1e-8)


@neural
def test_validation_overlap_and_fixture_feature_cutoffs_rejected():
    s=sample(); train=context(s,[1,2,3]); train.validation=context(s,[3])
    with pytest.raises(ValueError,match="validation issuance"):
        TemporalGraphAdapter(config(),s).fit_controlled(train,TrainingControl(max_steps=1))
    train.validation=None
    with pytest.raises(ValueError,match="feature_cutoff"):
        TemporalGraphAdapter(config(fixture_features=("unused",)),s).fit(train)


def test_lazy_optional_dependency():
    script = """import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('blocked for test')
sys.meta_path.insert(0, BlockTorch())
import xdiyo_analytics.graphs
assert 'torch' not in sys.modules
from xdiyo_analytics.graphs.adapter import _torch
try:
    _torch()
except ImportError as e:
    assert '[graphs]' in str(e)
else:
    raise AssertionError('optional import was not blocked')
"""
    subprocess.run([sys.executable,"-c",script],check=True)


@neural
def test_synthetic_learning_weighted_and_fresh_fits():
    s=sample(); c=config(learning_rate=.03,fixture_features=("unused",))
    fit=context(s,[1,2,3,4,5])
    # A declared causal feature deliberately encodes three separable classes.
    fit.X["unused"]=[-1.,0.,1.,-1.,0.]
    fit.y.iloc[:,0]=[-1,0,1,-1,0]
    issue=s.issue_times(fit.metadata,c)
    fit.metadata["feature_cutoff_at"]=[c.history_cutoff(t) for t in issue]
    fit.sample_weight=pd.Series([1.,2.,1.,0.,1.],index=fit.X.index)
    a=TemporalGraphAdapter(c,s)
    a.fit_controlled(fit,TrainingControl(max_steps=60,restore_best=False,seed=4))
    losses=a.training_history_.query("metric=='train_loss'").value
    assert losses.iloc[-1] < losses.iloc[0]*.15
    b=TemporalGraphAdapter(c,s)
    b.fit_controlled(fit,TrainingControl(max_steps=1,seed=4))
    assert a.backend_.model is not b.backend_.model
    assert a.backend_.model_version != b.backend_.model_version


@neural
def test_cpu_device_and_thread_scope():
    import torch
    from xdiyo_analytics.training.execution import configure_model,ExecutionPolicy
    s=sample(); model=TemporalGraphAdapter(config(),s)
    info=configure_model(model,ExecutionPolicy(device="cpu",inner_threads=1))
    old=torch.get_num_threads()
    model.fit_controlled(context(s,[1,2]),TrainingControl(max_steps=1))
    assert torch.get_num_threads()==old and info["device"]=="cpu"
    assert next(model.backend_.model.parameters()).device.type=="cpu"


@neural
def test_cuda_cpu_equivalence_and_gpu_training(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime unavailable")
    s=sample()
    cpu=TemporalGraphAdapter(config(),s)
    cpu.fit_controlled(context(s,[1,2,3]),TrainingControl(max_steps=2))
    TemporalGraphSerializer(s).save(cpu,tmp_path)
    gpu=TemporalGraphSerializer(s,device="cuda:0").load(tmp_path)
    pred=context(s,[7,8],fit=False)
    np.testing.assert_allclose(cpu.predict(pred)["predict_proba"],
                               gpu.predict(pred)["predict_proba"],rtol=2e-5,atol=2e-6)
    gpu.fit_controlled(context(s,[1,2,3]),TrainingControl(max_steps=2))
    assert next(gpu.backend_.model.parameters()).is_cuda
    assert np.isfinite(gpu.predict(pred)["predict_proba"].to_numpy()).all()

@neural
def test_native_runner_fresh_fold_models_and_library_persistence(tmp_path):
    from xdiyo_analytics.datasets import ModelDataset
    from xdiyo_analytics.splits import Fold, SplitPlan
    from xdiyo_analytics.training import TrainingRunner, save_model, load_model
    from xdiyo_analytics.training.execution import ExecutionPolicy
    from xdiyo_analytics.selection import Candidate
    s=sample(); all_rows=context(s,list(range(9)))
    frames=[x.reset_index(drop=True) for x in (all_rows.X,all_rows.y,all_rows.metadata)]
    data=ModelDataset(*frames,"match",all_rows.match_columns,all_rows.match_columns,"home")
    folds=[Fold(np.array([1,2,3]),np.array([5,6]),np.array([5,6])),
           Fold(np.array([1,2,3,4]),np.array([7,8]),np.array([7,8]))]
    candidate=Candidate("graph",lambda:TemporalGraphAdapter(config(),s),config={"layers":2})
    result=TrainingRunner(candidate.model_factory,control=TrainingControl(max_steps=2),
                          execution=ExecutionPolicy(device="cpu")).run(data,SplitPlan(folds,9,np.arange(9)))
    assert result.folds[0].model.backend_.model is not result.folds[1].model.backend_.model
    ser=TemporalGraphSerializer(s)
    path=save_model(result,tmp_path/"native",fold_id=0,serializer=ser)
    loaded=load_model(path,serializer=ser)
    predicted=loaded.predict(data,positions=np.array([5,6]))["predict_proba"]
    pd.testing.assert_frame_equal(predicted,result.folds[0].predictions["predict_proba"])


def test_arrow_missing_scores_and_source_immutability():
    s=sample()
    h=s.history
    h["goals_for"]=pd.array(h.goals_for,dtype="int64[pyarrow]")
    h["goals_against"]=pd.array(h.goals_against,dtype="int64[pyarrow]")
    before=h.copy(deep=True)
    a=TemporalEventStore(h,s.team_seasons)
    pd.testing.assert_frame_equal(h,before)
    copy=a.history; copy["goals_for"]=888
    assert not a.history.goals_for.eq(888).any()


def test_exact_snapshot_provenance_two_hops_and_no_outside_evidence():
    h=history([(1,"2020-01-01",1,2,1,0,10,900,1),
               (2,"2020-01-02",2,3,1,0,10,900,2),
               (3,"2020-01-03",3,4,1,0,10,900,3),
               (4,"2020-01-04",9,10,1,0,10,900,4)])
    m=sample().team_seasons
    s=TemporalEventStore(h,m).snapshot(pd.Timestamp("2020-02-01",tz="UTC"),config())
    assert s.evidence(1,1)==((10,900,1),)
    assert s.evidence(1,2)==((10,900,1),(10,900,2))


@neural
def test_validation_never_optimizes_or_fits_preprocessing():
    s=sample(); a=context(s,[1,2,3]); a.validation=context(s,[5,6])
    b=replace(a,validation=replace(a.validation,y=a.validation.y.copy()))
    b.validation.y.iloc[:,0]=1
    one=TemporalGraphAdapter(config(),s); two=TemporalGraphAdapter(config(),s)
    control=TrainingControl(max_steps=1,restore_best=False,seed=22)
    one.fit_controlled(a,control); two.fit_controlled(b,control)
    assert deep_state(one.backend_.model)==deep_state(two.backend_.model)
    assert one.backend_.preprocessor.state()==two.backend_.preprocessor.state()


@neural
def test_one_day_ahead_training_and_forecast_same_day_invariance():
    s=sample(); c=config(forecast_policy="one_day_ahead")
    a=TemporalGraphAdapter(c,s)
    a.fit_controlled(context(s,[1,2,3]),TrainingControl(max_steps=2))
    pred=context(s,[8],fit=False)
    time=s.issue_times(pred.metadata,c).iloc[0]
    before=a.predict(pred)["predict_proba"]
    extra=history([(100,str(time.date())+"T01:00",1,2,99,0,10,900,10)])
    # The event is on the local issuance day and therefore excluded.
    a.attach_event_store(TemporalEventStore(pd.concat([s.history,extra]),s.team_seasons))
    pd.testing.assert_frame_equal(before,a.predict(pred)["predict_proba"])


def test_explicit_round_anchor_preserved_after_rescheduling():
    s=sample(); schedule=s.schedule
    schedule["round_issue_at"]=schedule.kickoff_at
    schedule.loc[0,"kickoff_at"]=pd.Timestamp("2020-05-01",tz="UTC")
    revised=TemporalEventStore(s.history,s.team_seasons,schedule=schedule)
    assert revised.issue_times(schedule.iloc[[0]],config()).iloc[0]==pd.Timestamp("2020-01-01",tz="UTC")

def test_reviewed_completion_bound_uses_existing_library_contract():
    s=sample(); h=s.history
    h["result_available_at"]=pd.NaT
    h["result_available_at"]=pd.to_datetime(h.result_available_at,utc=True)
    h.loc[h.event_id.eq(1),"result_available_at"]=pd.Timestamp("2020-02-01T00:00Z")
    store=TemporalEventStore(h,s.team_seasons)
    early=store.snapshot(pd.Timestamp("2020-01-30T12:00Z"),config())
    later=store.snapshot(pd.Timestamp("2020-02-02T12:00Z"),config())
    assert 1 not in set(early.edges.event_id)
    assert 1 in set(later.edges.event_id)


@neural
def test_cold_only_fit_and_large_nullable_league_exports():
    s=sample(); schedule=s.schedule
    big=2**63+99
    m=s.team_seasons
    m["competition_id"]=pd.array([big]*len(m),dtype="UInt64")
    empty=s.history.iloc[:0]
    cold=TemporalEventStore(empty,m,schedule=schedule)
    a=TemporalGraphAdapter(config(),cold)
    a.fit_controlled(context(s,[1,2]),TrainingControl(max_steps=1))
    q=pd.DataFrame({"team_id":[1,999],"issuance_at":pd.to_datetime(["2020-02-01T12:00Z"]*2)})
    z=a.encode_teams(q)
    assert z.league.iloc[0]==big and pd.isna(z.league.iloc[1])
    assert z.support.eq(0).all()
    np.testing.assert_array_equal(z.filter(regex="^z_").to_numpy(),z.filter(regex="^prior_").to_numpy())


@neural
def test_regularized_retained_zero_reference_and_context_fallback():
    import torch
    from xdiyo_analytics.graphs.network import TemporalNetwork
    model=TemporalNetwork(config(),1,5,5)
    with torch.no_grad():
        model.global_prior.fill_(.2)
        model.league.weight[1].fill_(.1)
        model.entry.weight[1].fill_(.05)
    league=torch.tensor([1,1,0]); entry=torch.tensor([0,1,1])
    prior=model.prior(league,entry).detach().numpy()
    np.testing.assert_allclose(prior[:,0],[.3,.35,.2],atol=1e-7)
    assert torch.equal(model.entry.weight[0],torch.zeros(4))


def test_selected_statistics_are_receiver_paired_and_masked():
    from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
    s=sample(); h=s.history
    col="team::ALL::Shots::shotsOnGoal::value"
    other=col.replace("team::","opponent::")
    h[col]=np.where(h.side.eq("home"),4.,np.nan)
    h[other]=np.where(h.side.eq("away"),4.,np.nan)
    enriched=TemporalEventStore(h,s.team_seasons)
    snap=enriched.snapshot(pd.Timestamp("2020-02-01T12:00Z"),config())
    c=config(statistics=(col,other))
    pre=GraphPreprocessor(c).fit(enriched.history,[snap],enriched.schedule,np.zeros((9,5)))
    features=pre.edges(snap)
    assert features[0,6:8].tolist()==[0.,1.]
    assert features[1,6:8].tolist()==[1.,0.]
    bad=h.copy(); bad.loc[1,other]=9.
    with pytest.raises(ValueError,match="Inconsistent"):
        TemporalEventStore(bad,s.team_seasons)
