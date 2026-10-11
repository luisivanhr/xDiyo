"""Bounded By Round benchmark and same-encoder three-block feature reuse.

Run from an installed checkout:
python examples/temporal_team_gat.py --data-root PATH --output OUT --device cuda:0
Input files remain read-only. OUT must be new and outside the source data root.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from xdiyo_analytics.data import load_season
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.config import CLASS_ORDER, MATCH_KEYS
from xdiyo_analytics.training import TrainingControl, EarlyStopping
from xdiyo_analytics.training.contracts import FitContext, PredictionContext


def make_context(fixtures, *, fitting=True):
    X = pd.DataFrame({"graph_input": np.zeros(len(fixtures))}, index=fixtures.index)
    common = dict(X=X, metadata=fixtures.copy(), layout="match", match_columns=MATCH_KEYS, fold_id=0)
    if not fitting:
        return PredictionContext(**common)
    labels = np.where(fixtures.goals_for > fixtures.goals_against, 1,
                      np.where(fixtures.goals_for < fixtures.goals_against, -1, 0))
    return FitContext(**common, y=pd.DataFrame({"result": labels}, index=fixtures.index))


def scores(y, probabilities):
    y = np.asarray([CLASS_ORDER.index(x) for x in y])
    p = np.asarray(probabilities)
    truth = np.eye(3)[y]
    confidence, predicted = p.max(1), p.argmax(1)
    bins = np.minimum((confidence*10).astype(int),9)
    calibration = []
    ece = 0.
    for b in sorted(set(bins)):
        keep = bins==b
        accuracy = float(np.mean(predicted[keep]==y[keep]))
        mean_confidence = float(confidence[keep].mean())
        ece += keep.mean()*abs(accuracy-mean_confidence)
        calibration.append(dict(bin=int(b),count=int(keep.sum()),confidence=mean_confidence,accuracy=accuracy))
    return dict(count=len(y),log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
                brier=float(((p-truth)**2).sum(1).mean()),ece_10=float(ece),calibration=calibration)


def frozen_fixture_embeddings(model, fixtures):
    """One fitted coordinate system only; refuse pre-training-boundary exports."""
    issue = model.event_store.issue_times(fixtures,model.config)
    query = pd.DataFrame({
        "team_id": pd.array([*fixtures.home_id.tolist(),*fixtures.away_id.tolist()],dtype="UInt64"),
        "issuance_at": [*issue.tolist(),*issue.tolist()]})
    exported = model.encode_teams(query)
    if exported.model_version.nunique()!=1:
        raise ValueError("Cannot pool embeddings from different encoder coordinate systems.")
    z=exported.filter(regex="^z_").to_numpy()
    a,b=z[:len(fixtures)],z[len(fixtures):]
    return np.column_stack((a,b,a-b,a*b)),exported


def baseline_features(store, config, fixtures):
    """Existing native rolling-goals expressions, using the identical cutoffs."""
    from xdiyo_analytics.features import evaluate_features, RollingMean, MatchScore, ForAgainst
    h=store.history
    schedule=store.schedule
    issued=store.issue_times(schedule,config)
    issue_map=dict(zip(schedule.event_id.tolist(),issued.tolist()))
    cutoffs=pd.Series([config.history_cutoff(issue_map[e]) for e in h.event_id],index=h.index)
    definitions={
        "goals_for": RollingMean(ForAgainst(MatchScore(score_field="current"),"for"),window=10,min_periods=1,venue="all"),
        "goals_against": RollingMean(ForAgainst(MatchScore(score_field="current"),"against"),window=10,min_periods=1,venue="all"),
    }
    f=evaluate_features(h,definitions,group_by=("team_id",),cutoffs=cutoffs,available_at=h.kickoff_at)
    keyed=pd.concat([h[["event_id","side"]],f.reset_index(drop=True)],axis=1).set_index(["event_id","side"])
    a=keyed.loc[[(e,"home") for e in fixtures.event_id]].to_numpy(dtype=float)
    b=keyed.loc[[(e,"away") for e in fixtures.event_id]].to_numpy(dtype=float)
    return np.column_stack((a,b,a-b))


def classifier():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),
                         LogisticRegression(C=1.,max_iter=500,random_state=17))


def class_probabilities(model,X):
    p=model.predict_proba(X)
    return p[:,[list(model.classes_).index(c) for c in CLASS_ORDER]]


def load_inputs(root):
    histories,memberships,provenance=[],[],[]
    # Entire available 2010/11-2024/25 German two-division population, no age truncation.
    for year in range(10,25):
        for league in ("Bundesliga","Bundesliga_2"):
            stem=f"{league}_{year:02}_{year+1:02}"
            d=load_season(root,stem,tables=("matches","team_seasons"),verify_hashes=True)
            h=build_team_history(d)
            h["source_season"]=f"{year:02}_{year+1:02}"
            h["source_league"]=league
            histories.append(h)
            m=d["team_seasons"].copy()
            # Declared effective domestic season boundary. Reviewed movement
            # metadata is treated as known at July 1 for this retrospective example.
            m["entry_at"]=pd.Timestamp(f"20{year:02}-07-01",tz="Europe/Berlin").tz_convert("UTC")
            memberships.append(m)
            provenance.append(d.provenance)
    h=pd.concat(histories,ignore_index=True)
    m=pd.concat(memberships,ignore_index=True)
    return TemporalEventStore(h,m,provenance={"sources":provenance,"season_entry_assumption":"July 1 local midnight"})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--max-steps",type=int,default=12)
    args=parser.parse_args()
    root,out=args.data_root.resolve(),args.output.resolve()
    if out.is_relative_to(root) or out.exists():
        raise ValueError("output must be a new directory outside the immutable source data root.")
    out.mkdir(parents=True)
    start=time.perf_counter()
    store=load_inputs(root)
    cfg=TemporalGraphConfig("Europe/Berlin")
    schedule=store.schedule
    target=schedule[schedule.source_league.eq("Bundesliga") & schedule.status.eq("finished") &
                    schedule.goals_for.notna() & schedule.goals_against.notna()].copy()
    fit=target[target.source_season.eq("21_22")]
    valid=target[target.source_season.eq("22_23")]
    test=target[target.source_season.eq("23_24")]
    downstream_test=target[target.source_season.eq("24_25")]
    fitting=make_context(fit); fitting.validation=make_context(valid)
    control=TrainingControl(max_steps=args.max_steps,early_stopping=EarlyStopping(patience=3),seed=17)
    report=dict(config=asdict(cfg),control=asdict(control),device=args.device,
                source_event_store_version=store.version,sources=store.provenance,
                populations={},validation_candidates=[],test={},downstream={})
    for name,frame in (("encoder_fit",fit),("encoder_validation",valid),("standalone_test_and_downstream_fit",test),("downstream_test",downstream_test)):
        issue=store.issue_times(frame,cfg)
        report["populations"][name]=dict(count=len(frame),first_issue=str(issue.min()),last_issue=str(issue.max()),
            first_kickoff=str(frame.kickoff_at.min()),last_kickoff=str(frame.kickoff_at.max()))
    report["historical_population"]=dict(matches=len(schedule),directed_edges=len(store.history),
        first_kickoff=str(schedule.kickoff_at.min()),last_kickoff=str(schedule.kickoff_at.max()),
        extra_statistics=[],neutral_venue="missing; explicit mask",
        context_assumption="Reviewed movement records known at July 1 local midnight; actual schedule export, no historical schedule-vintage claims.")
    best={}
    for layers in (1,2):
        for half_life in (90.,180.,365.):
            tick=time.perf_counter()
            model=TemporalGraphAdapter(replace(cfg,layers=layers,half_life_days=half_life),store,control=control,device=args.device)
            def progress(event):
                if not event.final:
                    print(f"layers={layers} H={half_life:g} step={event.step} validation={event.metrics['validation_loss']:.5f}",flush=True)
            fitting.observer=progress
            model.fit(fitting)
            loss=model.training_summary_["monitored_value"]
            report["validation_candidates"].append(dict(layers=layers,half_life_days=half_life,
                validation_log_loss=loss,seconds=time.perf_counter()-tick,training=model.training_summary_))
            if layers not in best or loss<best[layers][0]:
                best[layers]=(loss,model)
            (out/"progress.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    prediction_frames=[]
    ytest=make_context(test).y.result
    movement=store.team_seasons.set_index(["competition_id","season_id","team_id"]).movement
    moved=np.array([movement.get((c,s,h),"unknown") in ("promoted","relegated") or
                    movement.get((c,s,a),"unknown") in ("promoted","relegated")
                    for c,s,h,a in test[["competition_id","season_id","home_id","away_id"]].itertuples(index=False,name=None)])
    groups={"all":np.ones(len(test),bool),"rounds_1_to_5":test["round"].le(5).to_numpy(dtype=bool),
            "movement_fixture":moved,"retained_or_unknown_fixture":~moved}
    def evaluate(name,probabilities):
        report["test"][name]={g:scores(ytest.iloc[np.flatnonzero(mask)],np.asarray(probabilities)[mask])
                              for g,mask in groups.items() if mask.sum()>=20}
        p=pd.DataFrame(np.asarray(probabilities),columns=CLASS_ORDER,index=test.index)
        p["model"]=name
        for col in MATCH_KEYS:
            p[col]=test[col]
        prediction_frames.append(p)
    for layers,(_,model) in best.items():
        evaluate(f"gat_{layers}_layer",model.predict(make_context(test,fitting=False))["predict_proba"])
        TemporalGraphSerializer(store).save(model,out/f"gat_{layers}_layer")
    print("Building native rolling-goals baseline",flush=True)
    all_targets=pd.concat([fit,valid,test,downstream_test])
    rolling=baseline_features(store,cfg,all_targets)
    nfit,nvalid,ntest=len(fit),len(valid),len(test)
    baseline=classifier()
    baseline.fit(rolling[:nfit],make_context(fit).y.result)
    evaluate("native_rolling_goals_10",class_probabilities(baseline,rolling[nfit+nvalid:nfit+nvalid+ntest]))
    # Three blocks: encoder (fit+validation), downstream estimator fit, evaluation.
    frozen=best[2][1]
    print("Exporting one frozen encoder for downstream blocks",flush=True)
    train_z,train_export=frozen_fixture_embeddings(frozen,test)
    eval_z,eval_export=frozen_fixture_embeddings(frozen,downstream_test)
    downstream=classifier()
    if cfg.labels_known_at(test.kickoff_at).max() > min(cfg.history_cutoff(t) for t in store.issue_times(downstream_test,cfg)):
        raise ValueError("Downstream fitting labels overlap evaluation issuance.")
    if set(train_export.model_version) != set(eval_export.model_version):
        raise ValueError("Downstream blocks must share one frozen encoder version.")
    downstream.fit(train_z,ytest)
    p=class_probabilities(downstream,eval_z)
    report["downstream"]=dict(scores=scores(make_context(downstream_test).y.result,p),
        model_version=frozen.backend_.model_version,encoder_trained_through=str(frozen.backend_.trained_through),
        encoder_block="2021/22 fitting + 2022/23 validation selection",
        downstream_fit_block="2023/24",evaluation_block="2024/25")
    exported=pd.concat([train_export.assign(block="downstream_fit"),eval_export.assign(block="downstream_test")],ignore_index=True)
    # Evidence tuples are serialized explicitly for portable parquet.
    exported["evidence"]=exported.evidence.map(lambda x:json.dumps(x))
    exported.to_parquet(out/"frozen_encoder_exports.parquet",index=False)
    pd.concat(prediction_frames).to_parquet(out/"standalone_predictions.parquet",index=False)
    report["elapsed_seconds"]=time.perf_counter()-start
    report["limitations"]=[
        "One country, one seed, 12-step default compute cap; no predictive superiority claim.",
        "Movement metadata and schedule are retrospective exports under the declared operational previous-day contract.",
        "No release timestamp reconstruction; season entry context assumes reviewed entries known at July 1.",
        "Calibration is descriptive (10 confidence bins), with small movement/early-season subgroups.",
        "The same encoder is frozen for both downstream blocks; no cross-fit coordinate pooling."]
    (out/"results.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(json.dumps({k:{"log_loss":v["all"]["log_loss"],"count":v["all"]["count"]} for k,v in report["test"].items()},indent=2),flush=True)


if __name__=="__main__":
    main()
