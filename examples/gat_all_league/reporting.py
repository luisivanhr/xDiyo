"""Auditable saved-prediction report for the fixed all-league protocol."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import core,metrics,selection,analysis,execution
from gat_suite.reporting import table
from gat_cross_league.reporting import publish,audit_predictions

def collect(run):
    run=Path(run);p,store=core.load(run,frozen=True);decision=selection.verify_gate(run)
    resources=[];curves=[];fits=[]
    for job in p['matrix']:
        folder=run/job['path'];fit=json.loads((folder/'fit.json').read_text());result=json.loads((folder/'result.json').read_text())
        assert fit['identity']==execution.identity(run,job) and result['fit_sha256']==core.digest(folder/'fit.json')
        assert result['selection_sha256']==core.digest(run/'SELECTION.json')
        curve=pd.read_csv(folder/'training.csv',float_precision='round_trip');execution.verify_progress(curve,fit['summary'])
        cal=json.loads((folder/'calibration.json').read_text());population=core.populations(run,job['fold'])
        for phase,expected in population.items():
            pred=pd.read_parquet(folder/f'{phase}_predictions.parquet');audit_predictions(pred,expected,cal)
            for key in ('round_stage','round_issue_at'):
                np.testing.assert_array_equal(pred[key].to_numpy(),expected[key].to_numpy())
            issue=store.issue_times(expected,core.config(job['configuration']))
            np.testing.assert_array_equal(pd.to_datetime(pred.issuance_at,utc=True).array,issue.array)
            for view,columns in [('raw',metrics.P_COLUMNS),('calibrated',metrics.C_COLUMNS)]:
                # An independent two-threshold expression, not metrics.rps().
                prob=pred[columns].to_numpy();y=pred.y.to_numpy()
                oracle=.5*((prob[:,0]-(y==0))**2+(prob[:,0]+prob[:,1]-(y!=2))**2)
                np.testing.assert_allclose(pred[f'{view}_rps'],oracle,atol=1e-14,rtol=1e-13)
                actual=metrics.scores(pred,view)
                for k in metrics.ADDITIVE:np.testing.assert_allclose(actual[k],result['metrics'][phase][view][k],atol=1e-12,rtol=1e-12)
            if phase=='inner':
                np.testing.assert_allclose(pred.raw_log_loss.mean(),fit['selection_score'],atol=1e-12,rtol=0)
                for league,part in pred.groupby('competition_id'):
                    fitted=metrics.base.draw_calibration(part[metrics.P_COLUMNS],part.y)
                    np.testing.assert_allclose(fitted['intercept'],cal[str(league)]['intercept'],rtol=1e-12,atol=1e-12)
            if phase!='outer':
                best=curve.set_index('step').loc[fit['summary']['best_step']]
                monitor=core.MONITOR if phase=='inner' else 'train_loss'
                np.testing.assert_allclose(pred.raw_log_loss.mean(),best[monitor],rtol=1e-5,atol=1e-6)
        beta=0.1 if job['configuration']=='time_C' else 0.
        for phase in ('train','validation'):
            np.testing.assert_allclose(curve[f'{phase}_objective'],curve[f'{phase}_loss']+beta*curve[f'{phase}_goal_loss']+curve[f'{phase}_regularization'],atol=1e-12,rtol=1e-12)
        assert fit['reload_exact'] and fit['public_boundaries_verified']
        timing=fit['training'];memory=timing['memory'];fits.append(fit)
        resources.append(dict(**job,selected_update=fit['summary']['best_step'],completed_updates=len(curve),parameters=fit['parameters'],
            raw_inner_ce=fit['selection_score'],fit_seconds=timing['fit_seconds'],preparation_seconds=timing['preparation_seconds'],
            warm_update_seconds=fit['warm_update_seconds'],outer_seconds=result['outer_seconds'],
            allocated_gib=timing['cuda_allocated_peak']/2**30,reserved_gib=timing['cuda_reserved_peak']/2**30,
            rss_gib=memory['sampled_peak_rss_bytes']/2**30,private_gib=memory['private_bytes']/2**30,attempts=fit['attempts']))
        curves.append((job,curve))
    return p,decision,pd.DataFrame(resources),curves,fits

def plots(run,curves,intervals,means,goals,calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(run)/'03_comparisons';colors=dict(snapshot_B='#627480',time_B='#135f99',time_C='#be6b15')
    for which,names in [('outcome',['train_loss',core.MONITOR,'all13_inner_raw_hda_ce_macro']),
                        ('goal',['train_goal_loss','validation_goal_loss']),('rps',['all13_inner_raw_rps_micro'])]:
        fig,axes=plt.subplots(2,len(names),figsize=(6*len(names),8),squeeze=False)
        for job,curve in curves:
            if which=='goal' and job['configuration']!='time_C':continue
            for j,name in enumerate(names):
                ax=axes[int(job['fold'][-1])-1,j]
                ax.plot(curve.step,curve[name],color=colors[job['configuration']],alpha=.7,
                    label=f'{job["configuration"]}, seed {job["seed"]}')
                ax.set(title=f'{job["fold"]}: {name}',xlabel='Full-population optimizer updates',ylabel='Loss' if which!='rps' else 'Normalized RPS')
        for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(root/f'{which}_curves.png',dpi=140);plt.close(fig)
    shared=[]
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    pairs=[('time_C','time_B'),('time_B','snapshot_B'),('time_C','snapshot_B')]
    by_job={(j['fold'],j['seed'],j['configuration']):c for j,c in curves}
    for f,fold in enumerate(core.FOLDS):
        for column,(left,right) in enumerate(pairs):
            for seed in core.SEEDS:
                a=by_job[fold,seed,left];b=by_job[fold,seed,right]
                pair=a[['step',core.MONITOR]].merge(b[['step',core.MONITOR]],on='step',suffixes=('_left','_right'),validate='one_to_one')
                pair['difference']=pair[f'{core.MONITOR}_left']-pair[f'{core.MONITOR}_right']
                pair['fold']=fold;pair['seed']=seed;pair['contrast']=f'{left} - {right}';shared.append(pair)
                axes[f,column].plot(pair.step,pair.difference,label=f'Seed {seed}')
            axes[f,column].set(title=f'{fold}: {left} minus {right}',xlabel='Updates observed in both runs',ylabel='All13 inner CE difference')
    for ax in axes.flat:ax.axhline(0,color='gray',linewidth=.8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'paired_learning_curves.png',dpi=140);plt.close(fig)
    pd.concat(shared).to_csv(root/'paired_learning_curves.csv',index=False)
    rows=intervals[(intervals.fold=='combined')&(intervals.view=='raw')&(intervals.block_weeks==4)&(intervals.aggregation=='micro')]
    fig,ax=plt.subplots(figsize=(12,7))
    for i,r in enumerate(rows.itertuples()):ax.hlines(i,r.lower,r.upper,color='#135f99');ax.plot(r.difference,i,'o',color='#135f99')
    ax.set_yticks(range(len(rows)),rows.contrast,fontsize=8);ax.axvline(0,color='gray');ax.set_xlabel('Raw outer micro CE difference; paired 95% four-week interval')
    fig.tight_layout();fig.savefig(root/'paired_differences.png',dpi=140);plt.close(fig)
    league=means[(means.fold=='combined')&(means.split=='outer')&(means.view=='raw')&(means.aggregation=='league')]
    view=league.pivot(index='competition_id',columns='model',values='log_loss')[['validation_selected','pooled_logistic','league_logistic']]
    ax=view.plot.bar(figsize=(14,6),color=['#135f99','#be6b15','#627480'],rot=0)
    ax.set(xlabel='Competition ID (league names in coverage table)',ylabel='Raw outer CE',title='Every league: selected procedure and logistic references')
    plt.tight_layout();plt.savefig(root/'league_comparison.png',dpi=140);plt.close()
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    cal=calibration[(calibration.split=='outer')&(calibration.fold=='combined')&(calibration.competition_id==0)]
    for i,side in enumerate(('home','away')):
        for (name,seed),part in cal[cal.side==side].groupby(['model','seed']):
            axes[i].plot(part.predicted,part.observed,'o-',alpha=.75,label=f'{name}, seed {seed}')
        axes[i].plot([0,4],[0,4],'k--');axes[i].set(title=f'{side.title()} goal rates',xlabel='Mean predicted goals within rate bin',ylabel='Observed mean goals')
        axes[i].legend(fontsize=7);axes[i].grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'goal_calibration.png',dpi=140);plt.close(fig)

def report(run):
    run=Path(run);p,decision,resources,curves,fits=collect(run);root=run/'03_comparisons'
    resources.to_csv(root/'resources.csv',index=False);all_scores=[];goal_scores=[];cal=[];quantiles=[]
    for phase in ('fit','inner','outer'):
        models=analysis.load_models(run,phase);scores=analysis.score_models(models,phase);all_scores.append(scores)
        g,c,q=analysis.goal_diagnostics(run,phase,models);goal_scores.append(g);cal.append(c);quantiles.append(q)
        if phase=='outer':
            intervals=analysis.paired_intervals(run,models);paired=analysis.paired_details(run,models,scores)
    scores=pd.concat(all_scores,ignore_index=True);scores.to_csv(root/'scores_by_seed.csv',index=False)
    grouping=['model','fold','split','view','aggregation','competition_id']
    means=scores.groupby(grouping,as_index=False)[['count',*analysis.NUMERIC]].mean();means.to_csv(root/'scores_mean.csv',index=False)
    goals=pd.concat(goal_scores,ignore_index=True);goals.to_csv(root/'goals_by_seed.csv',index=False)
    calibration=pd.concat(cal,ignore_index=True);calibration.to_csv(root/'goal_rate_calibration.csv',index=False)
    pd.concat(quantiles,ignore_index=True).to_csv(root/'goal_rate_quantiles.csv',index=False)
    plots(run,curves,intervals,means,goals,calibration)
    summary=means[(means.fold=='combined')&(means.split=='outer')&means.aggregation.isin(['micro','macro'])]
    primary=intervals[(intervals.fold=='combined')&(intervals.view=='raw')&(intervals.block_weeks==4)]
    selected=primary[primary.contrast.str.startswith('validation_selected')]
    choices=pd.DataFrame([dict(fold=row['fold'],configuration=cfg['configuration'],mean_inner_micro_ce=cfg['mean_inner_micro_ce'],
        selected=cfg['configuration']==row['configuration'],**{f'seed_{seed}':value for seed,value in zip(cfg['seeds'],cfg['inner_scores'])})
        for row in decision['fold_choices'] for cfg in row['configurations']])
    choices.to_csv(root/'configuration_selection.csv',index=False)
    repair=json.loads((run/'shared/checks/stage_repair.json').read_text());correct=json.loads((run/'shared/checks/correctness.json').read_text())
    counts=pd.DataFrame([dict(fold=f,phase=s,fixtures=v['count'],added=v['added']) for f,ss in p['folds'].items() for s,v in ss.items()])
    raw=summary[(summary.view=='raw')&(summary.aggregation=='micro')]
    league=means[(means.fold=='combined')&(means.split=='outer')&(means.view=='raw')&(means.aggregation=='league')]
    support=[]
    for baseline in ('pooled_logistic','league_logistic'):
        delta=league[league.model=='validation_selected'].set_index('competition_id').log_loss-league[league.model==baseline].set_index('competition_id').log_loss
        d=paired[(paired.fold=='combined')&(paired.view=='raw')&(paired.aggregation=='micro')&(paired.contrast==f'validation_selected - {baseline}')]
        support.append(f'Against {baseline}, the selected procedure improves CE in {int((delta<0).sum())}/13 leagues and {int((d.log_loss<0).sum())}/3 combined seed comparisons; seed differences range from {d.log_loss.min():+.6f} to {d.log_loss.max():+.6f}.')
    fixed=primary[(primary.aggregation=='micro')&primary.contrast.isin(['time_C - time_B','time_B - snapshot_B','time_C - snapshot_B'])]
    cost=resources.groupby('configuration',as_index=False)[['fit_seconds','warm_update_seconds','allocated_gib','rss_gib','parameters']].mean()
    goal_mean=goals[(goals.fold=='combined')&(goals.split=='outer')&(goals.competition_id==0)].groupby('model',as_index=False).mean(numeric_only=True)
    text='# Corrected clocks and all-league validation\n\n'
    text+=f'Completed all 18 declared pooled GAT fits, six necessary outcome-baseline updates and two goal-reference updates. Reused 48 outcome and 24 goal references after invariance checks. Latest-fold inner choice: **{decision["latest_fold_recommendation"]}**. This is the single configuration recommendation; no deployment was performed.\n\n'
    text+='These are previously inspected development periods. This experiment jointly changes Belgian timing and checkpoint selection, so differences from the previous experiment cannot isolate either change.\n\n'
    text+='## 1. Stage repair, population and causal history\n\n'
    text+=f'{repair["changed_clocks"]} Belgian clocks corrected; no non-Belgian clocks changed; {repair["membership_changes"]} membership boundaries changed. Stage identity is metadata, with unchanged numeric scalers and canonical untrained weights. All added rows are Belgian playoff fixtures.\n\n'+table(counts)+'\n\n'
    text+=table(pd.DataFrame(correct['real_examples'])[['event','old_issue','corrected_issue','eligible_late_belgian_edges','max_output_error']])+'\n\n'
    text+='Late-history counts above are directed graph edges, not doubled supervised targets. Target/future-result perturbations leave eligible inputs exactly unchanged. CUDA predictions agree within the existing 1e-6 absolute / 1e-5 relative policy. Full-stage anchors are retained when target rows are filtered. [Coverage, class proportions and stages](league_inventory.csv); [fixture input missingness](shared/checks/fixture_coverage.csv).\n\n'
    text+='## 2. Fold-local validation selection\n\n'+table(choices)+'\n\n'
    text+=f'Checkpoint selection and patience use **{core.MONITOR}**, the equal-fixture mean over all 13 inner leagues, once per update. Macro, goal, regularization, calibration and RPS metrics do not select checkpoints. Each fold chooses the lowest mean over its three retained seed scores; exact ties use snapshot_B, time_B, time_C. Choices were sealed before new GAT outer scoring. Fold 2 inner information was not used to choose the fold 1 configuration.\n\n'
    text+='## 3. Outer outcome results\n\n'+table(raw[['model','count','log_loss','brier','rps','draw_log_loss','draw_brier','conditional_winner_log_loss','winner_contribution','draw_auc','draw_ap']])+'\n\n'
    text+='Lower CE, Brier and RPS are better. RPS uses ordered home/draw/away classes and equals half the summed squared errors of the two cumulative probabilities. Its range is 0–1. The three-class CE is exactly draw CE plus the listed winner contribution; conditional-winner CE is evaluated among non-draws.\n\n'
    text+='Primary selected-procedure comparisons (negative differences favor the selected GAT):\n\n'+table(selected[['contrast','aggregation','difference','lower','upper']])+'\n\n'
    text+='Intervals use 20,000 paired common-calendar-week resamples, four-week blocks, original fixed league/fold weights, seed 20261027; schedule-only empty-stratum rejection. Two/eight-week sensitivities and both raw/calibrated views are in [paired_intervals.csv](03_comparisons/paired_intervals.csv). They describe uncertainty in this inspected development evaluation.\n\n'
    text+='![Paired differences](03_comparisons/paired_differences.png)\n\n'
    text+='## 4. League, seed and calibrated diagnostics\n\n'+'\n\n'.join(support)+'\n\n'
    text+=table(summary[['model','view','aggregation','log_loss','brier','rps','draw_log_loss','winner_contribution']])+'\n\n'
    text+='Micro combines corrected outer counts 4,315 and 4,275. Macro first combines folds within each league using its original counts, then averages all 13 leagues equally. AUC/AP are descriptive point scores from concatenated fixtures within each separately scored seed; macro averages league scores. Probability ensembles are never evaluated.\n\n'
    text+='![League comparison](03_comparisons/league_comparison.png)\n\n[All league/fold/seed scores](03_comparisons/scores_by_seed.csv) · [Paired seed differences](03_comparisons/paired_seed_differences.csv) · [Per-round differences](03_comparisons/per_round_differences.csv) · [Per-fixture differences](03_comparisons/per_fixture_differences.parquet).\n\n'
    text+='## 5. Fixed time and auxiliary comparisons\n\n'+table(fixed[['contrast','difference','lower','upper']])+'\n\n'
    text+='These contrasts compare each arm’s own outcome-selected checkpoint. The paired curves below include only updates observed in both runs and do not extrapolate stopped trajectories or select a comparison epoch. Auxiliary goal quality is a separate diagnostic and cannot establish outcome benefit.\n\n'
    text+='![Paired learning curves](03_comparisons/paired_learning_curves.png)\n\n![Outcome curves](03_comparisons/outcome_curves.png)\n\n![RPS curves](03_comparisons/rps_curves.png)\n\n'
    text+='## Goal diagnostics and computational cost\n\n'+table(goal_mean[['model','poisson_nll','home_mae','away_mae','mean_predicted_home','mean_observed_home','mean_predicted_away','mean_observed_away']])+'\n\n'
    text+='The goal score is the sum of two full marginal Poisson NLLs including exact factorial constants; it makes no claim of a joint independent goal distribution. The comparator uses each league’s fitting-only mean rates. Untrained B-arm goal heads are not goal baselines. [Rate distributions](03_comparisons/goal_rate_quantiles.csv) and [side-rate calibration](03_comparisons/goal_rate_calibration.csv) retain every league and seed.\n\n'
    text+='![Goal calibration](03_comparisons/goal_calibration.png)\n\n![Goal curves](03_comparisons/goal_curves.png)\n\n'+table(cost)+'\n\n'
    text+=table(resources[['configuration','fold','seed','selected_update','completed_updates','raw_inner_ce','fit_seconds','attempts']])+'\n\n'
    text+='Each update covers the full fitting population. Maximum 200 updates, patience 15, min_delta 0, earliest-best restoration, unchanged AdamW and regularization. Last-update optimizer/RNG continuations and best inference weights are saved separately. [Resources](03_comparisons/resources.csv) include CUDA allocated/reserved peaks and host RSS/private memory.\n\n'
    text+='## Verification and artifacts\n\n'
    text+='Focused tests cover stage repair, RPS normalization, causal inputs, all-league monitor weighting, foreign-label isolation, single validation pass, earliest-best restoration, resume identity, fold-local choice, fixed-mixture bootstrap and a fresh CUDA fit/export. Checkpoint probabilities round-trip exactly and public prediction boundaries remain enforced. [Checks](shared/checks/correctness.json), [tests](shared/checks/focused_tests.txt), [regressions](shared/checks/regression.txt), [baseline reuse](shared/checks/baseline_reuse.json), [source/input freeze](FREEZE.json), [selection seal](SELECTION.json).\n\n'
    text+='The declared experiment is complete. Stop for review; no beta expansion, extra fits or deployment.\n'
    publish(run,'summary',text)
    for name in core.CONFIGS:
        selected_rows=resources[resources.configuration==name]
        score=summary[summary.model==name]
        contrasts=primary[primary.contrast.str.contains(name,regex=False)]
        report=f'# {name}: corrected all-league experiment\n\nSix fits; checkpoint selection uses {core.MONITOR}. Each seed is scored separately.\n\n'
        report+=table(selected_rows[['fold','seed','selected_update','completed_updates','raw_inner_ce','fit_seconds','parameters']])+'\n\n'
        report+=table(score[['view','aggregation','log_loss','brier','rps','draw_log_loss','winner_contribution','draw_auc','draw_ap']])+'\n\n'
        report+=table(contrasts[['contrast','aggregation','difference','lower','upper']])+'\n\nSee the [complete report](../../summary.html) for coverage, goals, causal checks and interpretation.\n'
        publish(run/'01_pooled'/name,'report',report)
    core.write_json(run/'REPORT.json',dict(status='reported',fits=len(fits),scored_seeds=True,probability_ensemble=False,
        selected_configuration=decision['latest_fold_recommendation'],selection_sha256=core.digest(run/'SELECTION.json'),
        completed_at=pd.Timestamp.now(tz='UTC').isoformat()))
    print(json.dumps(dict(status='reported',fits=len(fits),recommendation=decision['latest_fold_recommendation'])),flush=True)

def audit(run):
    run=Path(run);p,decision,resources,curves,fits=collect(run)
    assert len(fits)==18 and len(resources)==18
    prepared=json.loads((run/'02_baselines/COMPLETE.json').read_text())
    assert prepared['updated_outcome_models']==6 and prepared['updated_goal_references']==2
    assert prepared['reused_outcome_models']==48 and prepared['reused_goal_references']==24
    import torch
    for fit in fits:
        folder=run/fit['job']['path'];found=None
        for attempt in reversed(sorted(folder.glob('attempt_*'))):
            if (attempt/'last_training.pt').exists():found=attempt;break
        assert found is not None
        receipt=json.loads((found/'last_training.json').read_text());assert receipt['identity']==fit['identity']
        assert receipt['sha256']==core.digest(found/'last_training.pt')
        state=torch.load(found/'last_training.pt',map_location='cpu',weights_only=True)
        assert state['active']['step']==fit['summary']['attempts'][0]['steps']
        assert state['active']['best_step']==fit['summary']['best_step']
        assert len(state['active']['backend']['optimizer']['state'])>0
    for item in prepared['models']:
        folder=run/item['path'];pop=core.populations(run,item['fold']);cal=json.loads((folder/'calibration.json').read_text())
        for phase,frame in pop.items():
            expected=frame if item['league']=='pooled' else frame[frame.competition_id.eq(int(item['league']))]
            pred=pd.read_parquet(folder/f'{phase}_predictions.parquet');audit_predictions(pred,expected,cal)
            for view in ('raw','calibrated'):
                np.testing.assert_allclose(pred[f'{view}_rps'].mean(),metrics.scores(pred,view)['rps'],atol=1e-14,rtol=0)
    protected=json.loads((run/'shared/checks/protected_artifacts.json').read_text())
    for path,expected in protected.items():assert core.digest(core.ROOT/path)==expected,path
    intervals=pd.read_csv(run/'03_comparisons/paired_intervals.csv');assert len(intervals)==11*3*2*3*2
    assert intervals.replicates.eq(20000).all() and intervals.undefined_replicates.eq(0).all()
    scores=pd.read_csv(run/'03_comparisons/scores_mean.csv')
    for row in intervals.itertuples():
        selected=scores[(scores.fold==row.fold)&(scores.split=='outer')&(scores.view==row.view)&(scores.aggregation==row.aggregation)]
        terms=dict(analysis.contrasts())[row.contrast]
        expected=sum(sign*selected[selected.model==name].log_loss.iloc[0] for name,sign in terms)
        np.testing.assert_allclose(row.difference,expected,rtol=1e-12,atol=1e-12)
    # Gate seal precedes every new GAT outer evaluation receipt.
    for job in p['matrix']:
        result=json.loads((run/job['path']/'result.json').read_text())
        assert pd.Timestamp(result['evaluated_at'])>=pd.Timestamp(decision['sealed_at'])
    core.write_json(run/'shared/checks/final_audit.json',dict(status='passed',fits=18,protected_files=len(protected),
        outcome_references=54,goal_references=26,probability_and_rps_audits=True,continuations_verified=True,
        raw_selection_before_outer=True,all_contrast_point_estimates_independently_reconciled=True))
    excluded={'COMPLETE.json','PROGRESS.json','STATUS.md','runner.log','audit.log'}
    files={path.relative_to(run).as_posix():core.digest(path) for path in run.rglob('*')
        if path.is_file() and path.name not in excluded and not path.name.endswith('.tmp')}
    core.write_json(run/'COMPLETE.json',dict(status='complete_reviewed',fits=18,files=files,
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(),recommendation=decision['latest_fold_recommendation']))
    print(json.dumps(dict(status='audited',fits=18,protected_files=len(protected))),flush=True)
