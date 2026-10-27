"""Auditable saved-prediction report for the bounded-gap and opponent-propagation protocol."""
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

def plots(run,curves,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(run)/'03_comparisons';colors={core.FULL:'#12699a',core.PRIOR:'#bb6528'}
    by_job={(j['fold'],j['seed'],j['configuration']):c for j,c in curves};paired=[]
    names=['train_loss',core.MONITOR,'all13_inner_raw_rps_micro']
    fig,axes=plt.subplots(2,3,figsize=(18,9))
    for job,curve in curves:
        for j,name in enumerate(names):
            ax=axes[int(job['fold'][-1])-1,j]
            ax.plot(curve.step,curve[name],color=colors[job['configuration']],alpha=.7,
                linestyle={17:'-',42:'--',73:':'}[job['seed']],label=f"{'Full' if job['configuration']==core.FULL else 'Prior'} / {job['seed']}")
            ax.set(title=f"{job['fold']}: {name}",xlabel='Full-population optimizer updates',ylabel='RPS' if 'rps' in name else 'CE')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,fold in enumerate(core.FOLDS):
        for seed in core.SEEDS:
            a=by_job[fold,seed,core.FULL];b=by_job[fold,seed,core.PRIOR]
            pair=a[['step',core.MONITOR,'all13_inner_raw_rps_micro']].merge(b[['step',core.MONITOR,'all13_inner_raw_rps_micro']],on='step',suffixes=('_full','_prior'),validate='one_to_one')
            for j,(metric,column) in enumerate((('CE',core.MONITOR),('RPS','all13_inner_raw_rps_micro'))):
                pair[metric+'_difference']=pair[column+'_full']-pair[column+'_prior']
                axes[i,j].plot(pair.step,pair[metric+'_difference'],label=f'Seed {seed}')
                axes[i,j].set(title=f'{fold}: full minus prior inner {metric}',xlabel='Updates observed in both runs',ylabel=f'{metric} difference')
            pair['fold']=fold;pair['seed']=seed;paired.append(pair)
    for ax in axes.flat:ax.axhline(0,color='gray',linewidth=.8);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'paired_learning_curves.png',dpi=140);plt.close(fig)
    pd.concat(paired).to_csv(root/'paired_learning_curves.csv',index=False)
    primary=intervals[(intervals.fold=='combined')&(intervals.view=='raw')&(intervals.block_weeks==4)&(intervals.aggregation=='micro')]
    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    labels=['Full - prior','Full - archived time B','Full - pooled LR','Full - own-league LR','Prior - pooled LR','Prior - own-league LR']
    for ax,metric in zip(axes,('log_loss','rps')):
        for i,(label,_) in enumerate(analysis.contrasts()):
            row=primary[(primary.metric==metric)&(primary.contrast==label)].iloc[0]
            ax.hlines(i,row.lower,row.upper,color='#12699a');ax.plot(row.difference,i,'o',color='#12699a')
        ax.axvline(0,color='gray');ax.set_yticks(range(6),labels);ax.set_xlabel(f'Raw {metric}: difference and 95% four-week interval');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(root/'ce_rps_intervals.png',dpi=140);plt.close(fig)

def contrast_sentence(intervals,paired,league,label,metric):
    row=intervals[(intervals.contrast==label)&(intervals.fold=='combined')&(intervals.view=='raw')&
        (intervals.block_weeks==4)&(intervals.aggregation=='micro')&(intervals.metric==metric)].iloc[0]
    seeds=paired[(paired.contrast==label)&(paired.fold=='combined')&(paired.view=='raw')&(paired.aggregation=='micro')][metric]
    name='CE' if metric=='log_loss' else 'RPS'
    direction='favors the first model' if row.difference<0 else 'favors the second model' if row.difference>0 else 'is zero'
    uncertainty='The interval spans zero; this does not establish equivalence.' if row.lower<=0<=row.upper else 'The interval excludes zero under the declared conditional resampling procedure.'
    return f'{label}: {name} difference **{row.difference:+.6f}**, 95% interval [{row.lower:+.6f}, {row.upper:+.6f}]. The point estimate {direction}; {int((seeds<0).sum())}/3 matched seeds favor the first model. {uncertainty}'

def report(run):
    run=Path(run);p,decision,resources,curves,fits=collect(run);root=run/'03_comparisons';all_scores=[]
    resources.to_csv(root/'resources.csv',index=False)
    for phase in ('fit','inner','outer'):
        models=analysis.load_models(run,phase);scores=analysis.score_models(models,phase);all_scores.append(scores)
        if phase=='outer':
            intervals=analysis.paired_intervals(run,models);paired=analysis.paired_details(run,models,scores);outer_models=models
    scores=pd.concat(all_scores,ignore_index=True);scores.to_csv(root/'scores_by_seed.csv',index=False)
    grouping=['model','fold','split','view','aggregation','competition_id']
    means=scores.groupby(grouping,as_index=False)[['count',*analysis.NUMERIC]].mean();means.to_csv(root/'scores_mean.csv',index=False)
    choices=pd.DataFrame([dict(fold=row['fold'],configuration=cfg['configuration'],mean_inner_micro_ce=cfg['mean_inner_micro_ce'],
        selected=cfg['configuration']==row['configuration'],**{f'seed_{seed}':value for seed,value in zip(cfg['seeds'],cfg['inner_scores'])})
        for row in decision['fold_choices'] for cfg in row['configurations']]);choices.to_csv(root/'configuration_selection.csv',index=False)
    league=means[(means.fold=='combined')&(means.split=='outer')&(means.view=='raw')&(means.aggregation=='league')]
    league_differences=[]
    for label,terms in analysis.contrasts():
        for league_id in p['competitions']:
            values={metric:sum(sign*league[(league.model==name)&(league.competition_id==league_id)][metric].iloc[0]
                for name,sign in terms) for metric in analysis.NUMERIC}
            league_differences.append(dict(contrast=label,competition_id=league_id,**values))
    league_differences=pd.DataFrame(league_differences);league_differences.to_csv(root/'paired_league_differences.csv',index=False)
    case=[]
    for name in (core.FULL,core.PRIOR,'archived_time_B'):
        for seed,frame in zip(core.SEEDS,outer_models['fold_1',name]):
            row=frame[frame.event_id.eq(11425870)];assert len(row)==1
            case.append(dict(model=name,seed=seed,fold='fold_1',phase='outer',**row[[*core.MATCH_KEYS,*metrics.P_COLUMNS,'raw_log_loss','raw_rps']].iloc[0].to_dict()))
    case=pd.DataFrame(case);case.to_csv(run/'01_rest_fix/feature_diagnostics/case_11425870_predictions.csv',index=False)
    plots(run,curves,intervals)
    raw=means[(means.fold=='combined')&(means.split=='outer')&(means.view=='raw')&means.aggregation.isin(['micro','macro'])]
    primary=intervals[(intervals.fold=='combined')&(intervals.view=='raw')&(intervals.block_weeks==4)]
    all_views=means[(means.fold=='combined')&(means.split=='outer')&means.aggregation.isin(['micro','macro'])]
    main=analysis.contrasts()[0][0];repair=analysis.contrasts()[1][0]
    main_intervals=intervals[(intervals.contrast==main)&(intervals.view=='raw')&(intervals.block_weeks==4)]
    main_seeds=paired[(paired.contrast==main)&(paired.view=='raw')&(paired.aggregation=='micro')]
    cost=resources.groupby('configuration',as_index=False)[['fit_seconds','warm_update_seconds','allocated_gib','reserved_gib','rss_gib','parameters']].mean()
    resumed=sum(any(json.loads(path.read_text())['resumed'] for path in (run/f['job']['path']).glob('attempt_*/attempt.json')) for f in fits)
    failed=sum(len(list((run/f['job']['path']).glob('attempt_*/failure.txt'))) for f in fits)
    text='# Bounded observed gaps and opponent propagation\n\n'
    text+=f'Completed **12 fresh fits**, zero reused corrected controls, {resumed} resumed fits and {failed} failed engineering attempts. '
    text+=f'The own-fold inner rule recommends **{decision["latest_fold_recommendation"]}** for the next feature experiment. '
    text+='CE was the sole checkpoint/configuration selection metric. RPS is a predeclared substantive secondary outcome. These are previously inspected development periods; conclusions remain exploratory.\n\n'
    text+='## Gap contract and preservation\n\n'
    text+='For each side, elapsed days d are measured from the latest eligible archived match kickoff to **forecast issuance**. The new unscaled inputs are `d/(d+7)`, `no_eligible_history`, and `no_eligible_observation_30d`. Known history gives `(d/(d+7), 0, 1[d>30])`; absent history gives `(0,1,1)`. Exactly 30 days is recent. This measures observed archive coverage, not physiological rest. Negative/nonfinite known gaps are rejected.\n\n'
    text+='The neural fixture width changes from 24 to 26 and each joint decoder input from 152 to 154. All unrelated fitted scaler values, 14 edge-time coordinates, historical eligibility, target exclusions, ordered target IDs, stage-aware round clocks and data lower bound remain unchanged. Raw gaps and their actual observation timestamps survive only as diagnostic metadata. No raw standardized gap aliases or duplicate masks enter the corrected tensor. Existing models lacking encoding metadata load explicitly with the original transform and fingerprint.\n\n'
    text+='The original untrained encoder, unchanged outcome/goal columns and biases were copied by semantic column name. Equivalent missing-history columns were copied; four new continuous/30-day columns started at zero. Both arms share bit-identical 35,993-parameter initial states within each seed/fold. No trained warm start was used.\n\n'
    text+='For Cittadella–Reggiana (11425870), issuance is 2023-08-18 18:30 UTC, cutoff 2023-08-17 22:00 UTC and the latest eligible Reggiana observation is 2021-05-10 12:00 UTC. The 830.270833-day gap becomes approximately `(0.9916395,0,1)`; kickoff remains 2023-08-20 18:30 UTC. Its saved outer probabilities below are a case study, never a fitting or success criterion.\n\n'+table(case)+'\n\n'
    text+='[All phase/league input distributions and flag rates](01_rest_fix/feature_diagnostics/distributions.csv) · [Case provenance](01_rest_fix/feature_diagnostics/case_11425870.json) · [Population/input checks](shared/checks/correctness.json).\n\n'
    text+='## Sender intervention and correctness\n\n'
    text+='The prior-context control sends the current trainable layer-zero prior through both K and V at every layer. Receivers, queries, residuals, edge/time projections, attention normalization, support gates and own-history updates remain intact. Priors are recomputed each forward without detach or carried memory. One-layer outputs/gradients match; first-layer states/attention match in two-layer models, and second-layer states can differ. A fixed-topology J–K goal intervention changes full-model A–B predictions while leaving the control invariant. Both routes and nonzero prior/own-history/time/decoder gradients are tested on CPU and CUDA. No unchanged two-layer initial logits are assumed.\n\n'
    text+='Corrected persistence, legacy exact predictions, target exclusion, metadata validation, mode/schema cache identity and cross-mode continuation rejection passed. The complete-component execution policy is unchanged. [Focused tests](shared/checks/focused_tests.txt) · [Regressions](shared/checks/regression.txt) · [Frozen source/configuration](FREEZE.json).\n\n'
    text+='## Main comparison: updated opponent propagation\n\n'
    for metric in ('log_loss','rps'):text+=contrast_sentence(intervals,paired,league,main,metric)+'\n\n'
    text+=table(main_intervals[['fold','aggregation','metric','difference','lower','upper']])+'\n\n'
    text+=table(main_seeds[['fold','seed','log_loss','rps','draw_log_loss','winner_contribution']])+'\n\n'
    selected_leagues=league_differences[league_differences.contrast==main]
    text+=f'Updated propagation has lower combined raw CE in {int((selected_leagues.log_loss<0).sum())}/13 leagues and lower RPS in {int((selected_leagues.rps<0).sum())}/13. '
    text+='[League differences](03_comparisons/paired_league_differences.csv) preserve all leagues; no favorable subgroup was selected. The draw-CE interval is shown separately: a three-class or RPS improvement alone does not establish better draw prediction.\n\n'
    text+='## Repair and logistic comparisons\n\n'
    for metric in ('log_loss','rps'):text+=contrast_sentence(intervals,paired,league,repair,metric)+'\n\n'
    text+='The archived reference is the parent time B with its original standardized gap inputs and own selected checkpoint. This contrast describes the complete encoding repair and reselection, not bounding alone. The prior study’s C results and selections remain a separate historical record. Pooled and own-league logistic models and frequency references were reused from identical populations and clocks; none were refitted.\n\n'
    text+=table(primary[['contrast','aggregation','metric','difference','lower','upper']])+'\n\n![CE and RPS intervals](03_comparisons/ce_rps_intervals.png)\n\n'
    text+='## Outcome scores and selection\n\n'+table(raw[['model','aggregation','log_loss','rps','brier','draw_log_loss','draw_brier','winner_contribution','conditional_winner_log_loss','draw_auc','draw_ap']])+'\n\n'
    text+=table(choices)+'\n\n'
    for fold in core.FOLDS:
        rows=choices[choices.fold==fold].set_index('configuration');delta=rows.loc[core.FULL,'mean_inner_micro_ce']-rows.loc[core.PRIOR,'mean_inner_micro_ce']
        text+=f'{fold}: mean selected inner CE, full minus prior, is {delta:+.8f}. The fold-local decision is sealed before corrected outer scoring; exact ties prefer prior-context.\n\n'
    text+='Working architecture: retain the latest-fold inner winner for the next separately declared feature experiment. The scientific assessment above retains both CE and RPS, their magnitudes, seed/fold/league variation and uncertainty. Small favorable RPS changes are substantive evidence; an interval spanning zero neither proves equivalence nor persistence on future periods. A competitive prior-context control questions the incremental updated-opponent route under these inputs, without establishing that attention, temporal history or graph models are useless. A full-model benefit supports that route, without certifying the complete architecture.\n\n'
    text+='## Learning dynamics and cost\n\n'
    text+='Each arm is assessed at its own raw-inner-CE-selected checkpoint. Paired curves include only updates actually observed in both runs; no extrapolation or favorable comparison epoch. The curves distinguish optimization speed from different validation minima.\n\n![Learning curves](03_comparisons/learning_curves.png)\n\n![Paired learning curves](03_comparisons/paired_learning_curves.png)\n\n'
    text+=table(cost)+'\n\n'+table(resources[['configuration','fold','seed','selected_update','completed_updates','raw_inner_ce','fit_seconds','attempts']])+'\n\n'
    text+=f'Total fitting time was {resources.fit_seconds.sum()/3600:.3f} hours across {int(resources.completed_updates.sum())} full-population updates. '
    text+='Both models have the same parameter count. Runtime and CUDA/host peaks are measured rather than presumed. Maximum 200 updates, patience 15, min_delta zero, unchanged AdamW/regularization, earliest-best restoration. Last optimizer/RNG state and best inference weights are separate. Goal beta remains zero; dormant head rates are not trained goal forecasts. This study makes no claim about corrected C goal forecasts.\n\n'
    text+='## Aggregation, calibration and uncertainty\n\n'
    text+='Seeds are scored separately and then averaged. Micro weights the two outer folds by 4,315/8,590 and 4,275/8,590. Macro combines folds by original counts within each league, then weights all 13 leagues equally. Three-class Brier is summed over coordinates. RPS uses H/D/A order and the factor 1/2. CE equals draw CE plus the non-draw-weighted conditional winner CE. AUC/AP use concatenated fold fixtures within each seed for micro, and the equal mean of per-league scores for macro; no ranking-bootstrap claims are made.\n\n'
    text+='Intervals reuse the verified parent multiplicities: 20,000 paired Monday-start Europe/Berlin calendar resamples, seed 20261027, four-week blocks primary and two/eight-week sensitivity. Fold resampling is separate, with common weights across models/seeds and original fixed league/fold mixtures. Seed losses are averaged before resampling, never treated as replicated fixtures. Intervals condition on these fitted models and the observed seeds.\n\n'
    text+='Per-league draw intercepts are fitted only to each new model’s own inner predictions after raw selection. Calibrated scores are secondary and do not alter any selection.\n\n'+table(all_views[['model','view','aggregation','log_loss','rps','brier','draw_log_loss','winner_contribution']])+'\n\n'
    text+='[Every fold/seed/league metric](03_comparisons/scores_by_seed.csv) · [All paired intervals](03_comparisons/paired_intervals.csv) · [Per-fixture differences](03_comparisons/per_fixture_differences.parquet) · [Per-round differences](03_comparisons/per_round_differences.csv) · [Resource measurements](03_comparisons/resources.csv) · [Selection seal](SELECTION.json).\n\n'
    text+='The declared two-stage experiment is complete. Stop for review; no extra arms, seasons, likelihoods or feature families were added.\n'
    publish(run,'summary',text)
    for name,stage,label in ((core.FULL,'01_rest_fix','Observed-gap repair'),(core.PRIOR,'02_opponent_propagation','Opponent-propagation control')):
        scope=[repair,*[v[0] for v in analysis.contrasts()[2:] if name in v[0]]] if name==core.FULL else [main]
        report=f'# {label}\n\nSix fresh {name} fits. All six full controls are stored once in `01_rest_fix` and referenced by the matched comparison.\n\n'
        report+=table(resources[resources.configuration==name][['fold','seed','selected_update','completed_updates','raw_inner_ce','fit_seconds','parameters']])+'\n\n'
        report+=table(primary[primary.contrast.isin(scope)][['contrast','aggregation','metric','difference','lower','upper']])+'\n\n'
        report+='CE is the declared selection metric; RPS is a substantive secondary outcome with paired uncertainty. See the [complete report](../summary.html) for feature provenance, all metrics, matched initialization, diagnostics and interpretation.\n'
        publish(run/stage,'report',report)
    core.write_json(run/'REPORT.json',dict(status='reported',fits=12,resumed=resumed,failed_attempts=failed,
        selected_configuration=decision['latest_fold_recommendation'],scored_seeds=True,probability_ensemble=False,
        selection_sha256=core.digest(run/'SELECTION.json'),completed_at=pd.Timestamp.now(tz='UTC').isoformat()))

def audit(run):
    import torch
    run=Path(run);p,decision,resources,curves,fits=collect(run)
    assert len(fits)==12 and len(resources)==12 and resources.parameters.eq(35993).all()
    for fit in fits:
        folder=run/fit['job']['path'];found=next(a for a in reversed(sorted(folder.glob('attempt_*'))) if (a/'last_training.pt').exists())
        receipt=json.loads((found/'last_training.json').read_text());assert receipt['identity']==fit['identity']
        assert receipt['sha256']==core.digest(found/'last_training.pt')
        state=torch.load(found/'last_training.pt',map_location='cpu',weights_only=True)
        assert state['active']['step']==fit['summary']['attempts'][0]['steps']
        assert state['active']['best_step']==fit['summary']['best_step']
        assert len(state['active']['backend']['optimizer']['state'])>0
        semantics=state['active']['backend']['graph_semantics']
        assert semantics['sender_state_mode']==core.config(fit['job']['configuration']).sender_state_mode
    for fold in core.FOLDS:
        for seed in core.SEEDS:
            pair=[f for f in fits if f['job']['fold']==fold and f['job']['seed']==seed]
            assert len(pair)==2 and pair[0]['initial_digest']==pair[1]['initial_digest']
    protected=json.loads((run/'shared/checks/protected_artifacts.json').read_text())
    for path,expected in protected.items():assert core.digest(core.ROOT/path)==expected,path
    intervals=pd.read_csv(run/'03_comparisons/paired_intervals.csv');assert len(intervals)==13*3*2*3*2
    assert intervals.replicates.eq(20000).all() and intervals.undefined_replicates.eq(0).all()
    scores=pd.read_csv(run/'03_comparisons/scores_mean.csv')
    for row in intervals.itertuples():
        selected=scores[(scores.fold==row.fold)&(scores.split=='outer')&(scores.view==row.view)&(scores.aggregation==row.aggregation)]
        expected=sum(sign*selected[selected.model==name][row.metric].iloc[0] for name,sign in dict(analysis.contrasts())[row.contrast])
        np.testing.assert_allclose(row.difference,expected,rtol=1e-11,atol=1e-12)
    for job in p['matrix']:
        result=json.loads((run/job['path']/'result.json').read_text())
        assert pd.Timestamp(result['evaluated_at'])>=pd.Timestamp(decision['sealed_at'])
    core.write_json(run/'shared/checks/final_audit.json',dict(status='passed',fits=12,protected_files=len(protected),
        probability_and_independent_rps_audits=True,continuations_and_paired_initialization_verified=True,
        raw_selection_before_outer=True,all_468_contrast_points_reconciled=True))
    excluded={'COMPLETE.json','PROGRESS.json','STATUS.md','runner.log','audit.log','execution.lock'}
    files={path.relative_to(run).as_posix():core.digest(path) for path in run.rglob('*')
        if path.is_file() and path.name not in excluded and not path.name.endswith('.tmp')}
    core.write_json(run/'COMPLETE.json',dict(status='complete_reviewed',fits=12,files=files,
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(),recommendation=decision['latest_fold_recommendation']))
    print(json.dumps(dict(status='audited',fits=12,protected_files=len(protected))),flush=True)
