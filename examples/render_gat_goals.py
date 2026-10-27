"""Verify and report the frozen seven-fit goal auxiliary comparison."""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_gat_goals as experiment
from compare_gat_temporal import digest, write_json, validate_training_progress


def markdown_table(frame):
    def cell(x):
        if isinstance(x, (float, np.floating)): return f'{x:.6f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def report(run):
    protocol,store,frames=experiment.load(run)
    expected={(x['arm'],x['seed']) for x in protocol['matrix']}
    results={}
    curves={}
    tables={}
    for arm,seed in sorted(expected):
        folder=run/'fit'/f'{arm}-{seed}'
        r=json.loads((folder/'result.json').read_text())
        if (r['arm'],r['seed'])!=(arm,seed) or r['protocol_sha256']!=digest(run/'protocol.json'):
            raise AssertionError('Result/protocol mismatch.')
        if r['config']['goal_auxiliary']!=protocol['arms'][arm] or not r['reload_exact'] or not r['public_boundary_enforced']:
            raise AssertionError('Arm or reload verification failed.')
        progress=json.loads((folder/'progress.json').read_text())
        validate_training_progress(progress,r['summary'],dict(max_steps=protocol['control']['max_steps'],early_stopping=protocol['control']['early_stopping']))
        curve=pd.DataFrame(progress).set_index('step')
        for split in ('train','validation'):
            # Recompute every retained metric independently from saved predictions/labels.
            table=pd.read_parquet(folder/f'{split}_predictions.parquet')
            if not table[list(experiment.MATCH_KEYS)].reset_index(drop=True).equals(frames[split][list(experiment.MATCH_KEYS)].reset_index(drop=True)):
                raise AssertionError('Scored fixture identities changed.')
            y=experiment.goal_labels(store,frames[split],store.label_availability(frames[split],experiment.TemporalGraphConfig(**protocol['config'])).max())
            np.testing.assert_array_equal(y,table[['goals_home','goals_away']].to_numpy())
            metrics,_=experiment.score_population(frames[split],table[['p_home','p_draw','p_away']].to_numpy(),table[['lambda_home','lambda_away']].to_numpy(),y)
            for name in ('log_loss','brier','draw_log_loss','draw_brier','winner_contribution','conditional_winner_log_loss','full_poisson_nll','home_mae','away_mae','goal_loss'):
                np.testing.assert_allclose(metrics[name],r[split][name],rtol=1e-12,atol=1e-12)
            beta=protocol['arms'][arm]['beta']
            np.testing.assert_allclose(curve[f'{split}_objective'],curve[f'{split}_loss']+beta*curve[f'{split}_goal_loss']+curve[f'{split}_regularization'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(metrics['log_loss'],curve.loc[r['summary']['best_step'],f'{split}_loss'],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(metrics['goal_loss'],curve.loc[r['summary']['best_step'],f'{split}_goal_loss'],rtol=1e-5,atol=1e-6)
            tables[arm,seed,split]=table
        for point in ('initial','selected'):
            if not r[point]['state_unchanged']: raise AssertionError('Diagnostic mutated state.')
        results[arm,seed],curves[arm,seed]=r,curve
    if set(results)!=expected: raise AssertionError('Wrong fit matrix.')
    # Detached outcome branch must remain numerically equivalent across its entire trajectory.
    a,b=curves['control',17],curves['detached',17]
    if not a.index.equals(b.index): raise AssertionError('Detached stopping point differs.')
    detached_max={}
    for field in ('train_loss','validation_loss','train_regularization','validation_regularization'):
        np.testing.assert_allclose(a[field],b[field],rtol=1e-5,atol=1e-6)
        detached_max[field]=float(np.max(np.abs(a[field]-b[field])))
    if results['control',17]['summary']['best_step']!=results['detached',17]['summary']['best_step']:
        raise AssertionError('Detached selected checkpoint differs.')
    for split in ('train','validation'):
        aa=tables['control',17,split][['p_home','p_draw','p_away']].to_numpy()
        bb=tables['detached',17,split][['p_home','p_draw','p_away']].to_numpy()
        np.testing.assert_allclose(aa,bb,rtol=1e-5,atol=1e-6)
        detached_max[split+'_probabilities']=float(np.max(np.abs(aa-bb)))
    paired=[]; fixture=[]; matched_curves=[]
    for seed in protocol['seeds']:
        if results['attached',seed]['initial_weights_sha256']!=results['control',seed]['initial_weights_sha256']:
            raise AssertionError('Initial states were not matched.')
        for split in ('train','validation'):
            candidate,baseline=results['attached',seed][split],results['control',seed][split]
            row=dict(seed=seed,split=split)
            for key in ('log_loss','brier','draw_log_loss','draw_brier','conditional_winner_log_loss','winner_contribution','full_poisson_nll','home_mae','away_mae'):
                row[key]=candidate[key]-baseline[key]
            paired.append(row)
            table=tables['attached',seed,split][[*experiment.MATCH_KEYS,'round','kickoff_at']].copy()
            table['seed'],table['split']=seed,split
            for key in ('log_loss','brier','draw_log_loss','draw_brier','winner_contribution','full_poisson_nll'):
                table[key]=tables['attached',seed,split][key].to_numpy()-tables['control',seed,split][key].to_numpy()
            fixture.append(table)
        aa,bb=curves['attached',seed],curves['control',seed]
        common=aa.index.intersection(bb.index)
        difference=pd.DataFrame(dict(seed=seed,step=common))
        for field in ('train_loss','validation_loss','train_goal_loss','validation_goal_loss'):
            difference[field]=aa.loc[common,field].to_numpy()-bb.loc[common,field].to_numpy()
        matched_curves.append(difference)
    paired=pd.DataFrame(paired); paired.to_csv(run/'paired_metrics.csv',index=False)
    fixture=pd.concat(fixture,ignore_index=True); fixture.to_parquet(run/'paired_fixture_losses.parquet',index=False)
    metrics=['log_loss','brier','draw_log_loss','draw_brier','winner_contribution','full_poisson_nll']
    rounds=fixture.groupby(['split','seed','competition_id','season_id','round'],dropna=False)[metrics].mean().reset_index()
    rounds.to_csv(run/'paired_round_losses.csv',index=False)
    matched_curves=pd.concat(matched_curves,ignore_index=True); matched_curves.to_csv(run/'paired_common_epoch_curves.csv',index=False)
    summary=[]; gradients=[]; ranges=[]
    for (arm,seed),r in results.items():
        for split in ('train','validation'):
            s=r[split]
            summary.append(dict(arm=arm,seed=seed,split=split,completed=r['summary']['attempts'][0]['steps'],selected=r['summary']['best_step'],
                fit_seconds=r['fit_seconds'],parameters=r['parameters'],**{key:s[key] for key in ('log_loss','brier','draw_log_loss','draw_brier','conditional_winner_log_loss','winner_contribution','non_draw_fraction','decomposition_error','full_poisson_nll','goal_loss','home_mae','away_mae')},
                mean_home=s['home_rate']['mean'],mean_away=s['away_rate']['mean'],observed_home=s['observed_mean_home'],observed_away=s['observed_mean_away']))
            for side in ('home','away'):
                ranges.append(dict(arm=arm,seed=seed,split=split,side=side,mean=s[side+'_rate']['mean'],std=s[side+'_rate']['std'],**s[side+'_rate']['quantiles']))
        for point in ('initial','selected'):
            g=r[point]
            gradients.append(dict(arm=arm,seed=seed,checkpoint=point,**{k:g[k] for k in ('outcome_norm','goal_norm','weighted_gradient_ratio','cosine_similarity')},zero_norms=','.join(g['zero_norms'])))
    summary=pd.DataFrame(summary); summary.to_csv(run/'summary.csv',index=False)
    gradients=pd.DataFrame(gradients); gradients.to_csv(run/'gradients.csv',index=False)
    ranges=pd.DataFrame(ranges)
    ranges.to_csv(run/'goal_rate_distributions.csv',index=False)
    cfg=experiment.TemporalGraphConfig(**protocol['config'])
    yfit=experiment.goal_labels(store,frames['train'],store.label_availability(frames['train'],cfg).max())
    means=yfit.mean(0)
    constants=[]
    for split,frame in frames.items():
        ys=experiment.goal_labels(store,frame,store.label_availability(frame,cfg).max())
        rates=np.broadcast_to(means,ys.shape)
        # Outcome probabilities are irrelevant to this comparator; only goal columns are retained.
        scores,_=experiment.score_population(frame,np.full((len(frame),3),1/3),rates,ys)
        constants.append(dict(split=split,mean_home=means[0],mean_away=means[1],**{k:scores[k] for k in ('full_poisson_nll','home_mae','away_mae')}))
    constants=pd.DataFrame(constants); constants.to_csv(run/'constant_goal_comparator.csv',index=False)
    # Prior results and frozen source remain byte-for-byte preserved.
    for name,expected_hash in protocol['preserved_artifacts'].items():
        if digest(experiment.ROOT/name)!=expected_hash: raise AssertionError(f'Prior artifact changed: {name}')
    parent_errors={}
    for seed in protocol['seeds']:
        path=Path(protocol['parent'])/'fit'/f'snapshot-{seed}'/'progress.json'
        previous=pd.DataFrame(json.loads(path.read_text())).set_index('step')
        current=curves['control',seed]
        common=previous.index.intersection(current.index)
        np.testing.assert_allclose(previous.loc[common,'validation_loss'],current.loc[common,'validation_loss'],rtol=1e-5,atol=1e-6)
        parent_errors[str(seed)]=float(np.max(np.abs(previous.loc[common,'validation_loss']-current.loc[common,'validation_loss'])))
    validation=paired[paired.split.eq('validation')]
    improvement=int(validation.log_loss.lt(0).sum())
    change=float(validation.log_loss.mean())
    screen=change<0 and improvement>=2
    mean=summary.groupby(['arm','split']).mean(numeric_only=True).reset_index()
    mean.to_csv(run/'mean_separately_scored_seeds.csv',index=False)
    make_plots(run,curves,matched_curves,rounds)
    attached=mean[mean.arm.eq('attached') & mean.split.eq('validation')].iloc[0]
    baseline=mean[mean.arm.eq('control') & mean.split.eq('validation')].iloc[0]
    detached=results['detached',17]['validation']
    const=constants[constants.split.eq('validation')].iloc[0]
    runtime=float(attached.fit_seconds/baseline.fit_seconds)
    goal_parameters=results['control',17]['goal_parameters']
    selected_grad=gradients[gradients.arm.eq('attached') & gradients.checkpoint.eq('selected')]
    questions=[
        f"1. **Validation outcome:** attached minus control is {change:+.6f} log loss on average. The preliminary screen {'passes' if screen else 'does not pass'}.",
        f"2. **Seed consistency:** {improvement}/3 matched seeds improve at their own outcome-selected checkpoints. The complete per-seed differences appear below.",
        f"3. **Draws and winners:** mean draw-log-loss change is {float(validation.draw_log_loss.mean()):+.6f}; conditional winner-log-loss change is {float(validation.conditional_winner_log_loss.mean()):+.6f}. Negative values improve the corresponding task. A three-class gain alone is not evidence of a draw gain.",
        f"4. **Goal versus outcome gains:** attached validation marginal Poisson NLL is {attached.full_poisson_nll:.6f}, versus {const.full_poisson_nll:.6f} for fitting-only constant rates. {'The outcome screen also passes, provisionally.' if screen else 'The outcome screen fails, regardless of goal-task improvements.'} The untrained control goal head is not a meaningful goal baseline.",
        f"5. **Readable baseline information:** the detached seed-17 head reaches validation goal NLL {detached['full_poisson_nll']:.6f}, home/away MAE {detached['home_mae']:.6f}/{detached['away_mae']:.6f}, while its maximum validation outcome-curve discrepancy from control is {detached_max['validation_loss']:.3g}. {'It beats the constant marginal NLL, suggesting useful goal information was readable without auxiliary encoder updates.' if detached['full_poisson_nll']<const.full_poisson_nll else 'It does not beat the constant marginal NLL in this diagnostic.'}",
        f"6. **Gradient scale:** at the attached selected checkpoints, beta-weighted goal/outcome norm ratios range from {selected_grad.weighted_gradient_ratio.min():.4f} to {selected_grad.weighted_gradient_ratio.max():.4f}; cosine similarities range from {selected_grad.cosine_similarity.min():.4f} to {selected_grad.cosine_similarity.max():.4f}. These are local full-fitting-population measurements, not generalization evidence; AdamW transforms the gradients.",
        f"7. **Optimization and attained minima:** the primary difference {change:+.6f} compares the best validation checkpoints observed within the declared budgets. Paired curves below show every epoch observed in both runs. The stopped suffix is not extrapolated, and no favorable common epoch is selected. These runs cannot identify an optimization limit beyond their observed trajectories.",
        f"8. **Cost:** attached fits average {runtime:.3f} times the control runtime under the same policy. {'This is a candidate for separately declared evaluation, with the measured cost and exploratory uncertainty retained.' if screen else 'The observed outcome evidence does not justify enabling the auxiliary objective by default.'}"
    ]
    validation_table=summary[summary.split.eq('validation')][['arm','seed','completed','selected','log_loss','brier','draw_log_loss','draw_brier','conditional_winner_log_loss','fit_seconds','parameters']]
    goal_table=summary[summary.split.eq('validation')][['arm','seed','full_poisson_nll','home_mae','away_mae','mean_home','mean_away','observed_home','observed_away']]
    text='\n\n'.join([
        '# Auxiliary goal supervision: frozen snapshot comparison',
        f"**Exploratory result: preliminary screen {'passes' if screen else 'does not pass'}.** Mean attached-control validation outcome log-loss difference is **{change:+.6f}**, with improvements in **{improvement}/3** seeds. Lower losses are better. Goal-forecast improvements alone do not count as outcome success.",
        '## Protocol and scope',
        f'Snapshot mode, two layers, identical 306-fitting/306-validation fixture populations from the reference experiment, unchanged graph inputs, preprocessing, AdamW and regularization. All arms contain the same {goal_parameters}-parameter linear goal head. The seven fits are control beta=0 and attached beta=0.1 at seeds 17/42/73, plus detached beta=0.1 at seed 17. All initial shared, outcome-head and goal-head tensors are copied from saved states within seed; all 612 initial fixture probabilities and rates match exactly. Initial logits also match the original outcome-only model exactly.',
        'Maximum 200 complete-population updates, patience 15, min_delta=0, best-checkpoint restoration, selection only by validation outcome loss. Training and validation curves are reevaluated after each update; retained metrics are recomputed after restoration. Fit runtime includes initialization verification and initial gradient diagnostics, training, validation and post-update training curves. Selected-checkpoint gradients are timed separately; total wall time also includes retained scoring and export/reload verification. No new test predictions or test-score selection occurred.',
        'The reference data have already been inspected. These are development findings, not fresh confirmatory evidence. Seed results are scored separately before averaging; no probability ensemble is used. Goal targets are raw, oriented counts and each fixture contributes once. Two marginal losses are summed, then averaged over globally weighted fixtures. Full reported Poisson NLL adds exact log-factorial constants. The training goal loss omits these constants. The separately logged regularizer is the prior-embedding penalty; decoupled AdamW weight decay remains an unchanged optimizer operation. Total objectives at different beta values are not performance comparisons.',
        '## Questions and findings', '\n\n'.join(questions),
        '## Validation outcomes at each arm\'s selected checkpoint',markdown_table(validation_table),
        '## Per-seed paired changes: attached minus control',markdown_table(validation[['seed','log_loss','brier','draw_log_loss','draw_brier','conditional_winner_log_loss','winner_contribution']]),
        '## Validation goal forecasts',markdown_table(goal_table),
        'The beta=0 rates belong to an untrained head. Use the fitting-target constant comparator below when assessing goal forecasting:',markdown_table(constants),
        '## Validation goal-rate distributions',markdown_table(ranges[ranges.split.eq('validation')][['arm','seed','side','mean','std','0','0.5','0.99','1']].rename(columns={'0':'minimum','0.5':'median','0.99':'p99','1':'maximum'})),
        '## Separate task gradients on shared encoder parameters',markdown_table(gradients),
        'All fitting fixtures and original group weights are used. Both exclusive heads and the regularizer are excluded from these gradient vectors. A zero norm gives an unavailable cosine; weighted ratios with zero outcome norm are unavailable. Detached goal gradients on the encoder are exactly zero. Measurements preserve parameters, optimizer state and existing gradient buffers. Positive cosine is not proof of transfer; negative cosine is not proof of harm.',
        '## Learning curves',
        '![Outcome curves](outcome_curves.png)', '![Goal loss curves](goal_curves.png)',
        '![Paired outcome curves over common epochs](paired_curves.png)',
        'Each line ends at its observed stopping epoch. Vertical markers identify each arm\'s selected update; no extrapolation or post-hoc common-epoch selection is used.',
        '## Per-round and per-fixture changes', '![Validation round differences](round_differences.png)',
        'Exact paired fixture losses are in `paired_fixture_losses.parquet`; per-round means are in `paired_round_losses.csv`. `paired_common_epoch_curves.csv` contains the entire intersection of observed epochs for each matched pair. `summary.csv` includes retained training and validation metrics, the non-draw proportion and the exact decomposition: three-class log loss = draw log loss + non-draw proportion × conditional winner log loss. `goal_rate_distributions.csv` includes min, 1/5/25/50/75/95/99 percentiles and max for both sides, seeds, arms and splits.',
        '## Isolation, verification and limitations',
        f"All stopping/selection receipts, retained scores, goal labels, decomposition, same-device reloads and public prediction boundaries were verified. The detached branch agrees with control across all observed outcome updates within rtol=1e-5/atol=1e-6 (errors: `{detached_max}`). There is no global gradient clipping, shared gradient normalization or adaptive task weighting; AdamW uses per-parameter moments. Beta=0 excludes the auxiliary graph from backward entirely, so the unused head receives no optimizer state or decay. Detached goal-head updates cannot scale shared gradients. The previous snapshot validation curves match the new controls within the same policy (maximum differences by seed: `{parent_errors}`).",
        f"All {len(protocol['preserved_artifacts'])} recorded previous artifact files retain their hashes. Frozen source and settings are in `frozen_source.zip` and `protocol.json`. The auxiliary rates define marginal forecasts; this experiment establishes neither goal independence nor a calibrated joint score distribution. Temporal encodings are inactive in snapshot mode.",
        '**Stopped for review.** The feature remains disabled by default. No weight sweep, historical-mode fit, deeper head, longer training run or Poisson outcome decoder has been added.'
    ])+'\n'
    (run/'report.md').write_text(text,encoding='utf-8')
    # Self-contained page except for its local static charts; no external scripts.
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables'])
    except ImportError:
        from markdown_it import MarkdownIt
        body=MarkdownIt('commonmark').enable('table').render(text)
    page='<!doctype html><html><head><meta charset="utf-8"><title>GAT goal auxiliary experiment</title><style>body{font:16px/1.55 system-ui;margin:40px auto;max-width:1250px;padding:0 24px;color:#182b39}h1,h2{color:#154864}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px 10px;border:1px solid #dce3e8;text-align:right}th{background:#eaf1f5}img{max-width:100%}code{background:#edf1f4}pre{white-space:pre-wrap}strong{color:#183d52}</style></head><body>'+body+'</body></html>'
    (run/'report.html').write_text(page,encoding='utf-8')
    verification=dict(matrix_fits=len(results),mean_validation_difference=change,improved_seeds=improvement,preliminary_screen=screen,
        detached_errors=detached_max,parent_snapshot_curve_errors=parent_errors,previous_artifacts_preserved=len(protocol['preserved_artifacts']),
        no_test_evaluation=True,independent_metric_and_stopping_verification=True)
    write_json(run/'verification.json',verification)
    files={str(p.relative_to(run)).replace('\\','/'):digest(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='COMPLETE.json'}
    write_json(run/'COMPLETE.json',dict(status='complete',**verification,files=files))
    print(json.dumps(verification),flush=True)


def make_plots(run,curves,paired,rounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'control':'#3474a2','attached':'#db7b27','detached':'#4c9874'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind,filename in (('loss','outcome_curves.png'),('goal_loss','goal_curves.png')):
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        for col,seed in enumerate((17,42,73)):
            for row,split in enumerate(('train','validation')):
                ax=axes[row,col]
                for arm in colors:
                    if (arm,seed) not in curves: continue
                    c=curves[arm,seed]
                    ax.plot(c.index,c[f'{split}_{kind}'],label=arm,color=colors[arm],linestyle='--' if arm=='detached' else '-')
                    ax.axvline(c.validation_loss.idxmin(),color=colors[arm],alpha=.3,linewidth=.8)
                ax.set(title=f'{split.capitalize()}, seed {seed}',xlabel='Full-population updates',ylabel='Outcome log loss' if kind=='loss' else 'Goal loss (constants omitted)')
                ax.grid(alpha=.15)
        axes[0,0].legend()
        fig.savefig(run/filename,dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for seed,frame in paired.groupby('seed'):
        ax.plot(frame.step,frame.validation_loss,label=f'seed {seed}')
    ax.axhline(0,color='black',linewidth=.7);ax.legend();ax.grid(alpha=.15)
    ax.set(xlabel='Common observed update',ylabel='Attached - control validation log loss',title='Matched curves: no extrapolation beyond either stopping point')
    fig.savefig(run/'paired_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    v=rounds[rounds.split.eq('validation')]
    for seed,frame in v.groupby('seed'):
        ax.plot(frame['round'],frame.log_loss,alpha=.6,label=f'seed {seed}')
    average=v.groupby('round').log_loss.mean()
    ax.plot(average.index,average,color='black',linewidth=2,label='Mean of per-seed differences')
    ax.axhline(0,color='black',linewidth=.7);ax.legend();ax.grid(alpha=.15)
    ax.set(xlabel='Validation round',ylabel='Attached - control log loss',title='Each arm evaluated at its own selected checkpoint')
    fig.savefig(run/'round_differences.png',dpi=150);plt.close(fig)
