"""Frozen diagnostics of the patience-15 temporal-GAT experiment.

Saved test predictions are analyzed without refitting. New matched-function
training and mechanism probes use training/validation only, in isolated workers.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import compare_gat_temporal as experiment

ROOT = experiment.ROOT
FILES = ('examples/gat_temporal_diagnostics.py', 'examples/gat_temporal_mechanisms.py')
QUANTILES = [0., .01, .05, .25, .5, .75, .95, .99, 1.]


def summarize(values):
    a = np.asarray(values,dtype=float).reshape(-1)
    if not len(a):
        return dict(count=0)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite diagnostic values.')
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),rms=float(np.sqrt(np.mean(a*a))),
                quantiles={f'{q:g}':float(v) for q,v in zip(QUANTILES,np.quantile(a,QUANTILES))})


def probabilities(p):
    p = np.asarray(p,dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=2e-6,rtol=0):
        raise ValueError('Expected finite ordered H/D/A probabilities summing to one.')
    return p/p.sum(1,keepdims=True)


def score_rows(y, p):
    p = probabilities(p)
    y = np.asarray(y)
    indices = np.array([experiment.CLASS_ORDER.index(x) for x in y])
    draw = y == 0
    q = p[:,1]
    if (p <= 0).any():
        raise ValueError('The loss decomposition requires strictly positive probabilities.')
    three = -np.log(p[np.arange(len(p)),indices])
    binary = -np.where(draw,np.log(q),np.log1p(-q))
    conditional = p[:,0]/(p[:,0]+p[:,2])
    winner = np.zeros(len(p))
    winner[~draw] = -np.where(y[~draw]==1,np.log(conditional[~draw]),np.log1p(-conditional[~draw]))
    np.testing.assert_allclose(three,binary+winner,atol=2e-13,rtol=2e-13)
    truth = np.eye(3)[indices]
    return dict(log_loss=three,brier=np.square(p-truth).sum(1),draw_log_loss=binary,
                draw_brier=np.square(q-draw),winner_contribution=winner)


def prediction_summary(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    p = probabilities(p)
    rows = score_rows(y,p)
    non_draw = np.asarray(y) != 0
    entropy = -(p*np.log(p)).sum(1)
    result = {name:float(values.mean()) for name,values in rows.items()}
    result.update(count=len(p),non_draw_count=int(non_draw.sum()),
        conditional_winner_log_loss=float(rows['winner_contribution'][non_draw].mean()),
        decomposition_error=float(np.max(np.abs(rows['log_loss']-rows['draw_log_loss']-rows['winner_contribution']))),
        draw_auc=float(roc_auc_score(np.asarray(y)==0,p[:,1])) if len(np.unique(np.asarray(y)==0))==2 else None,
        draw_average_precision=float(average_precision_score(np.asarray(y)==0,p[:,1])) if len(np.unique(np.asarray(y)==0))==2 else None,
        probabilities={k:summarize(p[:,j]) for j,k in enumerate(('home','draw','away'))},
        entropy=summarize(entropy),kl_from_uniform=summarize(np.log(3)-entropy),
        max_probability=summarize(p.max(1)),tv_from_uniform=summarize(.5*np.abs(p-1/3).sum(1)))
    return result


def prediction_difference(y, candidate, reference):
    a,b = probabilities(candidate),probabilities(reference)
    difference = a-b
    losses = score_rows(y,a)['log_loss']-score_rows(y,b)['log_loss']
    return dict(total_variation=summarize(.5*np.abs(difference).sum(1)),
        max_class_change=summarize(np.abs(difference).max(1)),
        draw_change=summarize(difference[:,1]),probability_rmse=float(np.sqrt(np.mean(difference**2))),
        argmax_disagreement=float(np.mean(a.argmax(1)!=b.argmax(1))),
        candidate_improves_count=int((losses < -1e-12).sum()),candidate_worsens_count=int((losses > 1e-12).sum()),
        mean_log_loss_change=float(losses.mean()),
        improvement_mass_per_fixture=float(np.minimum(losses,0).mean()),
        deterioration_mass_per_fixture=float(np.maximum(losses,0).mean()))


def diagnostic_sources():
    return {**experiment.source_hashes(),**{name:experiment.digest(ROOT/name) for name in FILES}}


def prepare(args):
    base = args.base.resolve()
    parent,_,_ = experiment.load(base)
    completion = json.loads((base/'COMPLETE.json').read_text())
    for name,value in completion['files'].items():
        if experiment.digest(base/name) != value:
            raise ValueError('Parent completion evidence changed.')
    if args.run.exists() or args.run.resolve().is_relative_to(Path(parent['data_root']).resolve()) or args.run.resolve().is_relative_to(base):
        raise ValueError('Use a new diagnostic directory outside the parent run and source data.')
    evidence = ['protocol.json','COMPLETE.json','results.json','rolling_logistic_predictions.parquet']
    evidence += [f'fit/{mode}-{seed}/{name}' for mode in parent['modes'] for seed in parent['seeds']
        for name in ('result.json','progress.json','predictions.parquet','model/graph.json','model/weights.pt')]
    protocol = dict(schema=1,created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        base=str(base),parent_evidence={name:experiment.digest(base/name) for name in evidence},
        source_hashes=diagnostic_sources(),recommendations_sha256=experiment.digest(args.recommendations),
        seed=17,max_steps=200,early_stopping=dict(patience=15,min_delta=0.),
        matched_arms=['zero_time','learn_time'],matched_evaluation_splits=['train','validation'],
        mechanism_seed=17,mechanism_groups='first, middle and last issuance in train and validation',
        fixed_sender_comparison='Both selected time_features and historical parameter states on all validation fixtures',
        feature_distribution='Every root edge occurrence in all training and validation issuances',
        activation_scope='All root-query incoming edges in the six selected issuances; lower dependency queries excluded',
        activation_quantiles='Deterministic sample of at most 32768 scalar coordinates per projection per issuance; moments exact over root edges',
        equivalence_tolerance=dict(atol=1e-6,rtol=1e-5),
        bootstrap=parent['bootstrap'],device=parent['device'],input_cache_bytes=parent['input_cache_bytes'],
        input_cache_headroom_bytes=parent['input_cache_headroom_bytes'],worker_rss_limit_gib=parent['worker_rss_limit_gib'],
        cuda_allocator_config=parent['cuda_allocator_config'],cuda_memory_fraction=parent['cuda_memory_fraction'])
    args.run.mkdir(parents=True)
    shutil.copyfile(args.recommendations,args.run/'recommendations.txt')
    experiment.write_json(args.run/'protocol.json',protocol)
    print(json.dumps(dict(stage='prepared',run=str(args.run),new_training='seed 17, train/validation only, max 200 and patience 15')))


def load(run):
    protocol = json.loads((run/'protocol.json').read_text())
    if diagnostic_sources() != protocol['source_hashes']:
        raise ValueError('Diagnostic source changed after freezing.')
    base = Path(protocol['base'])
    for name,expected in protocol['parent_evidence'].items():
        if experiment.digest(base/name) != expected:
            raise ValueError(f'Parent evidence changed: {name}')
    parent,store,frames = experiment.load(base)
    return protocol,parent,store,frames


def saved_scores(args):
    protocol,parent,_,frames = load(args.run)
    base = Path(protocol['base'])
    y = experiment.make_context(frames['test']).y.result.to_numpy()
    training_y = experiment.make_context(frames['train']).y.result.to_numpy()
    frequencies = np.array([(training_y==label).mean() for label in experiment.CLASS_ORDER])
    model_predictions, per_seed, histories = {}, {}, []
    normalization_error = 0.
    fixture_rows = []
    for mode in parent['modes']:
        model_predictions[mode],per_seed[mode] = [],[]
        for seed in parent['seeds']:
            folder = base/'fit'/f'{mode}-{seed}'
            f = pd.read_parquet(folder/'predictions.parquet')
            assert experiment.frame_hash(f)==parent['populations']['test']['identity_sha256']
            np.testing.assert_array_equal(f.truth,y)
            raw = f[['p_home','p_draw','p_away']].to_numpy()
            normalization_error = max(normalization_error,float(np.abs(raw.sum(1)-1).max()))
            p = probabilities(raw)
            model_predictions[mode].append(p)
            per_seed[mode].append(dict(seed=seed,**prediction_summary(y,p)))
            result = json.loads((folder/'result.json').read_text())
            progress = json.loads((folder/'progress.json').read_text())
            best = result['training_summary']['retained_step']
            at = {r['step']:r for r in progress}
            histories.append(dict(mode=mode,seed=seed,epochs=len(progress),selected_epoch=best,
                training_loss_is_pre_update=True,
                train_at_1=at[1]['train_loss'],train_at_20=at[20]['train_loss'],
                train_at_selected_pre_update=at[best]['train_loss'],train_at_stop_pre_update=progress[-1]['train_loss'],
                validation_at_1=at[1]['validation_loss'],validation_at_20=at[20]['validation_loss'],
                validation_best=result['training_summary']['monitored_value'],validation_at_stop=progress[-1]['validation_loss'],
                last_5_train_slope=float(np.polyfit(np.arange(5),[r['train_loss'] for r in progress[-5:]],1)[0]),
                last_5_validation_slope=float(np.polyfit(np.arange(5),[r['validation_loss'] for r in progress[-5:]],1)[0])))
    logistic = pd.read_parquet(base/'rolling_logistic_predictions.parquet').to_numpy()
    previous = json.loads((base/'results.json').read_text())
    np.testing.assert_allclose(experiment.metrics(y,logistic)['log_loss'],previous['rolling_logistic']['log_loss'],atol=1e-12,rtol=0)
    model_predictions.update(rolling_logistic=[probabilities(logistic)],
        training_frequency=[np.broadcast_to(frequencies,(len(y),3)).copy()],uniform=[np.full((len(y),3),1/3)])
    for mode in ('rolling_logistic','training_frequency','uniform'):
        per_seed[mode] = [prediction_summary(y,model_predictions[mode][0])]
    metrics = ('log_loss','brier','draw_log_loss','draw_brier','winner_contribution')
    means = {mode:{name:float(np.mean([r[name] for r in values])) for name in (*metrics,'conditional_winner_log_loss','draw_auc','draw_average_precision')}
             for mode,values in per_seed.items()}
    contrasts = {}
    pairs = [('time_features','snapshot'),('historical','snapshot'),('historical','time_features')]
    pairs += [(mode,baseline) for mode in parent['modes'] for baseline in ('rolling_logistic','training_frequency')]
    losses = {mode:{name:np.mean([score_rows(y,p)[name] for p in predictions],axis=0) for name in metrics}
              for mode,predictions in model_predictions.items()}
    for candidate,reference in pairs:
        contrasts[f'{candidate}_minus_{reference}'] = {name:experiment.paired_blocks(
            losses[candidate][name]-losses[reference][name],frames['test']['round'].tolist(),**protocol['bootstrap']) for name in metrics}
    differences = {}
    for candidate,reference in pairs[:3]:
        key=f'{candidate}_minus_{reference}'
        differences[key] = []
        for i,seed in enumerate(parent['seeds']):
            a,b = model_predictions[candidate][i],model_predictions[reference][i]
            differences[key].append(dict(seed=seed,**prediction_difference(y,a,b)))
            rows = frames['test'][list(experiment.MATCH_KEYS)].copy()
            rows['comparison'],rows['seed']=key,seed
            rows['total_variation']=.5*np.abs(a-b).sum(1)
            rows['draw_change']=a[:,1]-b[:,1]
            rows['log_loss_change']=score_rows(y,a)['log_loss']-score_rows(y,b)['log_loss']
            fixture_rows.append(rows)
    pd.concat(fixture_rows,ignore_index=True).to_parquet(args.run/'fixture_differences.parquet',index=False)
    pd.DataFrame(histories).to_csv(args.run/'learning_curve_summary.csv',index=False)
    pd.DataFrame(means).T.to_csv(args.run/'score_summary.csv')
    experiment.write_json(args.run/'scores.json',dict(protocol_sha256=experiment.digest(args.run/'protocol.json'),
        mean_scores=means,per_seed=per_seed,paired=contrasts,prediction_differences=differences,
        training_class_frequencies=dict(zip(('home','draw','away'),frequencies.tolist())),
        test_class_frequencies={k:float(np.mean(y==label)) for k,label in zip(('home','draw','away'),experiment.CLASS_ORDER)},
        maximum_original_row_sum_error=normalization_error,
        probability_convention='Float64 row normalization for exact decomposition; average per-seed scores, never ensemble probabilities.',
        learning_curves=histories))
    print(json.dumps(dict(means=means,class_frequencies=frequencies.tolist(),max_row_sum_error=normalization_error)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','scores','probe','matched'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=ROOT/'experiments/gat_temporal_cuda_patience15_20261025')
    parser.add_argument('--recommendations',type=Path)
    parser.add_argument('--mode',choices=experiment.MODES)
    parser.add_argument('--arm',choices=['zero_time','learn_time'])
    args=parser.parse_args()
    args.run=args.run.resolve()
    if args.command=='prepare':
        prepare(args)
    elif args.command=='scores':
        saved_scores(args)
    elif args.mode or args.arm:
        protocol,parent,store,frames=load(args.run)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        os.environ['PYTORCH_ALLOC_CONF']=protocol['cuda_allocator_config']
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(protocol['cuda_memory_fraction'],protocol['device'])
        from xdiyo_analytics.graphs.execution import deterministic_execution
        from gat_temporal_mechanisms import probe,matched
        with deterministic_execution(experiment.TemporalGraphConfig(**experiment.migrate_gap_config(parent['config'])),protocol['device']), experiment.MemoryProbe(protocol['worker_rss_limit_gib']):
            (probe if args.command=='probe' else matched)(args,protocol,parent,store,frames)
    else:
        for value in (experiment.MODES if args.command=='probe' else ['zero_time','learn_time']):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),args.command,'--run',str(args.run),
                '--mode' if args.command=='probe' else '--arm',value],check=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


if __name__=='__main__':
    main()
