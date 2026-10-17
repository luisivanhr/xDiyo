"""Compare disposable receipts/states; never infer training interchangeability."""
from pathlib import Path
import argparse
import json
import statistics
import torch


def difference(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    delta = b-a
    nonzero = a != 0
    return dict(max_absolute=float(delta.abs().max()) if len(delta) else 0.,
        relative_l2=float(delta.norm()/a.norm().clamp_min(1e-300)),
        max_relative_nonzero=float((delta[nonzero].abs()/a[nonzero].abs()).max()) if nonzero.any() else 0.,
        zero_reference_changed=int(((a == 0) & (b != 0)).sum()),
        near_zero_count=int((a.abs() <= 1e-7).sum()),
        near_zero_sign_flips=int(((a.abs() <= 1e-7) & (a*b < 0)).sum()))


def tensor_map(a, b):
    assert a.keys() == b.keys()
    per_tensor = {key: difference(a[key], b[key]) for key in a}
    result = difference(torch.cat([v.flatten() for v in a.values()]),
                        torch.cat([v.flatten() for v in b.values()]))
    worst = max(per_tensor, key=lambda k: per_tensor[k]['max_absolute'])
    result.update(worst_tensor=worst, per_tensor=per_tensor)
    return result


def times(receipt):
    phases = {
        'update_including_validation': [sum(s['seconds'] for s in row['steps']) for row in receipt['records']],
        'training_backward': [sum(p['seconds'] for p in row['loss_phases'] if p['backward']) for row in receipt['records']],
        'validation': [sum(p['seconds'] for p in row['loss_phases'] if p['population']=='validation') for row in receipt['records']],
        'train_metric_replay': [row['train_metric_replay_seconds'] for row in receipt['records']],
        'validation_prediction_replay': [row['validation_prediction_replay_seconds'] for row in receipt['records']],
    }
    phases['cold_preparation'] = [row['seconds'] for row in receipt['cold_preparation_records']]
    return {key: dict(values=values, median=statistics.median(values), minimum=min(values), maximum=max(values),
                      stdev=statistics.stdev(values) if len(values)>1 else None) for key, values in phases.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    left = json.loads((args.baseline/'receipt.json').read_text())
    right = json.loads((args.candidate/'receipt.json').read_text())
    assert left['input_hashes'] == right['input_hashes']
    assert left['torch'] == right['torch'] and left['counts'] == right['counts']
    a = torch.load(args.baseline/'initial-state.pt', map_location='cpu', weights_only=True)
    b = torch.load(args.candidate/'initial-state.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])
    result = dict(baseline=str(args.baseline), candidate=str(args.candidate),
                  fixed_weights={key:difference(a['fixed_validation'][key],b['fixed_validation'][key]) for key in a['fixed_validation']},
                  changed_classes=int((a['fixed_validation']['probabilities'].argmax(1)!=b['fixed_validation']['probabilities'].argmax(1)).sum()),
                  initial_losses={'native':a['components'],'candidate':b['components']})
    result['loss_differences'] = {pop:{key:dict(absolute=abs(b['components'][pop][key]-value),
        relative=abs(b['components'][pop][key]-value)/max(abs(value),1e-300))
        for key,value in fields.items()} for pop,fields in a['components'].items()}
    ca = torch.load(args.baseline/'final-state-0.pt', map_location='cpu', weights_only=True)
    cb = torch.load(args.candidate/'final-state-0.pt', map_location='cpu', weights_only=True)
    result['gradients'] = tensor_map(ca['gradients'], cb['gradients'])
    result['model_state'] = tensor_map(ca['model'], cb['model'])
    # Keep immutable model buffers out of the parameter relative-norm measure.
    # Full model_state above still covers every saved entry, including any
    # parameter with an absent gradient. Report coverage explicitly.
    result['parameters_with_gradients'] = tensor_map(
        {name:ca['model'][name] for name in ca['gradients']},
        {name:cb['model'][name] for name in cb['gradients']})
    result['parameter_coverage'] = dict(
        elements_with_gradients=sum(value.numel() for value in ca['gradients'].values()),
        trainable_elements=left['counts']['trainable_parameters'],
        immutable_buffers_or_absent_gradients=[name for name in ca['model'] if name not in ca['gradients']])
    result['adamw'] = {field:tensor_map({str(k):v[field] for k,v in ca['optimizer']['state'].items()},
                                     {str(k):v[field] for k,v in cb['optimizer']['state'].items()})
                       for field in ('exp_avg','exp_avg_sq','step')}
    result['post_update_losses'] = {
        key: dict(native=value, candidate=cb['replay_components'][key],
                  absolute=abs(cb['replay_components'][key]-value),
                  relative=abs(cb['replay_components'][key]-value)/max(abs(value),1e-300))
        for key,value in ca['replay_components'].items()}
    def repeatability(directory, receipt, first):
        checks=[]
        for repeat in range(1,len(receipt['records'])):
            other=torch.load(directory/f'final-state-{repeat}.pt',map_location='cpu',weights_only=True)
            checks.append(dict(repeat=repeat,
                model_exact=all(torch.equal(value,other['model'][key]) for key,value in first['model'].items()),
                gradients_exact=all(torch.equal(value,other['gradients'][key]) for key,value in first['gradients'].items()),
                optimizer_exact=all(torch.equal(value,other['optimizer']['state'][key][field])
                    for key,state in first['optimizer']['state'].items() for field,value in state.items()),
                components_exact=first['replay_components']==other['replay_components']))
        return checks
    result['within_mode_repeatability'] = dict(native=repeatability(args.baseline,left,ca),
                                              candidate=repeatability(args.candidate,right,cb))
    name = result['model_state']['worst_tensor']
    i = int((ca['model'][name]-cb['model'][name]).abs().flatten().argmax())
    result['worst_parameter'] = dict(name=name,index=i,native=float(ca['model'][name].flatten()[i]),candidate=float(cb['model'][name].flatten()[i]))
    if name in ca['gradients']:
        ga, gb = (float(c['gradients'][name].flatten()[i]) for c in (ca,cb))
        group = ca['optimizer']['param_groups'][0]
        lr, eps = group['lr'],group['eps']
        result['worst_parameter'].update(native_gradient=ga,candidate_gradient=gb,
            predicted_first_update_difference=abs(lr*ga/(abs(ga)+eps)-lr*gb/(abs(gb)+eps)),learning_rate=lr,epsilon=eps)
    result['timings'] = dict(native=times(left),candidate=times(right))
    result['speed_ratios'] = {key:result['timings']['native'][key]['median']/result['timings']['candidate'][key]['median'] for key in result['timings']['native']}
    result['memory'] = {name:{key:receipt.get(key) for key in ('initialization_host_peak','cold_host_peak','sampled_peak_host','os_memory_info','cuda_memory','cold_cuda_memory')} for name,receipt in (('native',left),('candidate',right))}
    result['end_to_end_seconds'] = dict(native=left['end_to_end_seconds'],candidate=right['end_to_end_seconds'])
    result['provenance'] = dict(input_hashes_match=True,initial_weights_exact=True,
        differing_source_files=[key for key in sorted(set(left['source_sha256'])|set(right['source_sha256']))
                                if left['source_sha256'].get(key)!=right['source_sha256'].get(key)],
        harness_hashes_match=left['script_sha256']==right['script_sha256'],
        baseline_receipt=str(args.baseline/'receipt.json'),candidate_receipt=str(args.candidate/'receipt.json'))
    result['notes'] = ['Relative L2 uses the native norm; pointwise ratios exclude exact zero native entries and separately count changed zeros.',
                       'Near zero means absolute native gradient <=1e-7. No epsilon, weight, architecture or optimizer setting was changed.',
                       'Single-update differences and runtime receipts do not establish full-training predictive acceptance.']
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('changed_classes','speed_ratios','worst_parameter')},indent=2))


if __name__ == '__main__':
    main()
