"""Local, measured temporal-GAT mechanism probes; no production-model edits."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import compare_gat_temporal as experiment
from gat_temporal_diagnostics import summarize, prediction_summary, prediction_difference
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.training import run_iterations
from xdiyo_analytics.training.residency import InputCachePolicy


class Moments:
    """Exact scalar moments, bounded deterministic samples for quantiles."""
    def __init__(self, sample=32768):
        self.sample=sample
        self.count=0
        self.total=self.squares=self.flat=self.derivative=self.small_derivative=0.
        self.minimum,self.maximum=np.inf,-np.inf
        self.samples=[]

    def add(self, values):
        import torch
        a=values.detach().reshape(-1)
        if not a.numel(): return
        d=a.double()
        derivative=1-d.tanh().square()
        numbers=torch.stack((d.sum(),d.square().sum(),d.min(),d.max(),(d.abs()>3).sum(),
                             derivative.sum(),(derivative<.01).sum())).cpu().tolist()
        self.count+=a.numel()
        self.total+=numbers[0]; self.squares+=numbers[1]
        self.minimum=min(self.minimum,numbers[2]); self.maximum=max(self.maximum,numbers[3])
        self.flat+=numbers[4]; self.derivative+=numbers[5]; self.small_derivative+=numbers[6]
        indices=torch.linspace(0,len(a)-1,min(self.sample,len(a)),device=a.device).long()
        self.samples.append(a[indices].cpu().numpy())

    def result(self):
        if not self.count: return dict(count=0)
        mean=self.total/self.count
        sampled=summarize(np.concatenate(self.samples))
        return dict(count=self.count,mean=mean,std=float(np.sqrt(max(0,self.squares/self.count-mean*mean))),
            rms=float(np.sqrt(self.squares/self.count)),minimum=self.minimum,maximum=self.maximum,
            fraction_abs_gt_3=self.flat/self.count,mean_tanh_derivative=self.derivative/self.count,
            fraction_tanh_derivative_lt_0_01=self.small_derivative/self.count,
            quantile_sample_count=sampled['count'],sample_quantiles=sampled['quantiles'])


def copy_snapshot_function(expanded, snapshot, base_dim):
    """Copy every common tensor and zero only appended E/U input connections."""
    import torch
    target,source=expanded.state_dict(),snapshot.state_dict()
    if set(target)!=set(source): raise ValueError('State keys differ beyond added input columns.')
    with torch.no_grad():
        for name,value in source.items():
            if name.endswith(('.e.weight','.u.weight')):
                if value.shape[1]!=base_dim or target[name].shape[0]!=value.shape[0]:
                    raise ValueError('Unexpected edge-projection shape.')
                target[name].zero_()
                target[name][:,:base_dim].copy_(value)
            else:
                if value.shape!=target[name].shape: raise ValueError('A common parameter changed shape.')
                target[name].copy_(value)
    expanded.load_state_dict(target)


def mask_time_channels(model, base_dim):
    """Mask projection inputs without mutating immutable prepared tensors."""
    import torch
    def masked(module, args):
        x=args[0]
        return (torch.cat((x[:,:base_dim],torch.zeros_like(x[:,base_dim:])),dim=1),)
    return [projection.register_forward_pre_hook(masked) for layer in model.layers for projection in (layer.e,layer.u)]


def gradient_diagnostics(backend, base_dim):
    backend.model.train()
    backend.optimizer.zero_grad(set_to_none=True)
    loss=backend._loss(backend.train_groups,backend.train_targets,backend.train_weights,True)
    penalty=backend.config.prior_penalty*(backend.model.league.weight.square().sum()+backend.model.entry.weight.square().sum())
    penalty.backward()
    result={}
    for i,layer in enumerate(backend.model.layers):
        for name in ('e','u'):
            weights=getattr(layer,name).weight
            for label,part in (('base',slice(None,base_dim)),('time',slice(base_dim,None))):
                if weights[:,part].numel():
                    result[f'layer_{i+1}.{name}.{label}']=dict(
                        weight=summarize(weights[:,part].detach().cpu().numpy()),
                        gradient=summarize(weights.grad[:,part].detach().cpu().numpy()))
    backend.optimizer.zero_grad(set_to_none=True)
    return dict(train_cross_entropy=loss,prior_penalty=float(penalty.detach()),projections=result)


def predict_groups(backend, groups, *, reference=None, base_dim=None):
    import torch
    p=np.empty((len(groups.plan.metadata),3))
    max_logits=max_probability=0.
    backend.model.eval()
    if reference is not None: reference.eval()
    with torch.no_grad():
        for i in range(len(groups)):
            with groups.lease(i) as group:
                positions,data,home,away,fixture=group
                logits=backend.model(data,home,away,fixture)
                probability=logits.softmax(-1)
                p[positions]=probability.cpu().numpy()
                if reference is not None:
                    original=reference(dict(data,x=data['x'][:,:base_dim]),home,away,fixture)
                    torch.testing.assert_close(logits,original,atol=1e-6,rtol=1e-5)
                    max_logits=max(max_logits,float((logits-original).abs().max()))
                    max_probability=max(max_probability,float((probability-original.softmax(-1)).abs().max()))
    return p,dict(maximum_logit_difference=max_logits,maximum_probability_difference=max_probability)


class ProjectionCapture:
    """Observe native projection outputs and native attention on root queries."""
    def __init__(self, model, base_dim):
        self.model,self.base_dim=model,base_dim
        self.stats={}
        self.buffers={}
        self.handles=[]
        self.decomposition_error=0.
        self.sender_states={}
        for i,layer in enumerate(model.layers):
            self.handles.append(layer.register_forward_pre_hook(self.before(i),with_kwargs=True))
            for name in ('q','k','e','v','u'):
                self.handles.append(getattr(layer,name).register_forward_hook(self.projection(i,name)))
            self.handles.append(layer.register_forward_hook(self.after(i)))

    def add(self, name, values):
        self.stats.setdefault(name,Moments()).add(values)

    def before(self, i):
        def hook(layer,args,kwargs):
            import torch
            h,x,receiver,sender,log_weight,gate=args
            index=(self.data['temporal_layers'][i]['root_edge_index'] if self.model.temporal_mode=='historical'
                   else torch.arange(len(receiver),device=x.device))
            self.buffers[i]=dict(index=index,x=x[index])
            ss=kwargs.get('sender_state')
            if self.keep_senders:
                self.sender_states[i]=(ss[index] if ss is not None else h[sender[index]]).detach().cpu().numpy()
        return hook

    def projection(self,i,name):
        def hook(module,args,output):
            self.buffers[i][name]=output.detach()[self.buffers[i]['index']]
        return hook

    def after(self,i):
        def hook(layer,args,output):
            import torch
            b=self.buffers.pop(i)
            x=b['x']
            base_e=torch.nn.functional.linear(x[:,:self.base_dim],layer.e.weight[:,:self.base_dim])
            base_u=torch.nn.functional.linear(x[:,:self.base_dim],layer.u.weight[:,:self.base_dim])
            time_e=torch.nn.functional.linear(x[:,self.base_dim:],layer.e.weight[:,self.base_dim:])
            time_u=torch.nn.functional.linear(x[:,self.base_dim:],layer.u.weight[:,self.base_dim:])
            self.decomposition_error=max(self.decomposition_error,float((b['e']-base_e-time_e).abs().max()),
                                         float((b['u']-base_u-time_u).abs().max()))
            prefix=f'layer_{i+1}.'
            values={'attention.receiver':b['q'],'attention.sender':b['k'],
                'attention.edge_base':base_e,'attention.edge_time':time_e,
                'attention.pre_leaky':b['q']+b['k']+b['e'],
                'value.sender':b['v'],'value.edge_base':base_u,'value.edge_time':time_u,
                'value.pre_tanh':b['v']+b['u'],'value.pre_tanh_without_time':b['v']+base_u}
            for name,value in values.items(): self.add(prefix+name,value)
            if not isinstance(output,tuple): raise ValueError('Probe must request native attention.')
            alpha=output[1][b['index']]
            receiver=self.data['receiver']
            n=len(self.data['league'])
            entropy=alpha.new_zeros((n,layer.heads))
            entropy.index_add_(0,receiver,-alpha*alpha.clamp_min(1e-30).log())
            degree=torch.bincount(receiver,minlength=n)
            present=degree>0
            self.add(prefix+'attention.entropy',entropy[present])
            self.add(prefix+'attention.effective_matches',entropy[present].exp())
            self.add(prefix+'attention.effective_fraction',entropy[present].exp()/degree[present,None])
            self.add(prefix+'gate',self.data['gate'][present])
        return hook

    def encode(self,data,home,away,fixture,*,keep_senders=False):
        import torch
        self.data,self.keep_senders=data,keep_senders
        self.sender_states={}
        h,_=self.model.encode(data,return_attention=True)
        a,b=h[home],h[away]
        logits=self.model.head(torch.cat((a,b,a-b,a*b,fixture),dim=1))
        return logits,self.sender_states

    def close(self):
        for handle in self.handles: handle.remove()

    def result(self):
        return dict(projections={k:v.result() for k,v in self.stats.items()},
                    component_decomposition_max_absolute_error=self.decomposition_error)


def root_features(backend, base_dim):
    columns=backend.preprocessor.temporal_columns
    output={}
    for split,groups in [('train',backend.train_groups),('validation',backend.validation_groups)]:
        rows=[]
        for i in range(len(groups)):
            with groups.lease(i) as group:
                rows.append(group[1]['x'][:,base_dim:].detach().cpu().numpy())
        a=np.concatenate(rows)
        output[split]={name:summarize(a[:,j]) for j,name in enumerate(columns)}
    return output


def fixed_sender_comparison(backend, protocol, parent, frames):
    """Same weights/inputs, changing only TemporalNetwork.temporal_mode."""
    import torch
    base=Path(protocol['base'])
    base_dim=backend.preprocessor.edge_dim-len(backend.preprocessor.temporal_columns)
    results={}
    for source_mode in ('time_features','historical'):
        folder=base/'fit'/f"{source_mode}-{protocol['mechanism_seed']}"/'model'
        meta=json.loads((folder/'graph.json').read_text())
        assert json.loads(json.dumps(backend.preprocessor.state()))==meta['preprocessing']
        backend.model.load_state_dict(torch.load(folder/'weights.pt',weights_only=True,map_location=backend.device))
        backend.model.eval()
        states={i:{'difference':Moments(),'current':Moments(),'historical':Moments(),'cosine':Moments()} for i in range(len(backend.model.layers))}
        predictions={mode:np.empty((len(frames['validation']),3)) for mode in ('time_features','historical')}
        # A lightweight sender-only hook avoids repeating all projection statistics.
        capture={}
        handles=[]
        current_data=None
        def sender_hook(i):
            def hook(layer,args,kwargs):
                h,x,receiver,sender,*_=args
                ix=current_data['temporal_layers'][i]['root_edge_index'] if backend.model.temporal_mode=='historical' else torch.arange(len(receiver),device=h.device)
                ss=kwargs.get('sender_state')
                capture[i]=(ss[ix] if ss is not None else h[sender[ix]]).detach().clone()
            return hook
        for i,layer in enumerate(backend.model.layers):
            handles.append(layer.register_forward_pre_hook(sender_hook(i),with_kwargs=True))
        with torch.no_grad():
            for j in range(len(backend.validation_groups)):
                with backend.validation_groups.lease(j) as group:
                    positions,current_data,home,away,fixture=group
                    paired={}
                    for mode in ('time_features','historical'):
                        backend.model.temporal_mode=mode
                        logits=backend.model(current_data,home,away,fixture)
                        predictions[mode][positions]=logits.softmax(-1).cpu().numpy()
                        paired[mode]=dict(capture)
                    for i,stats in states.items():
                        a,b=paired['time_features'][i],paired['historical'][i]
                        stats['current'].add(a.norm(dim=1)); stats['historical'].add(b.norm(dim=1))
                        stats['difference'].add((a-b).norm(dim=1))
                        stats['cosine'].add(torch.nn.functional.cosine_similarity(a,b,dim=1,eps=1e-12))
                    del paired
        for handle in handles: handle.remove()
        y=experiment.make_context(frames['validation']).y.result.to_numpy()
        results[source_mode]=dict(frozen_weights_sha256=experiment.digest(folder/'weights.pt'),
            sender_states={f'layer_{i+1}':{k:v.result() for k,v in stat.items()} for i,stat in states.items()},
            current=prediction_summary(y,predictions['time_features']),historical=prediction_summary(y,predictions['historical']),
            difference=prediction_difference(y,predictions['historical'],predictions['time_features']))
    backend.model.temporal_mode='historical'
    return results


def probe(args,protocol,parent,store,frames):
    import torch
    folder=args.run/'probe'/args.mode
    folder.mkdir(parents=True,exist_ok=False)
    cfg=replace(experiment.TemporalGraphConfig(**parent['config']),temporal_mode=args.mode)
    context=experiment.make_context(frames['train'])
    context.validation=experiment.make_context(frames['validation'])
    cache=InputCachePolicy(protocol['input_cache_bytes'],protocol['input_cache_headroom_bytes'])
    backend=_Backend(cfg,store,protocol['mechanism_seed'],protocol['device'],cache)
    backend.initialize(context)
    base_dim=backend.preprocessor.edge_dim-len(backend.preprocessor.temporal_columns)
    saved=Path(protocol['base'])/'fit'/f"{args.mode}-{protocol['mechanism_seed']}"/'model'
    meta=json.loads((saved/'graph.json').read_text())
    assert json.loads(json.dumps(backend.preprocessor.state()))==meta['preprocessing']
    output=dict(mode=args.mode,seed=protocol['mechanism_seed'],protocol_sha256=experiment.digest(args.run/'protocol.json'),states={})
    if args.mode=='time_features':
        experiment.write_json(args.run/'feature_distributions.json',root_features(backend,base_dim))
    initial={}
    dependency_features={}
    for stage in ('initial','selected'):
        if stage=='selected':
            backend.model.load_state_dict(torch.load(saved/'weights.pt',weights_only=True,map_location=backend.device))
            backend.refresh_version()
            assert backend.model_version==meta['model_version']
        print(json.dumps(dict(stage='probe',mode=args.mode,state=stage,phase='gradients')),flush=True)
        gradient=gradient_diagnostics(backend,base_dim)
        backend.model.eval()
        capture=ProjectionCapture(backend.model,base_dim)
        with torch.no_grad():
            for split,groups in [('train',backend.train_groups),('validation',backend.validation_groups)]:
                for index in (0,len(groups)//2,len(groups)-1):
                    with groups.lease(index) as group:
                        positions,data,home,away,fixture=group
                        capture.encode(data,home,away,fixture)
                        if args.mode=='historical' and stage=='initial':
                            for depth,batch in enumerate(data['temporal_layers']):
                                for j,name in enumerate(backend.preprocessor.temporal_columns):
                                    dependency_features.setdefault(f'layer_{depth+1}.{name}',Moments(sample=4096)).add(batch['x'][:,base_dim+j])
        capture.close()
        state=dict(gradient=gradient,activations=capture.result(),splits={})
        for split,groups in [('train',backend.train_groups),('validation',backend.validation_groups)]:
            p,_=predict_groups(backend,groups)
            y=experiment.make_context(frames[split]).y.result.to_numpy()
            state['splits'][split]=prediction_summary(y,p)
            if stage=='initial': initial[split]=p
            else: state['splits'][split]['change_from_initial']=prediction_difference(y,p,initial[split])
            f=frames[split][list(experiment.MATCH_KEYS)].copy()
            f[['p_home','p_draw','p_away']]=p
            f.to_parquet(folder/f'{stage}_{split}_predictions.parquet',index=False)
        if stage=='selected':
            np.testing.assert_allclose(state['splits']['validation']['log_loss'],meta['training_summary']['monitored_value'],atol=2e-6,rtol=0)
        output['states'][stage]=state
        print(json.dumps(dict(stage='probe_state_complete',mode=args.mode,state=stage,
            training_loss=state['splits']['train']['log_loss'],validation_loss=state['splits']['validation']['log_loss'])),flush=True)
    if dependency_features:
        output['dependency_feature_distributions']={k:v.result() for k,v in dependency_features.items()}
    if args.mode=='historical':
        output['fixed_sender_comparison']=fixed_sender_comparison(backend,protocol,parent,frames)
    experiment.write_json(folder/'result.json',output)


def matched(args,protocol,parent,store,frames):
    import torch
    folder=args.run/'matched'/args.arm
    folder.mkdir(parents=True,exist_ok=False)
    cfg=replace(experiment.TemporalGraphConfig(**parent['config']),temporal_mode='time_features')
    cache=InputCachePolicy(protocol['input_cache_bytes'],protocol['input_cache_headroom_bytes'])
    context=experiment.make_context(frames['train'])
    context.validation=experiment.make_context(frames['validation'])
    base_dim=None
    class MatchedBackend(_Backend):
        def initialize(self, context):
            nonlocal base_dim
            super().initialize(context)
            reference=_Backend(replace(cfg,temporal_mode='snapshot'),store,self.seed,self.device,InputCachePolicy())
            reference.initialize(context)
            assert self.preprocessor.state()==reference.preprocessor.state()
            base_dim=reference.preprocessor.edge_dim
            copy_snapshot_function(self.model,reference.model,base_dim)
            self.mask_handles=mask_time_channels(self.model,base_dim) if args.arm=='zero_time' else []
            self.initial_diagnostics={}
            for split,groups in [('train',self.train_groups),('validation',self.validation_groups)]:
                p,equivalence=predict_groups(self,groups,reference=reference.model,base_dim=base_dim)
                self.initial_diagnostics[split]=dict(equivalence=equivalence,**prediction_summary(experiment.make_context(frames[split]).y.result,p))
                f=frames[split][list(experiment.MATCH_KEYS)].copy()
                f[['p_home','p_draw','p_away']]=p
                f.to_parquet(folder/f'initial_{split}_predictions.parquet',index=False)
            del reference
            self.initial_diagnostics['gradients']=gradient_diagnostics(self,base_dim)
            time_grad=[v['gradient']['rms'] for k,v in self.initial_diagnostics['gradients']['projections'].items() if k.endswith('.time')]
            assert all(v==0 for v in time_grad) if args.arm=='zero_time' else all(v>0 for v in time_grad)
            self.refresh_version()
    control=experiment.training_control(protocol,protocol['seed'])
    progress=[]
    start=time.perf_counter()
    tick=start
    def observer(event):
        nonlocal tick
        if event.final: return
        torch.cuda.synchronize(protocol['device'])
        now=time.perf_counter()
        progress.append(dict(step=event.step,seconds_since_previous=now-tick,**event.metrics))
        tick=now
        experiment.write_json(folder/'progress.json',progress)
    context.observer=observer
    backend,history,summary=run_iterations(lambda seed:MatchedBackend(cfg,store,seed,protocol['device'],cache),context,control)
    elapsed=time.perf_counter()-start
    experiment.validate_training_progress(progress,summary,protocol)
    if args.arm=='zero_time':
        assert all(torch.count_nonzero(projection.weight[:,base_dim:])==0 for layer in backend.model.layers for projection in (layer.e,layer.u))
    result=dict(arm=args.arm,seed=protocol['seed'],protocol_sha256=experiment.digest(args.run/'protocol.json'),
        training_summary=summary,initial=backend.initial_diagnostics,elapsed_including_initialization_checks_seconds=elapsed,
        selected_gradients=gradient_diagnostics(backend,base_dim),selected={})
    for split,groups in [('train',backend.train_groups),('validation',backend.validation_groups)]:
        p,_=predict_groups(backend,groups)
        result['selected'][split]=prediction_summary(experiment.make_context(frames[split]).y.result,p)
        f=frames[split][list(experiment.MATCH_KEYS)].copy()
        f[['p_home','p_draw','p_away']]=p
        f.to_parquet(folder/f'selected_{split}_predictions.parquet',index=False)
    model=experiment.TemporalGraphAdapter(cfg,store,control=control,device=protocol['device'],inner_threads=1)
    model.configure_input_cache(cache.max_bytes,cache.headroom_bytes)
    model.backend_,model.training_history_,model.training_summary_=backend,history,summary
    experiment.TemporalGraphSerializer().save(model,folder/'model')
    history.to_csv(folder/'training.csv',index=False)
    experiment.write_json(folder/'result.json',result)
    print(json.dumps(dict(stage='matched_complete',arm=args.arm,epochs=len(progress),selected=summary['retained_step'],
        validation=summary['monitored_value'],reason=summary['termination_reason'])),flush=True)
