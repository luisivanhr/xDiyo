"""Native, data-only SGD checkpoint adapter used only on synthetic fixtures."""
import json
import numpy as np
import pandas as pd

class NativeFactory:
    def __init__(self,events,*,gate=None,steps=6,seed=23):
        self.events=events;self.gate=gate;self.steps=steps;self.seed=seed
    def cache_key(self):
        # Observability and injected interruption do not change the algorithm.
        return {'adapter':'native SGD v1','steps':self.steps,'seed':self.seed,'rate':.01,'momentum':.4}
    def __call__(self):return NativeSGD(self)

class NativeSGD:
    def __init__(self,owner):self.owner=owner
    def fit(self,context):raise AssertionError('capable adapter should use fit_resumable')
    def fit_resumable(self,context,*,control,checkpoint,save_checkpoint):
        assert control is None
        self.columns=tuple(context.X);self.targets=tuple(context.y);X=context.X.to_numpy();y=context.y.iloc[:,0].to_numpy()
        rng=np.random.default_rng(self.owner.seed)
        if checkpoint is None:
            self.mean=X.mean(axis=0);self.scale=X.std(axis=0);self.scale[self.scale==0]=1
            self.weights=np.zeros(X.shape[1]+1);velocity=np.zeros_like(self.weights);cursor=0;history=[]
        else:
            with np.load(checkpoint/'state.npz',allow_pickle=False) as state:
                self.mean=state['mean'];self.scale=state['scale'];self.weights=state['weights'];velocity=state['velocity']
            metadata=json.loads((checkpoint/'metadata.json').read_text())
            cursor=metadata['cursor'];history=metadata['history'];rng.bit_generator.state=metadata['rng']
            assert metadata['columns']==list(self.columns) and metadata['targets']==list(self.targets)
        self.owner.events.append(('start',cursor))
        design=np.c_[(X-self.mean)/self.scale,np.ones(len(X))]
        for step in range(cursor+1,self.owner.steps+1):
            for index in rng.permutation(len(X)):
                gradient=2*(design[index]@self.weights-y[index])*design[index]
                velocity=.4*velocity+gradient;self.weights-=.01*velocity
            loss=float(np.mean((design@self.weights-y)**2));history.append({'step':step,'loss':loss})
            self.owner.events.append(('step',step))
            def writer(directory):
                np.savez(directory/'state.npz',weights=self.weights,velocity=velocity,mean=self.mean,scale=self.scale)
                (directory/'metadata.json').write_text(json.dumps({'cursor':step,'history':history,
                    'rng':rng.bit_generator.state,'columns':list(self.columns),'targets':list(self.targets)}))
            save_checkpoint(writer)
            if step==3 and self.owner.gate is not None and not self.owner.gate.exists():
                raise RuntimeError('injected interruption after native checkpoint')
        self.training_history_=pd.DataFrame(history);self.training_summary_={'steps':self.owner.steps}
    def predict(self,context):
        design=np.c_[(context.X.to_numpy()-self.mean)/self.scale,np.ones(len(context.X))]
        return {'predict':pd.DataFrame(design@self.weights,index=context.X.index,columns=self.targets)}
