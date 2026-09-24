"""Data-only native prediction serializer, separate from training checkpoints."""
import json
import numpy as np
import pandas as pd

class NativeLinear:
    def fit(self,context):
        self.columns=tuple(context.X);self.targets=tuple(context.y)
        self.mean=context.X.mean().to_numpy();self.scale=context.X.std(ddof=0).to_numpy(copy=True);self.scale[self.scale==0]=1
        design=np.c_[(context.X.to_numpy()-self.mean)/self.scale,np.ones(len(context.X))]
        self.weights=np.linalg.lstsq(design,context.y.to_numpy(),rcond=None)[0]
    def predict(self,context):
        design=np.c_[(context.X.to_numpy()-self.mean)/self.scale,np.ones(len(context.X))]
        return {'predict':pd.DataFrame(design@self.weights,index=context.X.index,columns=self.targets)}

class NativeSerializer:
    format_id='synthetic.native-linear.v1'
    def save(self,adapter,directory):
        np.savez(directory/'state.npz',mean=adapter.mean,scale=adapter.scale,weights=adapter.weights)
        (directory/'names.json').write_text(json.dumps({'columns':adapter.columns,'targets':adapter.targets}))
    def load(self,directory):
        model=NativeLinear()
        with np.load(directory/'state.npz',allow_pickle=False) as state:
            model.mean=state['mean'];model.scale=state['scale'];model.weights=state['weights']
        names=json.loads((directory/'names.json').read_text());model.columns=tuple(names['columns']);model.targets=tuple(names['targets'])
        return model
