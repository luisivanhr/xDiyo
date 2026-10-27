"""Recorded numerical repair of diagnostic decomposition, without fitting changes.

The original fitting sources remain byte-for-byte frozen. Only the NumPy score
decomposition is replaced at the recovery entry point. Primary CE is bitwise
unchanged; conditional winner logs avoid subtracting a ratio close to one.
"""
import numpy as np
import pandas as pd
from gat_suite import metrics


def outcome_rows(p,y):
    p=metrics.normalized(p);y=np.asarray(y,dtype=int);draw=y==1;non=~draw
    chosen=p[np.arange(len(y)),y]
    nondraw=p[:,0]+p[:,2]
    loss=-np.log(chosen)
    draw_loss=-np.log(np.where(draw,p[:,1],nondraw))
    winner=np.zeros(len(y))
    winner[non]=-np.log(chosen[non]/nondraw[non])
    np.testing.assert_allclose(loss,draw_loss+winner,rtol=1e-13,atol=1e-13)
    return pd.DataFrame(dict(log_loss=loss,brier=((p-np.eye(3)[y])**2).sum(1),
        draw_log_loss=draw_loss,draw_brier=(p[:,1]-draw)**2,winner_contribution=winner,
        accuracy=(p.argmax(1)==y).astype(float),predicted_draw=p[:,1]))


def install():
    metrics.outcome_rows=outcome_rows
