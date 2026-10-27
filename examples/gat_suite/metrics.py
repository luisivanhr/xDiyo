"""Independent float64 scoring, draw-intercept calibration and paired resampling."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit, gammaln, xlogy
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, average_precision_score

P_COLUMNS=['p_home','p_draw','p_away']
C_COLUMNS=['cal_home','cal_draw','cal_away']
ADDITIVE=['log_loss','brier','draw_log_loss','draw_brier','winner_contribution','accuracy','predicted_draw']


def normalized(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or (p<=0).any():
        raise ValueError('Expected finite positive H/D/A probabilities.')
    np.testing.assert_allclose(p.sum(1),1,atol=3e-7,rtol=0)
    return p/p.sum(1,keepdims=True)


def draw_calibration(p,y):
    p=normalized(p); logits=logit(p[:,1]); frequency=np.mean(y==1)
    f=lambda b: expit(logits+b).mean()-frequency
    low,high=-20.,20.
    b=low if f(low)>=0 else high if f(high)<=0 else brentq(f,low,high,xtol=1e-14)
    return dict(intercept=float(b),bounds=[low,high],boundary_solution=b in (low,high),
                observed_draw=float(frequency),count=len(y),population='inner')


def calibrate(p,intercept):
    p=normalized(p); q=expit(logit(p[:,1])+intercept); r=p[:,0]/(p[:,0]+p[:,2])
    result=np.column_stack(((1-q)*r,q,(1-q)*(1-r)))
    np.testing.assert_allclose(result.sum(1),1,rtol=0,atol=3e-16)
    np.testing.assert_allclose(result[:,0]/(result[:,0]+result[:,2]),r,rtol=1e-14,atol=1e-14)
    np.testing.assert_array_equal(np.argsort(p[:,1],kind='stable'),np.argsort(result[:,1],kind='stable'))
    return result


def outcome_rows(p,y):
    p=normalized(p); y=np.asarray(y,dtype=int); draw=y==1; non=~draw
    q=p[:,1]; r=p[:,0]/(p[:,0]+p[:,2])
    loss=-np.log(p[np.arange(len(y)),y]); dl=-np.where(draw,np.log(q),np.log1p(-q))
    wc=np.where(non,-np.where(y==0,np.log(r),np.log1p(-r)),0.)
    np.testing.assert_allclose(loss,dl+wc,rtol=1e-13,atol=1e-13)
    return pd.DataFrame(dict(log_loss=loss,brier=((p-np.eye(3)[y])**2).sum(1),draw_log_loss=dl,
        draw_brier=(q-draw)**2,winner_contribution=wc,accuracy=(p.argmax(1)==y).astype(float),predicted_draw=q))


def outcome_metrics(p,y):
    y=np.asarray(y); p=normalized(p); rows=outcome_rows(p,y)
    result=rows.mean().to_dict(); non=y!=1
    result.update(count=len(y),non_draw_fraction=float(non.mean()),observed_draw=float((y==1).mean()),
        conditional_winner_log_loss=float(rows.winner_contribution[non].mean()) if non.any() else None,
        draw_auc=float(roc_auc_score(y==1,p[:,1])) if len(np.unique(y==1))==2 else None,
        draw_ap=float(average_precision_score(y==1,p[:,1])) if (y==1).any() else None,
        confusion=[[int(np.sum((y==a)&(p.argmax(1)==b))) for b in range(3)] for a in range(3)])
    return result


def count_rows(rates,y,family='poisson',alpha=0.):
    """Scoring oracle independent of torch, with exact rising-factorial sums."""
    rates,y=np.asarray(rates,float),np.asarray(y,float)
    if not np.isfinite(rates).all() or (rates<=0).any() or (y<0).any() or (y!=np.floor(y)).any():
        raise ValueError('Invalid means/counts.')
    eta=np.log(rates); constant=gammaln(y+1)-y*eta
    poisson=(rates+constant)
    def total(mean,count):
        if alpha==0: return mean
        # x is positive; the ratio avoids cancellation in the Poisson limit.
        x=alpha*mean
        ratio=np.where(x<1e-4,1-x/2+x*x/3-x**3/4+x**4/5,np.log1p(x)/x)
        out=mean*ratio+count*np.log1p(x)
        for index in np.ndindex(count.shape):
            out[index]-=np.log1p(alpha*np.arange(int(count[index]))).sum()
        return out
    if family=='poisson': marginal=poisson; joint=marginal.sum(1)
    else:
        marginal=constant+total(rates,y)
        joint=(constant.sum(1)+total(rates.sum(1),y.sum(1))) if family=='shared_gamma' else marginal.sum(1)
    return dict(goal_joint_nll=joint,goal_home_nll=marginal[:,0],goal_away_nll=marginal[:,1],
        goal_marginal_nll=marginal.sum(1),poisson_mean_nll=poisson.sum(1),
        home_absolute_error=np.abs(rates[:,0]-y[:,0]),away_absolute_error=np.abs(rates[:,1]-y[:,1]),
        home_variance=rates[:,0]+alpha*rates[:,0]**2,away_variance=rates[:,1]+alpha*rates[:,1]**2,
        predicted_covariance=alpha*rates[:,0]*rates[:,1] if family=='shared_gamma' else np.zeros(len(y)),
        home_residual=y[:,0]-rates[:,0],away_residual=y[:,1]-rates[:,1])


def prediction_table(frame,p,rates,goals,calibrator,*,family='poisson',alpha=0.):
    output=frame.copy().reset_index(drop=True)
    output['y']=np.where(goals[:,0]>goals[:,1],0,np.where(goals[:,0]==goals[:,1],1,2))
    p=normalized(p); q=calibrate(p,calibrator['intercept'])
    output[P_COLUMNS]=p; output[C_COLUMNS]=q
    for view,probs in [('raw',p),('calibrated',q)]:
        for key,value in outcome_rows(probs,output.y).items(): output[f'{view}_{key}']=value.to_numpy()
    if rates is not None:
        output[['lambda_home','lambda_away']]=rates; output[['goals_home','goals_away']]=goals
        for key,value in count_rows(rates,goals,family,alpha).items(): output[key]=value
        output['alpha']=alpha; output['kappa']=np.inf if alpha==0 else 1/alpha
        if alpha>0:
            if family=='shared_gamma':
                output['retrospective_shock_shape']=1/alpha+goals.sum(1)
                output['retrospective_shock_rate']=1/alpha+rates.sum(1)
                output['retrospective_shock_mean']=(1/alpha+goals.sum(1))/(1/alpha+rates.sum(1))
                output['retrospective_shock_variance']=(1/alpha+goals.sum(1))/(1/alpha+rates.sum(1))**2
            else:
                for j,side in enumerate(('home','away')):
                    output[f'retrospective_shock_{side}_shape']=1/alpha+goals[:,j]
                    output[f'retrospective_shock_{side}_rate']=1/alpha+rates[:,j]
                    output[f'retrospective_shock_{side}_variance']=(1/alpha+goals[:,j])/(1/alpha+rates[:,j])**2
                    output[f'retrospective_shock_{side}']=(1/alpha+goals[:,j])/(1/alpha+rates[:,j])
    return output


def table_metrics(table):
    result={view:outcome_metrics(table[cols].to_numpy(),table.y) for view,cols in [('raw',P_COLUMNS),('calibrated',C_COLUMNS)]}
    if 'goal_joint_nll' in table:
        names=['goal_joint_nll','goal_marginal_nll','goal_home_nll','goal_away_nll','poisson_mean_nll','home_absolute_error','away_absolute_error',
               'lambda_home','lambda_away','goals_home','goals_away','home_variance','away_variance','predicted_covariance']
        result['goals']={k:float(table[k].mean()) for k in names}
        result['goals']['residual_covariance']=float(np.cov(table.home_residual,table.away_residual,ddof=0)[0,1])
    return result


def rank_bootstrap(p,y,weights):
    """Weighted rank metrics under sampled fixture multiplicities, including ties."""
    order=np.argsort(-np.asarray(p),kind='stable'); score=np.asarray(p)[order]; truth=np.asarray(y,dtype=float)[order]
    ends=np.r_[np.flatnonzero(np.diff(score)!=0),len(score)-1]
    aucs=[]; aps=[]
    for start in range(0,len(weights),500):
        w=weights[start:start+500,order]
        pos=np.cumsum(w*truth,axis=1)[:,ends]; neg=np.cumsum(w*(1-truth),axis=1)[:,ends]
        dp=np.diff(np.column_stack((np.zeros(len(w)),pos)),axis=1)
        # Higher ranks accumulate first; favorable pairs are positives before negatives.
        dn=np.diff(np.column_stack((np.zeros(len(w)),neg)),axis=1)
        favorable=(dn*(pos-dp/2)).sum(1)
        aucs.extend(favorable/(pos[:,-1]*neg[:,-1]))
        precision=np.divide(pos,pos+neg,out=np.zeros_like(pos),where=(pos+neg)>0)
        aps.extend((dp*precision).sum(1)/pos[:,-1])
    return np.asarray(aucs),np.asarray(aps)
