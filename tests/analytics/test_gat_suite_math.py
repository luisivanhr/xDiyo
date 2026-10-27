"""Independent likelihood, distribution, gradient, calibration and reduction checks."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from scipy.special import gammaln
from scipy.stats import nbinom, binom

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from gat_suite.metrics import draw_calibration, calibrate, rank_bootstrap, count_rows
from xdiyo_analytics.graphs import GoalAuxiliary
from xdiyo_analytics.graphs.goal_network import AttackDefenceHead
from xdiyo_analytics.graphs.goals import count_goal_loss


@pytest.mark.parametrize('family',['poisson','independent_nb','shared_gamma'])
def test_independent_likelihood_oracle_and_gradcheck(family):
    eta=torch.tensor([[.4,-.3],[1.,.2],[-.1,.7]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[0.,0.],[3.,1.],[1.,4.]],dtype=torch.float64)
    raw=torch.tensor(-1.7,dtype=torch.float64,requires_grad=True)
    f=lambda e,r:count_goal_loss(e,y,family,torch.nn.functional.softplus(r))
    assert torch.autograd.gradcheck(f,(eta,raw),eps=1e-6,atol=2e-5,rtol=1e-4)
    alpha=float(torch.nn.functional.softplus(raw).detach()); k=1/alpha
    rate=eta.detach().numpy()
    rate=np.exp(rate); yy=y.numpy()
    if family=='poisson': expected=(rate-yy*np.log(rate)+gammaln(yy+1)).sum(1)
    elif family=='independent_nb': expected=-nbinom.logpmf(yy,k,k/(k+rate)).sum(1)
    else:
        total=yy.sum(1); mean=rate.sum(1)
        expected=gammaln(k)-gammaln(k+total)+gammaln(yy+1).sum(1)-k*np.log(k)-(yy*np.log(rate)).sum(1)+(k+total)*np.log(k+mean)
        np.testing.assert_allclose(expected,-nbinom.logpmf(total,k,k/(k+mean))-binom.logpmf(yy[:,0],total,rate[:,0]/mean),atol=1e-13)
        grad=torch.autograd.grad(f(eta,raw).sum(),eta)[0].detach().numpy()
        np.testing.assert_allclose(grad,rate*(k+total)[:,None]/(k+mean)[:,None]-yy,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(f(eta,raw).detach().numpy(),expected,atol=2e-13,rtol=1e-13)
    np.testing.assert_allclose(count_rows(rate,yy,family,alpha)['goal_joint_nll'],expected,atol=2e-13,rtol=1e-13)


@pytest.mark.parametrize('family',['independent_nb','shared_gamma'])
def test_normalization_marginals_moments_and_explicit_tail(family):
    mean=np.array([1.3,2.1]); alpha=.4; k=1/alpha; cap=100
    h,a=np.meshgrid(np.arange(cap+1),np.arange(cap+1),indexing='ij')
    y=np.column_stack((h.ravel(),a.ravel())).astype(float)
    eta=torch.tensor(np.log(mean)).expand(len(y),2)
    p=np.exp(-count_goal_loss(eta,torch.tensor(y),family,torch.tensor(alpha,dtype=torch.float64)).numpy()).reshape(cap+1,cap+1)
    tail_bound=nbinom.sf(cap,k,k/(k+mean)).sum()
    assert tail_bound<1e-30
    np.testing.assert_allclose(p.sum(),1,atol=1e-13)
    np.testing.assert_allclose(p.sum(1),nbinom.pmf(np.arange(cap+1),k,k/(k+mean[0])),atol=1e-14)
    np.testing.assert_allclose(p.sum(0),nbinom.pmf(np.arange(cap+1),k,k/(k+mean[1])),atol=1e-14)
    np.testing.assert_allclose([(p*h).sum(),(p*a).sum()],mean,atol=1e-12)
    np.testing.assert_allclose([(p*(h-mean[0])**2).sum(),(p*(a-mean[1])**2).sum()],mean+alpha*mean**2,atol=1e-11)
    np.testing.assert_allclose((p*(h-mean[0])*(a-mean[1])).sum(),alpha*np.prod(mean) if family=='shared_gamma' else 0,atol=1e-11)


@pytest.mark.parametrize('family',['independent_nb','shared_gamma'])
def test_poisson_limit_rates_gradients_and_unequal_weighted_groups(family):
    eta=torch.tensor([[.2,-.4],[1.1,.5],[0.,.7],[-.7,-.3]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[0.,0.],[3.,1.],[1.,2.],[0.,4.]],dtype=torch.float64)
    w=torch.tensor([.3,2.,1.,.7],dtype=torch.float64)
    poisson=count_goal_loss(eta,y).sum(); pg=torch.autograd.grad(poisson,eta)[0]
    for alpha in (0.,1e-10):
        loss=count_goal_loss(eta,y,family,torch.tensor(alpha,dtype=torch.float64)).sum()
        torch.testing.assert_close(loss,poisson,rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(torch.autograd.grad(loss,eta)[0],pg,rtol=1e-8,atol=1e-8)
    def calculate(parts):
        value=sum((count_goal_loss(eta[ix],y[ix],family,torch.tensor(.2,dtype=torch.float64))*w[ix]).sum()/w.sum() for ix in parts)
        return value,torch.autograd.grad(value,eta)[0]
    v,g=calculate([np.arange(4)]); vv,gg=calculate([[0],[1,2,3]])
    torch.testing.assert_close(v,vv,rtol=1e-14,atol=1e-14);torch.testing.assert_close(g,gg,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('family',['poisson','independent_nb','shared_gamma'])
def test_structured_orientation_neutrality_centering_and_parameter_gradients(family):
    cfg=GoalAuxiliary(architecture='attack_defence',likelihood=family,competitions=(11,23))
    head=AttackDefenceHead(cfg,2).double()
    with torch.no_grad():
        head.attack.weight.copy_(torch.tensor([[.3,-.1]]));head.defence.weight.copy_(torch.tensor([[.1,.2]]))
        head.scoring.copy_(torch.tensor([.2,-.3]));head.home_effect.copy_(torch.tensor([.4,.1]))
    h=torch.tensor([[1.,2.],[.3,-.7],[.4,.8]],dtype=torch.float64)
    a=torch.tensor([[.8,-.1],[.4,.9],[-.2,.6]],dtype=torch.float64)
    competitions=torch.tensor([11,23,11]);neutral=torch.tensor([0.,1.,float('nan')])
    out=head(h,a,competitions,neutral);mu=head.scoring[[0,1,0]];gamma=head.home_effect[[0,1,0]]
    expected=torch.stack((mu+torch.tensor([1,0,1])*gamma+(h@head.attack.weight.T).flatten()-(a@head.defence.weight.T).flatten(),
                          mu+(a@head.attack.weight.T).flatten()-(h@head.defence.weight.T).flatten()),1)
    torch.testing.assert_close(out['log_rates'],expected)
    ca,cd=.23,-.17
    centered=torch.stack((mu+ca-cd+torch.tensor([1,0,1])*gamma+out['attack'][:,0]-ca-(out['defence'][:,1]-cd),
                          mu+ca-cd+out['attack'][:,1]-ca-(out['defence'][:,0]-cd)),1)
    torch.testing.assert_close(centered.exp(),out['rates'],rtol=1e-14,atol=1e-14)
    params=dict(head.named_parameters());names=list(params);values=tuple(params.values())
    y=torch.tensor([[3.,1.],[0.,0.],[2.,1.]])
    def f(*args):
        d=torch.func.functional_call(head,dict(zip(names,args)),(h,a,competitions,neutral))
        return count_goal_loss(d['log_rates'],y,family,d.get('alpha'))
    assert torch.autograd.gradcheck(f,values,eps=1e-6,atol=2e-5,rtol=1e-4)
    with pytest.raises(ValueError,match='Unsupported'):head(h,a,torch.tensor([11,99,11]),neutral)


def test_calibration_invariance_and_resampled_rank_metrics():
    from sklearn.metrics import roc_auc_score,average_precision_score
    p=np.array([[.6,.2,.2],[.1,.3,.6],[.2,.3,.5],[.3,.4,.3],[.2,.4,.4],[.6,.1,.3]])
    y=np.array([0,1,2,1,0,2]);fit=draw_calibration(p,y);q=calibrate(p,fit['intercept'])
    np.testing.assert_allclose(q[:,1].mean(),np.mean(y==1),atol=1e-13)
    np.testing.assert_array_equal(np.argsort(p[:,1],kind='stable'),np.argsort(q[:,1],kind='stable'))
    w=np.array([[1,1,1,1,1,1],[2,0,1,2,0,1],[0,3,1,1,1,0]])
    auc,ap=rank_bootstrap(p[:,1],y==1,w)
    for i,counts in enumerate(w):
        ix=np.repeat(np.arange(len(y)),counts)
        np.testing.assert_allclose(auc[i],roc_auc_score(y[ix]==1,p[ix,1]),atol=1e-14)
        np.testing.assert_allclose(ap[i],average_precision_score(y[ix]==1,p[ix,1]),atol=1e-14)


def test_extreme_count_logits_and_zero_count_derivatives():
    for family in ('poisson','independent_nb','shared_gamma'):
        eta=torch.tensor([[-100.,100.],[-20.,30.]],dtype=torch.float64,requires_grad=True)
        raw=torch.tensor(-2.,dtype=torch.float64,requires_grad=True)
        loss=count_goal_loss(eta,torch.tensor([[0.,3.],[1.,0.]]),family,torch.nn.functional.softplus(raw)).sum()
        assert torch.isfinite(loss)
        grads=torch.autograd.grad(loss,(eta,raw),allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
