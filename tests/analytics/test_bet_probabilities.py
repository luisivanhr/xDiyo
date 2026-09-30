"""Probability aggregation follows literal BetOption settlement boundaries."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson
from xdiyo_analytics.features import Stat
from xdiyo_analytics.labels import BetOption, MatchTotal, Above, Outcome
from xdiyo_analytics.evaluation.probabilities import bet_probabilities, negative_binomial_bet_probabilities

TOTAL = MatchTotal(Stat('ALL', 'Match overview', 'cornerKicks'))


@pytest.mark.parametrize('selection,expected', [('under', .6), ('over', .4)])
def test_half_corner_line_sums_exact_mass_and_preserves_rows(selection, expected):
    p = pd.DataFrame([[.1, .2, .3, .4]], columns=[0, 6, 7, 8], index=pd.Index([2**63+5], dtype='uint64', name='match'))
    result = bet_probabilities(p, BetOption(TOTAL, selection, line=7.5))
    assert result.p_win.iloc[0] == pytest.approx(expected)
    assert result.p_push.iloc[0] == 0
    assert result.p_loss.iloc[0] == pytest.approx(1-expected)
    pd.testing.assert_index_equal(result.index, p.index)


@pytest.mark.parametrize('selection,on_equal,expected', [
    ('under','push',[.3,.3,.4]), ('over','push',[.4,.3,.3]),
    ('under','loss',[.3,0,.7]), ('over','loss',[.4,0,.6])])
def test_integer_corner_equality_is_push_or_loss(selection,on_equal,expected):
    p = pd.DataFrame([[.1,.2,.3,.4]],columns=[0,6,7,8])
    np.testing.assert_allclose(bet_probabilities(p,BetOption(TOTAL,selection,line=7,on_equal=on_equal)),[expected])


@pytest.mark.parametrize('selection,draw,expected', [
    ('win','loss',[.5,0,.5]),('loss','loss',[.2,0,.8]),('draw','loss',[.3,0,.7]),
    ('win','push',[.5,.3,.2]),('loss','push',[.2,.3,.5])])
def test_outcome_options(selection,draw,expected):
    p=pd.DataFrame([[.2,.3,.5]],columns=[-1,0,1])
    np.testing.assert_allclose(bet_probabilities(p,BetOption(Outcome(),selection,draw=draw)),[expected])


@pytest.mark.parametrize('selection,win', [('yes',.8),('no',.2)])
def test_binary_above_yes_no(selection,win):
    actual=bet_probabilities(pd.DataFrame([[.2,.8]],columns=[0,1]),BetOption(Above(TOTAL,7.5),selection))
    np.testing.assert_allclose(actual,[[win,0,1-win]])


def test_multitarget_requires_explicit_choice_and_uses_only_that_target():
    p=pd.DataFrame([[.3,.7,.8,.2]],columns=pd.MultiIndex.from_tuples([('corners',6),('corners',9),('shots',6),('shots',9)]))
    option=BetOption(TOTAL,'under',line=7.5)
    with pytest.raises(ValueError,match='target'):bet_probabilities(p,option)
    assert bet_probabilities(p,option,target='corners').p_win.iloc[0]==.3
    assert bet_probabilities(p,option,target='shots').p_win.iloc[0]==.8


@pytest.mark.parametrize('classes', [[0,7.5],[-1,7],['0','7'],['0','20+'],[0,0],[0,np.inf]])
def test_count_classes_must_be_distinct_exact_nonnegative_integers(classes):
    p=pd.DataFrame([[.3,.7]],columns=classes)
    with pytest.raises(ValueError):bet_probabilities(p,BetOption(TOTAL,'under',line=7.5))


@pytest.mark.parametrize('values', [[.2,.3],[-.2,1.2],[np.nan,.2],[np.inf,.2]])
def test_malformed_distributions_rejected(values):
    with pytest.raises(ValueError):bet_probabilities(pd.DataFrame([values],columns=[6,9]),BetOption(TOTAL,'under',line=7.5))


@pytest.mark.parametrize('line',[0,7,7.5,200.5])
@pytest.mark.parametrize('selection',['over','under'])
@pytest.mark.parametrize('on_equal',['push','loss'])
def test_nb_exact_tails_match_scipy_without_support_truncation(line,selection,on_equal):
    mu=pd.Series([0.,2.,10.,1000.],index=[91,4,7,2])
    alpha=pd.Series([.2,0.,.15,2.],index=mu.index)
    result=negative_binomial_bet_probabilities(mu,alpha,BetOption(TOTAL,selection,line=line,on_equal=on_equal))
    expected=[]
    for m,a in zip(mu,alpha):
        distribution=poisson(m) if a==0 else nbinom(1/a,1/(1+a*m))
        below=distribution.cdf(np.ceil(line)-1)
        above=distribution.sf(np.floor(line))
        equal=distribution.pmf(line) if float(line).is_integer() else 0.
        win,loss=(below,above) if selection=='under' else (above,below)
        expected.append([win,equal if on_equal=='push' else 0.,loss+(equal if on_equal=='loss' else 0.)])
    np.testing.assert_allclose(result,expected,atol=1e-14)
    np.testing.assert_allclose(result.sum(axis=1),1,atol=1e-14)
    pd.testing.assert_index_equal(result.index,mu.index)


def test_nb_zero_mean_degenerate_and_small_dispersion_poisson_limit():
    option=BetOption(TOTAL,'under',line=7.5)
    mean=pd.Series([0.,5.,20.])
    exact=negative_binomial_bet_probabilities(mean,0.,option)
    near=negative_binomial_bet_probabilities(mean,1e-8,option)
    np.testing.assert_allclose(exact.p_win,poisson.cdf(7,mean),atol=1e-14)
    np.testing.assert_allclose(near,exact,atol=1e-7)
    assert exact.iloc[0].tolist()==[1.,0.,0.]


def test_nb_rejects_misaligned_and_invalid_parameters():
    option=BetOption(TOTAL,'over',line=7.5)
    with pytest.raises(ValueError,match='align'):
        negative_binomial_bet_probabilities(pd.Series([2,3],index=[1,2]),pd.Series([.2,.3],index=[2,1]),option)
    for mean,alpha in [([-1],.2),([2],-.1),([np.nan],.2),([2],np.inf)]:
        with pytest.raises(ValueError):negative_binomial_bet_probabilities(pd.Series(mean),alpha,option)
