from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.composition.contracts import OutputSchema, OutputRef
from xdiyo_analytics.composition.graph import ModelNode, PredictionGraphSpec
from xdiyo_analytics.composition.reducers import Mean, HardVote
from xdiyo_analytics.evaluation.decision_layer import FrozenTable
from xdiyo_analytics.evaluation.stake_policy import StakeContext, FixedStake, RiskLimits, allocate_batch
from xdiyo_analytics.evaluation import compose_bets, Parlay, MultiBet, BetSlip
from test_bet_tickets import ledger


def test_legacy_nominal_golden():
    tickets, members, _ = compose_bets(ledger(('win', 'win', 'loss')),
        BetSlip({'single': Parlay(size=1), 'system': MultiBet(size=3, sizes=(2,3), stake=8, stake_mode='total')}))
    assert tickets.stake.tolist() == [1.,1.,1.,2.,2.,2.,2.]
    assert tickets.payout.tolist() == [2.,2.,0.,8.,0.,0.,0.]
    assert tickets.profit.sum() == 1.
    assert members.stake.eq(20).all()


def test_safe_table_copies_and_rejects_outcomes():
    original = pd.DataFrame({'odds':[2.]}, index=['x'])
    frozen = FrozenTable(original)
    original.iloc[0,0] = 9
    copy = frozen.frame
    copy.iloc[0,0] = 10
    assert frozen.frame.iloc[0,0] == 2
    for column in ['result', 'settlement', 'net_profit', 'actual_value']:
        with pytest.raises(ValueError, match='Outcome'):
            FrozenTable(original.assign(**{column:1}))


def test_schema_and_reducer_contracts():
    schema = OutputSchema('y')
    f = pd.DataFrame({'y':[1.,2.]}, index=[10,20])
    np.testing.assert_array_equal(Mean((1.,3.)).reduce([f,f*3],[schema,schema]), f*2.5)
    with pytest.raises(ValueError, match='identical'):
        Mean().reduce([f,f],[schema, replace(schema, units='metres')])
    for w in [(0.,0.),(-1.,1.),(np.nan,1.),(1.,)]:
        with pytest.raises(ValueError, match='Weights'):
            Mean(w).reduce([f,f],[schema,schema])
    with pytest.raises(ValueError, match='aligned'):
        Mean().reduce([f,f.iloc[::-1]],[schema,schema])


def test_graph_rejects_missing_cycle_and_depth_before_fitting():
    schema = {'predict': OutputSchema('y')}
    with pytest.raises(ValueError, match='Unresolved'):
        PredictionGraphSpec({'a':ModelNode(None,schema,inputs=(OutputRef('missing'),))},{'predict':OutputRef('a')}).order()
    with pytest.raises(ValueError, match='cycle'):
        PredictionGraphSpec({'a':ModelNode(None,schema,inputs=(OutputRef('a'),))},{'predict':OutputRef('a')}).order()
    nodes = {n:ModelNode(None,schema,inputs=() if i==0 else (OutputRef('abc'[i-1]),), features=object()) for i,n in enumerate('abc')}
    with pytest.raises(ValueError, match='depth'):
        PredictionGraphSpec(nodes,{'predict':OutputRef('c')}).order()


def test_batch_projection_is_order_invariant_and_replaces_nominal():
    context = StakeContext(100.,40.,'units','2025-01-01',open_exposure=(('fixture',1,10.),))
    table = pd.DataFrame({'nominal_stake':[2.,3.], 'decision_at':['2025-01-01']*2, 'fixture':[(1,2),(1,3)]},index=['a','b'])
    policy, limits = FixedStake(50.,'units'), RiskLimits(exposure_caps=(('fixture',30.),))
    first = allocate_batch(FrozenTable(table),policy,context,limits)
    second = allocate_batch(FrozenTable(table.iloc[::-1]),policy,context,limits)
    pd.testing.assert_series_equal(first.amounts,second.amounts.reindex(first.amounts.index))
    assert first.amounts.tolist() == [10.,10.]
    assert first.audit.nominal_stake.tolist() == [2.,3.]
