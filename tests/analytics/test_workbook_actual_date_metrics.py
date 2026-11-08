import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

path=Path(__file__).resolve().parents[2]/'examples/workbook_actual_date/analyze_results.py'
spec=importlib.util.spec_from_file_location('workbook_actual_date_analysis',path)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def rows():
    return pd.DataFrame(dict(competition_id=[17]*4,season_id=[1]*4,event_id=[1,2,3,4],
        original13=[True]*4,stage=['original13']*4,family=['time_C/E1']*4,seed=[17]*4,
        y=[0,1,2,1],goals_for=[1.,0.,0.,2.],goals_against=[0.,0.,1.,2.],
        p_home=[.5,.4,.25,.4],p_draw=[.25,.2,.25,.2],p_away=[.25,.4,.5,.4],
        lambda_home=[1.,1.,1.,1.],lambda_away=[1.,1.,1.,1.],
        kickoff_at=pd.to_datetime(['2024-01-01T12:00Z','2024-01-02T12:00Z','2024-01-03T12:00Z','2024-01-04T12:00Z'])))


def test_ce_rps_goal_rmse_and_seed_loss_scoring():
    a=analysis.fixture_losses(rows(),'time_C/E1')
    assert a.ce.iloc[0]==pytest.approx(-np.log(.5))
    assert a.rps.iloc[0]==pytest.approx((.5**2+.25**2)/2)
    assert a.goal_mse.iloc[0]==.5
    assert analysis.summarize(a,['family']).goal_rmse.iloc[0]==pytest.approx(np.sqrt(a.goal_mse.mean()))
    assert analysis.fixture_losses(rows(),'time_B/N').goal_mse.isna().all()


def test_finance_strict_ev_probability_missing_price_and_original13_scope():
    r=rows(); r.loc[2,'original13']=False
    odds=r[analysis.KEYS].copy();odds['draw_opening_odds']=[4.,5.01,100.,np.nan]
    odds['result_available_at']=r.kickoff_at
    ledger,summary,daily=analysis.finances(r,odds,[17])
    assert ledger.event_id.tolist()==[1,2,4]
    assert ledger.selected.tolist()==[False,True,False]
    assert summary.tickets.iloc[0]==1
    assert summary.profit.iloc[0]==pytest.approx(4.01)
    assert summary.missing_or_invalid_odds.iloc[0]==1
    assert daily.cumulative_profit.iloc[-1]==pytest.approx(4.01)


def test_invalid_probability_and_duplicate_odds_fail_closed():
    r=rows();r.loc[0,'p_home']=0
    with pytest.raises(ValueError,match='Invalid probabilities'):
        analysis.fixture_losses(r,'time_C/E1')
    odds=pd.DataFrame(dict(competition_id=[17,17],season_id=[1,1],event_id=[1,1]))
    with pytest.raises(ValueError,match='Duplicate odds'):
        analysis.finances(rows(),odds,[17])
