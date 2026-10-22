"""Independent numerical oracles for Wheatcroft equations (6)-(9)."""
from dataclasses import asdict
import pytest
from xdiyo_analytics.ratings import GAPParameters, GAPState, WheatcroftGAP, derive_shots_off_target
from xdiyo_analytics.ratings.gap_core import donor_mean


def test_equations_6_to_8_hand_calculation_and_nonmutation():
    h, a = GAPState(10, 6, 8, 7), GAPState(9, 8, 6, 12)
    algorithm = WheatcroftGAP()
    pred = algorithm.predict(h, a)
    assert (pred.home, pred.away, pred.difference) == (11, 6, 5)
    nh, na = algorithm.update(h, a, 15, 4, GAPParameters(.2, .75, .6))
    assert asdict(nh) == pytest.approx(dict(HA=10.6, HD=5.7, AA=8.2, AD=6.9))
    assert asdict(na) == pytest.approx(dict(HA=8.84, HD=8.32, AA=5.76, AD=12.48))
    assert h == GAPState(10, 6, 8, 7) and a == GAPState(9, 8, 6, 12)
    assert algorithm.predict(h, a) == pred


def test_clipping_and_asymmetric_cross_venue_weights():
    alg = WheatcroftGAP()
    # Home residual -10; away residual 0. Each component clips independently.
    h, a = alg.update(GAPState(2, 0, 10, 0), GAPState(0, 10, 0, 18), 0, 0, GAPParameters(1, .8, .25))
    assert h == GAPState(0, 0, 8, 0)
    assert a == GAPState(0, 2.5, 0, 15.5)
    h, a = alg.update(GAPState(), GAPState(), 4, 2, GAPParameters(.5, .75, .2))
    assert h == GAPState(1.5, .75, .5, .25)
    assert asdict(a) == pytest.approx(dict(HA=.8, HD=1.6, AA=.2, AD=.4))


@pytest.mark.parametrize('values', [(0,.5,.5), (-1,.5,.5), (float('inf'),.5,.5), (float('nan'),.5,.5),
                                     (.2,0,.5), (.2,1,.5), (.2,.5,0), (.2,.5,1), (.2,float('nan'),.5),
                                     (.2,.5,float('inf')), (True,.5,.5)])
def test_invalid_parameters(values):
    with pytest.raises(ValueError):
        GAPParameters(*values)


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), True])
def test_invalid_observations_and_states(bad):
    with pytest.raises(ValueError):
        WheatcroftGAP().update(GAPState(), GAPState(), bad, 0, GAPParameters(.2,.5,.5))
    with pytest.raises(ValueError):
        GAPState(HA=bad)


def test_strategy_selector_and_donors():
    with pytest.raises(ValueError):
        WheatcroftGAP('future_variant')
    assert donor_mean([GAPState(2,4,6,8), GAPState(4,6,8,10)]) == GAPState(3,5,7,9)
    with pytest.raises(ValueError, match='empty donor'):
        donor_mean([])


def test_shot_derivation_is_explicit_and_preserves_missing():
    import numpy as np
    import pandas as pd
    a = pd.Series([10., 0., np.nan], index=[4,2,8])
    b = pd.Series([4., 0., 1.], index=a.index)
    result = derive_shots_off_target(a, b, total_is_on_plus_off=True, definition_evidence='synthetic additive definition')
    assert result.index.equals(a.index)
    assert result.iloc[:2].tolist() == [6., 0.]
    assert pd.isna(result.iloc[2])
    for total, on, verified, evidence in [([2],[3],True,'x'), ([float('inf')],[1],True,'x'), ([2],[1],False,'x'),
                                          ([2],[1],True,''), ([2,3],[1],True,'x')]:
        with pytest.raises(ValueError):
            derive_shots_off_target(total, on, total_is_on_plus_off=verified, definition_evidence=evidence)
