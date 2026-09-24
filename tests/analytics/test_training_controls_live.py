"""Controlled-clock display events and SVG geometry; no live frontend claim."""
import re
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import IPython.display
import xdiyo_analytics.training.live as live_module
from xdiyo_analytics.training import LiveLossPlot, TrainingEvent


def event(step, value, *, attempt=0, final=False):
    return TrainingEvent(3, attempt, step, {'train_loss': value, 'validation_loss': value + 1}, 'train_loss', final=final)


def test_throttle_updates_one_display_handle_and_final_always_refreshes(monkeypatch):
    created, updated = [], []
    class Handle:
        def update(self, html): updated.append(html.data)
    def display(html, *, display_id):
        assert display_id is True; created.append(html.data); return Handle()
    clock = iter([0., .2, 1.2, 1.3])
    monkeypatch.setattr(live_module.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(IPython.display, 'display', display)
    plot = LiveLossPlot(interval=1)
    for e in [event(1, 3), event(2, 2), event(3, 1), event(3, 1, final=True)]: plot(e)
    assert len(created) == 1 and len(updated) == 2
    assert all(len(points) == 3 for points in plot.series.values())
    assert updated[-1] == plot.to_html() and '<svg' in updated[-1]


def test_downsample_preserves_skipped_nan_gap_and_full_saved_series():
    plot = LiveLossPlot(max_points=2)
    plot.series[(3, 0, 'train_loss')] = [(1, 5), (2, 4), (3, np.nan), (4, 2), (5, 1)]
    html = plot.to_html(); svg = ET.fromstring(re.search(r'<svg.*?</svg>', html).group(0))
    polylines = svg.findall('polyline'); circles = svg.findall('circle')
    assert len(polylines) == len(circles) == 2
    assert [float(poly.attrib['points'].split(',')[0]) for poly in polylines] == [60., 660.]
    assert all(' ' not in poly.attrib['points'] for poly in polylines)
    assert len(plot.series[(3, 0, 'train_loss')]) == 5
    np.testing.assert_allclose([float(poly.attrib['points'].split(',')[1]) for poly in polylines],
                              [250-215*(5-.8)/4.4, 250-215*(1-.8)/4.4], atol=.005)


def test_metric_names_are_escaped_and_folds_attempts_stay_separate():
    plot = LiveLossPlot(metrics='<img src=x onerror=alert(1)>')
    plot.series[(2, 0, '<img src=x onerror=alert(1)>')] = [(1, 2)]
    plot.series[(2, 1, '<img src=x onerror=alert(1)>')] = [(1, 1)]
    html = plot.to_html()
    assert '<img' not in html and '&lt;img' in html
    assert 'attempt 0' in html and 'attempt 1' in html and 'fold 2' in html
    assert html.count('<circle') == 2


@pytest.mark.parametrize('values', [[], [(1, np.nan)], [(1, np.inf), (2, -np.inf)]])
def test_nonfinite_only_series_is_explicit_waiting_state(values):
    plot = LiveLossPlot(); plot.series[(0, 0, 'train_loss')] = values
    assert 'Waiting for finite' in plot.to_html() and '<svg' not in plot.to_html()


@pytest.mark.parametrize('kwargs', [dict(interval=0), dict(interval=True), dict(interval=np.inf),
    dict(max_points=1), dict(max_points=True), dict(max_points=2.5)])
def test_invalid_display_settings_rejected(kwargs):
    with pytest.raises(ValueError): LiveLossPlot(**kwargs)
