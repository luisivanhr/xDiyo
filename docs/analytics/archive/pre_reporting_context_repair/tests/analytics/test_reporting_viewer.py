"""Viewer contracts, full downloads, safe ordinary content and notebook display."""
import base64
from copy import deepcopy
from html.parser import HTMLParser
import json
import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from IPython.display import IFrame
from reporting_samples import sample, Custom
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import Artifact, StudyResult, AnalysisReport


class Tags(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def simple_report(artifacts=None, tables=None, title='Synthetic verification'):
    return PreTrainingAnalysis({'custom': Custom('overall', 'all', lambda ctx:
        StudyResult(title, artifacts=artifacts or [], tables=tables or {}))}, title=title).run(sample())


def test_custom_artifact_layouts_text_escaping_and_full_csv_download(tmp_path):
    hostile = '<img src=x onerror="window.bad=1">'
    frame = pd.DataFrame({'label': [hostile] + ['value']*204, 'position': range(205)})
    artifacts = [Artifact('text', hostile), Artifact('table', frame, hostile),
                 Artifact('html', '<div id="trusted">trusted local HTML</div>'),
                 Artifact('custom', {'message': 'own layout'})]
    report = simple_report(artifacts, {'full': frame}, title=hostile)
    with pytest.raises(ValueError, match='renderer'):
        report.to_html()
    original = deepcopy(frame)
    path = tmp_path / 'nested/report.html'
    html = report.to_html(path, renderers={'custom': lambda artifact: '<section id="extension">own layout</section>'})
    assert path.read_text(encoding='utf-8') == html
    assert hostile not in html and '&lt;img' in html
    assert '<div id="trusted">' in html and '<section id="extension">' in html
    assert 'first 200 of 205 rows' in html
    links = [attrs for tag, attrs in Tags(html).tags if tag == 'a' and attrs.get('class') == 'download']
    decoded = [base64.b64decode(link['href'].split(',', 1)[1]).decode('utf-8-sig') for link in links]
    assert sum(text == frame.to_csv(index=False) for text in decoded) == 2
    assert any(text.splitlines()[0] == 'row_position' and len(text.splitlines()) == 13 for text in decoded)
    pd.testing.assert_frame_equal(frame, original)


def test_plot_json_is_escaped_and_plotly_embedded_once():
    hostile = '</script><script>window.untrusted=1</script>'
    figure = go.Figure(go.Scatter(x=[1, 2], y=[1, np.nan], name=hostile))
    figure.update_layout(title=hostile)
    original = figure.to_json()
    html = simple_report([Artifact('plotly', figure, hostile), Artifact('plotly', figure)]).to_html()
    assert hostile not in html
    assert html.count('plotly.js v') == 1
    specs = re.findall(r'<script type="application/json" id="plot-spec-\d+">(.*?)</script>', html, re.S)
    assert len(specs) == 2
    assert all(json.loads(spec)['data'][0]['name'] == hostile for spec in specs)
    assert all('\\u003c' in spec for spec in specs)
    assert figure.to_json() == original


def test_native_notebook_iframe_automatic_and_explicit_display_do_not_recalculate(monkeypatch):
    calls = []
    def calculate(ctx):
        calls.append('run')
        return StudyResult('one', artifacts=[Artifact('text', 'display once')])
    report = PreTrainingAnalysis({'r': Custom('overall', 'all', calculate)}, title='a "quoted" <title>').run(sample())
    shown = []
    monkeypatch.setattr('IPython.display.display', shown.append)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        frame = report.to_notebook(height=480)
        explicit = frame._repr_html_()
        automatic = report._repr_html_()
        assert report.show(height=320) is None
    assert caught == [] and isinstance(frame, IFrame) and isinstance(shown[0], IFrame)
    assert calls == ['run']
    for html, height in [(explicit, '480'), (automatic, '800')]:
        attrs = [attrs for tag, attrs in Tags(html).tags if tag == 'iframe'][0]
        assert attrs['src'] == 'about:blank' and attrs['height'] == height
        assert attrs['title'] == report.title
        assert attrs['srcdoc'].startswith('<!doctype html>')
        assert 'display once' in attrs['srcdoc']
        assert '<title>a &quot;quoted&quot; &lt;title&gt;</title>' in attrs['srcdoc']


@pytest.mark.parametrize('height', [0, -1, True, 1.5, '800'])
def test_invalid_notebook_height(height):
    with pytest.raises(ValueError, match='height'):
        AnalysisReport().to_notebook(height=height)


def test_empty_report_notebook_display_is_available():
    assert 'No reporters selected' in AnalysisReport().to_notebook()._repr_html_()


def test_bad_table_kind_has_actionable_error():
    with pytest.raises(TypeError, match='DataFrame'):
        simple_report([Artifact('table', ['not', 'a', 'frame'])]).to_html()
