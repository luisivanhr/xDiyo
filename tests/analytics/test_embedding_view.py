"""Rendering contracts; projections and fitted models are never recomputed."""
import numpy as np
import pandas as pd

from xdiyo_analytics.reporting.embedding_view import embedding_figure
from xdiyo_analytics.reporting.contracts import AnalysisReport, Artifact, StudyResult, StudyRun


def table():
    # Interleaved leagues and a flat axis used to produce invisible badges.
    return pd.DataFrame(dict(x=[-2., 0., 3.], y=[0., 0., 0.], league=[2, 1, 2],
                             team_name=['A', 'B', '<C>'], support=[1., 2., 3.], gate=[.1, .2, .3]))


def test_square_projection_keeps_geometry_and_badge_ownership():
    data = table(); before = data.copy(deep=True)
    fig = embedding_figure(data, ['a.png', None, 'c.png'], 'PCA')
    pd.testing.assert_frame_equal(data, before)
    assert fig.layout.width == fig.layout.height
    assert np.diff(fig.layout.xaxis.range) == np.diff(fig.layout.yaxis.range)
    assert fig.layout.yaxis.scaleanchor == 'x'
    assert list(fig.data[0].x) == [-2., 3.]
    assert fig.data[0].text == ('A', '&lt;C&gt;')
    assert [im.name for im in fig.layout.images] == ['badge:0:0', 'badge:0:1']
    assert all(im.sizex == im.sizey > 0 for im in fig.layout.images)
    assert fig.data[1].marker.size == (12,)
    assert all(trace.mode == 'markers' for trace in fig.data)


def test_overall_embedding_fold_selector_retains_each_fold_and_exports():
    fig = embedding_figure(table(), [None]*3, 'PCA')
    result = StudyResult('Embeddings', tables={'projection': table()}, artifacts=[
        Artifact('plotly', fig, f'Fold {fold}', options={'embedding_fold': str(fold)}) for fold in [0, 3]
    ])
    report = AnalysisReport([StudyRun('Embedding', 'overall', 'score', None, 'match', np.arange(3), 3, result)])
    html = report.to_html()
    assert '<select class="embedding-fold-selector">' in html
    assert '<option value="3">Fold 3</option>' in html
    assert '<div data-embedding-fold="3" hidden>' in html
    assert 'projection · 3 rows' in html
    assert html.count('class="plot" data-spec=') == 2
    result.artifacts.pop()
    assert '<select class="embedding-fold-selector">' not in report.to_html()
