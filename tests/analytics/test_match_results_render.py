"""Escaped embedded data, exact identity strings and renderer independence."""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import MatchResultReporter, TeamCatalog
from xdiyo_analytics.reporting.match_view import render_match_results
from match_results_samples import fixture_context,result_from_context,HOME,AWAY


def payload(html):
    return json.loads(re.search(r'<script type="application/json" class="match-data">(.*?)</script>',html,re.S).group(1))


def test_embedded_identity_and_numeric_round_canonicalization():
    ctx=fixture_context();report=MatchResultReporter(type='overall',partition='test',round=1).run(ctx)
    value=payload(render_match_results(report.artifacts[0]))
    assert value['initial']['round']=='1'
    assert [row['round'] for row in value['rows']]==['1','1','2',None]
    assert value['rows'][0]['home_id']==str(HOME) and value['rows'][0]['away_id']==str(AWAY)
    assert value['rows'][0]['event_id']==str(2**63+101)
    assert value['rows'][0]['fold_id']=='4' and value['rows'][0]['row_position']=='0'
    assert value['rows'][0]['result']==1. and len(value['rows'])==4


def test_names_targets_and_labels_cannot_close_embedded_script():
    hostile='</script><script>globalThis.INJECTED=1</script><img src=x onerror=alert(1)>&'
    ctx=fixture_context();ctx.y[hostile]=[hostile]*4;ctx.predictions['predict'][hostile]=[hostile]*4
    ctx.metadata['stage']=hostile
    report=MatchResultReporter(type='overall',partition='test',target=hostile,
        catalog=TeamCatalog({HOME:hostile,AWAY:'Other'})).run(ctx)
    html=render_match_results(report.artifacts[0]);embedded=payload(html)
    assert html.count('</script>')==1 and '<img' not in html and hostile not in html
    assert embedded['teams'][str(HOME)]['name']==hostile
    assert embedded['rows'][0]['target']==hostile and embedded['rows'][0]['prediction']==hostile


def test_rendering_does_not_reopen_badges_or_recompute_predictions(tmp_path,monkeypatch):
    path=tmp_path/'crest.svg';path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    ctx=fixture_context();training=result_from_context(ctx)
    reporter=MatchResultReporter(type='overall',catalog={HOME:{'name':'Home','badge_path':str(path)},AWAY:'Away'},show_badges=True)
    report=PostTrainingAnalysis({'fixtures':reporter}).run(training)
    def no_read(*args,**kwargs):raise AssertionError('Viewer performed badge IO')
    monkeypatch.setattr(Path,'read_bytes',no_read)
    html=report.to_html()
    assert payload(html)['teams'][str(HOME)]['badge'].startswith('data:image/svg+xml;base64,')


def test_large_integer_numeric_labels_are_not_rounded_in_payload():
    ctx=fixture_context();ctx.y['count']=pd.array([2**63+i for i in range(4)],dtype='uint64[pyarrow]')
    ctx.predictions['predict']['count']=ctx.y['count']
    report=MatchResultReporter(type='overall',comparison='categorical').run(ctx)
    values=payload(render_match_results(report.artifacts[0]))['rows']
    assert [r['result'] for r in values]==[str(2**63+i) for i in range(4)]
    assert all(r['status']=='correct' for r in values)


@pytest.mark.parametrize('type_,partition',[('timeline','score'),('overall','train'),('overall','model')])
def test_unsupported_report_scope_rejected(type_,partition):
    training=result_from_context(fixture_context())
    with pytest.raises(ValueError):PostTrainingAnalysis({'bad':MatchResultReporter(type=type_,partition=partition)}).run(training)
