"""Offline stable-ID catalog and bounded embedded-image validation."""
from copy import deepcopy
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.reporting import TeamCatalog
from xdiyo_analytics.reporting.teams import team_key
from match_results_samples import HOME,AWAY

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfN8AAAAASUVORK5CYII=')

def svg(tmp_path,body):
    path=tmp_path/'crest.svg';path.write_text(body,encoding='utf-8')
    return TeamCatalog({HOME:{'name':'Example','badge_path':str(path)}})


def test_catalog_normalizes_keys_copies_records_and_preserves_provenance():
    entries={np.uint64(HOME):{'name':'One','source_url':'recorded','license':'recorded'},AWAY:'Two'}
    before=deepcopy(entries);catalog=TeamCatalog(entries)
    assert list(catalog.entries)==[str(HOME),str(AWAY)]
    assert catalog.entries[str(HOME)]['license']=='recorded'
    catalog.entries[str(HOME)]['name']='changed';assert entries==before
    assert catalog.display(AWAY)==('Two',None,None)
    assert catalog.display(99)==('Team 99',None,None)
    assert team_key(1.0)=='1' and team_key(np.uint64(HOME))==str(HOME)
    assert team_key(np.nan)=='nan' and team_key(np.inf)=='inf'


def test_from_matches_keeps_ids_separate_and_later_frame_spelling():
    first=pd.DataFrame({'home_id':[HOME],'home_name':['Same name'],'away_id':[AWAY],'away_name':['Same name']})
    second=first.copy();second['home_name']='New spelling';second['away_name']=None
    catalog=TeamCatalog.from_matches([first,second])
    assert catalog.display(HOME)[0]=='New spelling' and catalog.display(AWAY)[0]=='Same name'
    assert len(catalog.entries)==2


def test_json_relative_badges_and_no_io_when_disabled(tmp_path,monkeypatch):
    image=tmp_path/'crest.png';image.write_bytes(PNG)
    mapping=tmp_path/'catalog.json';mapping.write_text(json.dumps({str(HOME):{'name':'One','badge_path':'crest.png'}}))
    catalog=TeamCatalog.from_json(mapping)
    assert catalog.entries[str(HOME)]['badge_path']==str(image.resolve())
    name,data,note=catalog.display(HOME,badges=True)
    assert name=='One' and note is None and base64.b64decode(data.partition(',')[2])==PNG
    def no_read(*args,**kwargs):raise AssertionError('Image read with badges disabled')
    monkeypatch.setattr(Path,'read_bytes',no_read)
    assert catalog.display(HOME)==('One',None,None)


@pytest.mark.parametrize('problem',['missing','unsupported','oversize'])
def test_invalid_badge_paths_return_name_and_note(tmp_path,problem):
    path=tmp_path/('image.gif' if problem=='unsupported' else 'image.svg')
    if problem=='unsupported':path.write_bytes(b'GIF89a')
    if problem=='oversize':path.write_bytes(b' '*2_000_001)
    name,data,note=TeamCatalog({HOME:{'name':'One','badge_path':str(path)}}).display(HOME,badges=True)
    assert name=='One' and data is None and 'badge unavailable' in note


@pytest.mark.parametrize('body',[
    '<svg><script>alert(1)</script></svg>', '<svg onload="alert(1)"/>',
    '<svg><foreignObject><div>content</div></foreignObject></svg>',
    '<svg><image href="https://example.invalid/x.png"/></svg>',
    '<svg><use href="javascript:alert(1)"/></svg>',
    '<svg><style>@import "x.css";</style></svg>',
    '<svg><rect style="fill:url(https://example.invalid/a)"/></svg>',
    '<!DOCTYPE svg [<!ENTITY x "value">]><svg/>','<not-svg/>','<svg>',
    '<svg><image href="data:image/svg+xml;base64,PHN2Zy8+"/></svg>',
    '<svg><image href="data:image/png;base64,invalid%%%"/></svg>',
    '<svg><image href="data:image/png;base64,YmFk"/></svg>',
])
def test_active_external_malformed_or_invalid_embedded_svg_falls_back(tmp_path,body):
    name,data,note=svg(tmp_path,body).display(HOME,badges=True)
    assert name=='Example' and data is None and 'badge unavailable' in note


@pytest.mark.parametrize('kind',['fragment','png','png_whitespace','jpeg'])
def test_self_contained_svg_references_preserve_original_composition(tmp_path,kind):
    if kind=='fragment':reference='#crest'
    elif kind=='jpeg':reference='data:image/jpeg;base64,'+base64.b64encode(b'\xff\xd8\xfffixture').decode()
    else:
        payload=base64.b64encode(PNG).decode()
        if kind=='png_whitespace':payload=' \n\t'.join(payload[i:i+8] for i in range(0,len(payload),8))
        reference='data:image/png;base64,'+payload
    body=f'<svg xmlns="http://www.w3.org/2000/svg"><image href="{reference}"/></svg>'
    _,data,note=svg(tmp_path,body).display(HOME,badges=True)
    assert note is None and base64.b64decode(data.partition(',')[2])==(tmp_path/'crest.svg').read_bytes()
