"""Independent output and lifetime checks for the library performance branch."""
import gc
import json
import weakref
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.experiments.recovery import pack, _iter_pack_json, dump_bundle, load_bundle


@pytest.mark.parametrize('size', [0, 3, 2050])
def test_streaming_recovery_identical_bytes_types_and_attrs(tmp_path, size):
    frame = pd.DataFrame({'a': pd.array(([1,None,0]*((size+2)//3))[:size],dtype='Int64'),
                          'b': pd.array((['é',None,'']*((size+2)//3))[:size], dtype='string'),
                          'c': pd.Categorical((['x',None,'z']*((size+2)//3))[:size], categories=['z','x'], ordered=True)})
    frame['time'] = pd.date_range('2024-01-01', periods=size, tz='Asia/Tokyo')
    frame.attrs = {'nested': {'geometry':[1, None, np.nan]}}
    payload = {'frame': frame, 'series': frame.a, 'tuple': (np.inf, -np.inf, pd.NaT, pd.NA),
               'empty': [], 'array': np.array([[1., np.nan]])}
    expected = json.dumps(pack(payload), ensure_ascii=False, allow_nan=False)
    assert ''.join(_iter_pack_json(payload)) == expected
    path=tmp_path/'bundle.json';dump_bundle(path,payload)
    assert path.read_text(encoding='utf-8') == '{"schema": 1, "payload": '+expected+'}'
    restored=load_bundle(path)
    pd.testing.assert_frame_equal(restored['frame'], frame)
    pd.testing.assert_series_equal(restored['series'], frame.a)
    with pytest.raises(FileExistsError): dump_bundle(path, payload)


def test_shared_strings_detached_and_rich_text_preserved(monkeypatch):
    from xdiyo_analytics.odds import workbook
    archive=BytesIO()
    with ZipFile(archive,'w') as zip:
        zip.writestr('xl/sharedStrings.xml', '<sst xmlns="'+workbook.NS[1:-1]+'">'+
            ''.join('<si><r><t xml:space="preserve"> a </t></r><r><t>b</t></r></si>' for _ in range(3000))+'</sst>')
    archive.seek(0)
    book=workbook._Workbook.__new__(workbook._Workbook);book.archive=ZipFile(archive);book.strings={}
    original=workbook.ET.iterparse
    roots=[]
    def parse(*args,**kwargs):
        first=True
        for event,element in original(*args,**kwargs):
            if first: roots.append(element); first=False
            yield event,element
    monkeypatch.setattr(workbook.ET,'iterparse',parse)
    book.resolve([2999]);assert book.strings == {2999:' a b'}
    assert len(roots[0]) == 0
    with pytest.raises(ValueError, match='Invalid shared-string'): book.resolve([3000])
    book.archive.close()


def test_csv_cache_tracks_content_mutation_and_preserves_links():
    from xdiyo_analytics.reporting.viewer import _csv_link
    cache={};frame=pd.DataFrame({'x':[1.]})
    assert _csv_link(frame,'one',cache=cache).replace('one.csv','two.csv') == _csv_link(frame,'two',cache=cache)
    assert len(cache)==1
    before=_csv_link(frame,'one',cache=cache);frame.iloc[0,0]=2
    assert _csv_link(frame,'one',cache=cache) != before and len(cache)==2


def test_odds_operation_over_eight_partitions_bounded_and_released():
    from xdiyo_analytics.odds._cache import operation, table, _active, _Tables
    calls=[]; refs=[]
    def load(key):
        calls.append(key); frame=pd.DataFrame({'price':[float(key)]})
        refs.append(weakref.ref(frame)); return frame
    @operation
    def nested():
        for key in range(12):
            assert table(key, lambda: load(key)).price.iloc[0] == key
    @operation
    def run():
        nested(); nested()
        assert len(calls)==12 and len(_active.get().entries)==12
    run();gc.collect()
    assert _active.get() is None and all(ref() is None for ref in refs)
    small=_Tables(limit=150)
    for key in range(20): small.get(key, lambda: pd.DataFrame({'x':[key]}))
    assert small.bytes <= 150 and len(small.entries) <= 1
    # Changed content identities cannot hit an old entry.
    cache=_Tables();a=cache.get(('path','hash1'),lambda: pd.DataFrame({'x':[1]}))
    b=cache.get(('path','hash2'),lambda: pd.DataFrame({'x':[2]}))
    assert a.x.iloc[0] == 1 and b.x.iloc[0] == 2
    @operation
    def fail():
        table('failure',lambda: load(99)); raise RuntimeError('cancel')
    with pytest.raises(RuntimeError):fail()
    assert _active.get() is None
