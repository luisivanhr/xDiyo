"""Crash/retry publication protocol; all writes use isolated temporary roots."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from examples.repair_bundesliga_standings_20261027 import (
    _publish, _publication_lock, digest, repair,
)


def prepared(root):
    evidence=root/'evidence.json'
    evidence.write_text(json.dumps({'season':'Fixture'}),encoding='utf-8')
    receipt={'status':'applied','evidence_sha256':digest(evidence),'files':[]}
    pending=[]
    for name in ['matches.parquet','nested.parquet','native.manifest.json','Fixture.manifest.json']:
        live=root/name;live.write_bytes(b'old '+name.encode())
        staged=live.with_suffix(live.suffix+'.standings-tmp');staged.write_bytes(b'new '+name.encode())
        pending.append((staged,live))
        receipt['files'].append(dict(path=name,before=digest(live),after=digest(staged)))
    journal=root/'Fixture.standings-pending.json'
    return evidence,receipt,pending,journal


@pytest.mark.parametrize('replacements',[0,1,2,3,4])
def test_process_termination_and_fresh_process_retry(tmp_path,replacements):
    evidence,receipt,pending,journal=prepared(tmp_path)
    (tmp_path/'plan.json').write_text(json.dumps(receipt),encoding='utf-8')
    child=r"""
import json,os,sys
from pathlib import Path
from examples.repair_bundesliga_standings_20261027 import _publish,_publication_lock
root=Path(sys.argv[1]);stop=int(sys.argv[2]);receipt=json.loads((root/'plan.json').read_text())
original=Path.replace
count=0
live={root/item['path'] for item in receipt['files']}
journal=root/'Fixture.standings-pending.json'
def interrupted(self,target):
 global count
 result=original(self,target)
 if Path(target)==journal and stop==0:os._exit(77)
 if Path(target) in live:
  count+=1
  if count==stop:os._exit(77)
 return result
Path.replace=interrupted
pending=[((root/i['path']).with_suffix(Path(i['path']).suffix+'.standings-tmp'),root/i['path']) for i in receipt['files']]
with _publication_lock(root):_publish(root,journal,pending,receipt)
"""
    result=subprocess.run([sys.executable,'-c',child,str(tmp_path),str(replacements)],capture_output=True,text=True)
    assert result.returncode==77,result.stderr
    assert journal.exists()
    snapshot={p:p.read_bytes() for _,p in pending}
    with pytest.raises(RuntimeError,match='rerun with --apply'):
        repair(tmp_path,evidence=evidence)
    assert snapshot=={p:p.read_bytes() for _,p in pending}
    retry=subprocess.run([sys.executable,'-c',
        'import sys; from examples.repair_bundesliga_standings_20261027 import repair; repair(sys.argv[1],apply=True,evidence=sys.argv[2])',
        str(tmp_path),str(evidence)],capture_output=True,text=True)
    assert retry.returncode==0,retry.stderr
    assert not journal.exists()
    for item in receipt['files']:
        assert digest(tmp_path/item['path'])==item['after']


@pytest.mark.parametrize('damage',['target','payload','evidence'])
def test_recovery_preflights_all_files_without_overwriting_external_changes(tmp_path,monkeypatch,damage):
    evidence,receipt,pending,journal=prepared(tmp_path)
    original=Path.replace
    def fail_first(self,target):
        if Path(target)==pending[0][1]:raise OSError('injected interruption')
        return original(self,target)
    with monkeypatch.context() as m:
        m.setattr(Path,'replace',fail_first)
        with pytest.raises(OSError,match='injected'):
            _publish(tmp_path,journal,pending,receipt)
    damaged={'target':pending[-1][1],'payload':pending[-1][0],'evidence':evidence}[damage]
    if damage=='evidence': damaged.write_text('{"season":"Fixture","changed":true}',encoding='utf-8')
    else: damaged.write_bytes(b'external edit')
    snapshot={p:p.read_bytes() for _,p in pending}
    with pytest.raises(ValueError,match='changed|differs'):
        repair(tmp_path,apply=True,evidence=evidence)
    assert snapshot=={p:p.read_bytes() for _,p in pending}
    assert journal.exists()


def test_live_publisher_lock_excludes_second_process(tmp_path):
    with _publication_lock(tmp_path):
        child=subprocess.run([sys.executable,'-c',
            'import sys; from pathlib import Path; from examples.repair_bundesliga_standings_20261027 import _publication_lock; '+
            'lock=_publication_lock(Path(sys.argv[1])); lock.__enter__()',str(tmp_path)],capture_output=True,text=True)
        assert child.returncode!=0
        assert 'Another Bundesliga standings publication is active' in child.stderr


@pytest.mark.parametrize('stop',[1,3])
def test_full_parquet_repair_recovers_before_source_hash_checks(tmp_path,monkeypatch,stop):
    import shutil
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from examples.repair_bundesliga_standings_20261027 import EVIDENCE
    source=Path(__file__).resolve().parents[2]/'data/xDiyo_data'
    review=json.loads(EVIDENCE.read_text(encoding='utf-8'))
    pub_name=review['season']+'.manifest.json'
    pub=json.loads((source/pub_name).read_text(encoding='utf-8'))
    manifest=json.loads((source/pub['manifest']).read_text(encoding='utf-8'))
    manifest_path=tmp_path/pub['manifest'];manifest_path.parent.mkdir(parents=True)
    native=manifest_path.parent/manifest['tables']['matches']['file']
    nested=tmp_path/pub['season_file']
    statistics=manifest_path.parent/manifest['tables']['statistics']['file']
    shutil.copyfile((source/pub['manifest']).parent/manifest['tables']['statistics']['file'],statistics)
    for path,original,info in ((native,(source/pub['manifest']).parent/manifest['tables']['matches']['file'],manifest['tables']['matches']),
                               (nested,source/pub['season_file'],pub)):
        table=pq.ParquetFile(original).read()
        table=table.filter(pc.not_equal(table['event_id'],review['event_id']))
        pq.write_table(table,path,compression='zstd',use_compliant_nested_type=False)
        info.update(sha256=digest(path),bytes=path.stat().st_size)
        info['rows' if path==native else 'matches']=len(table)
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    (tmp_path/pub_name).write_text(json.dumps(pub),encoding='utf-8')
    before={p:digest(p) for p in (native,nested,manifest_path,tmp_path/pub_name)}
    assert repair(tmp_path,evidence=EVIDENCE)['status']=='validated_dry_run'
    original_replace=Path.replace
    count=0
    def interrupted(self,target):
        nonlocal count
        result=original_replace(self,target)
        if Path(target) in before:
            count+=1
            if count==stop:raise OSError('injected real publication interruption')
        return result
    with monkeypatch.context() as m:
        m.setattr(Path,'replace',interrupted)
        with pytest.raises(OSError,match='injected real'):
            repair(tmp_path,apply=True,evidence=EVIDENCE)
    assert (tmp_path/(review['season']+'.standings-pending.json')).exists()
    result=repair(tmp_path,apply=True,evidence=EVIDENCE)
    assert result['status']=='applied'
    assert result['original_rows_preserved']
    assert repair(tmp_path,evidence=EVIDENCE)=={'status':'already_applied','matches':306}
    for item in result['files']:
        assert digest(tmp_path/item['path'])==item['after']
    assert digest(statistics)==manifest['tables']['statistics']['sha256']
    tables=[pq.ParquetFile(p).read() for p in (native,nested)]
    assert tables[0].equals(tables[1].select(tables[0].column_names),check_metadata=False)
