"""Recoverable manifest-only publication without duplicating dataset tables."""
import hashlib
import json
from pathlib import Path


def publish_integration(root, stem, metadata, *, expected_manifest, expected_publication):
    from ._source import _open_source
    from .native_integration import validate_integration
    root=Path(root).resolve()
    source=_open_source(root,stem)
    validate_integration(metadata,source.manifest['scope'])
    journal=root/'_team_seasons'/'native_integration'/(stem+'.pending.json')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    paths=[source.path,root/(stem+'.manifest.json')]
    if journal.exists():
        plan=json.loads(journal.read_bytes())
        if plan['metadata']!=metadata:raise ValueError('Pending native integration differs')
    else:
        files=[]
        for path,expected in zip(paths,(expected_manifest,expected_publication)):
            value=json.loads(path.read_bytes())
            if value.get('native_integration')==metadata:continue
            if digest(path)!=expected:raise ValueError(f'Native publication changed concurrently: {path}')
            value['native_integration']=metadata
            payload=(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode()
            files.append(dict(path=path.relative_to(root).as_posix(),before=expected,
                              after=hashlib.sha256(payload).hexdigest(),payload=payload.decode()))
        plan=dict(metadata=metadata,files=files)
        if files:
            journal.parent.mkdir(parents=True,exist_ok=True)
            with journal.open('x',encoding='utf8') as f:json.dump(plan,f,ensure_ascii=False)
    # Validate the entire transaction before any replacement. A retry accepts
    # only its exact old/new fingerprints and completes an interrupted pair.
    for item in plan['files']:
        path=(root/item['path']).resolve()
        if path not in paths or digest(path) not in (item['before'],item['after']):
            raise ValueError('Concurrent or invalid native integration journal')
        if hashlib.sha256(item['payload'].encode()).hexdigest()!=item['after']:
            raise ValueError('Corrupt native integration journal')
    for item in plan['files']:
        path=root/item['path']
        if digest(path)==item['after']:continue
        temp=path.with_suffix('.integration-tmp')
        temp.write_bytes(item['payload'].encode())
        temp.replace(path)
    if journal.exists():journal.unlink()
    return dict(stem=stem,manifest_sha256=digest(paths[0]),publication_sha256=digest(paths[1]),
                observation_tables_rewritten=False)


def materialize_awarded_rulings(root, stem):
    """Correct reviewed flags in both physical exports; retain source flags.

    Four-file recoverable publication, temporary staging only. Never change
    sporting scores or synthesize unavailable/abandoned matches.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from ._source import _open_source
    from .native_integration import attach_integration
    root=Path(root).resolve();source=_open_source(root,stem)
    publication_path=root/(stem+'.manifest.json')
    publication=json.loads(publication_path.read_bytes())
    metadata=source.manifest.get('native_integration',{})
    awarded={r['event_id'] for r in metadata.get('rulings',[]) if r.get('is_awarded') is True}
    journal=root/'_team_seasons'/'native_integration'/(stem+'.awards-pending.json')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    paths=[source.table_path('matches'),root/publication['season_file'],source.path,publication_path]
    if journal.exists():
        plan=json.loads(journal.read_bytes())
    else:
        if not awarded:return dict(stem=stem,changed=[])
        native=pq.ParquetFile(paths[0]).read()
        attach_integration(native.to_pandas(),metadata)  # bind exact reviewed identities
        ids=native['event_id'].to_pylist();flags=native['is_awarded'].to_pylist()
        changed=[eid for eid,flag in zip(ids,flags) if eid in awarded and flag is not True]
        if not changed:return dict(stem=stem,changed=[])
        files=[]
        for path in paths[:2]:
            table=pq.ParquetFile(path).read()
            missing=set(changed)-set(table['event_id'].to_pylist())
            if missing:raise ValueError(f'Nested awarded identities missing: {missing}')
            result=table
            if 'provider_is_awarded' not in result.column_names:
                result=result.append_column('provider_is_awarded',table['is_awarded'])
            mask=pc.is_in(table['event_id'],value_set=pa.array(changed,type=table['event_id'].type))
            result=result.set_column(result.schema.get_field_index('is_awarded'),'is_awarded',pc.if_else(mask,True,table['is_awarded']))
            preserved=[c for c in table.column_names if c!='is_awarded']
            if not table.select(preserved).equals(result.select(preserved),check_metadata=True):
                raise ValueError('Award repair changed observations')
            temp=path.with_suffix('.awards-tmp');pq.write_table(result,temp,compression='zstd')
            files.append(dict(path=path.relative_to(root).as_posix(),before=digest(path),after=digest(temp)))
        manifest=dict(source.manifest)
        manifest['tables']=dict(manifest['tables'])
        manifest['tables']['matches']={**manifest['tables']['matches'],
            'sha256':files[0]['after'],'bytes':paths[0].with_suffix('.awards-tmp').stat().st_size}
        publication.update(sha256=files[1]['after'],bytes=paths[1].with_suffix('.awards-tmp').stat().st_size)
        for path,value in zip(paths[2:],(manifest,publication)):
            temp=path.with_suffix('.awards-tmp')
            temp.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
            files.append(dict(path=path.relative_to(root).as_posix(),before=digest(path),after=digest(temp)))
        plan=dict(stem=stem,changed=changed,files=files)
        journal.parent.mkdir(parents=True,exist_ok=True)
        with journal.open('x',encoding='utf8') as stream:json.dump(plan,stream)
    for item in plan['files']:
        path=(root/item['path']).resolve();temp=path.with_suffix('.awards-tmp')
        if path not in paths or digest(path) not in (item['before'],item['after']):
            raise ValueError('Award publication changed concurrently')
        if digest(path)!=item['after'] and (not temp.is_file() or digest(temp)!=item['after']):
            raise ValueError('Missing or corrupt staged award publication')
    for item in plan['files']:
        path=root/item['path']
        if digest(path)!=item['after']:path.with_suffix('.awards-tmp').replace(path)
    journal.unlink()
    return plan
