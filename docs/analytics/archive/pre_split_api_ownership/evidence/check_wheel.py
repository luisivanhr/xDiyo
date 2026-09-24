"""Bounded package build and isolated imports; no installation."""
import json, shutil, subprocess, sys, tomllib, zipfile
from support import ROOT, SCRATCH, guard, save, sha
source_hashes = guard()
stage, output, unpacked = [SCRATCH/name for name in ['wheel_stage','wheel_output','wheel_unpacked']]
assert not any(path.exists() for path in [stage,output,unpacked])
stage.mkdir(); output.mkdir()
settings = tomllib.loads((ROOT/'pyproject.toml').read_text())['tool']['setuptools']
shutil.copy2(ROOT/'pyproject.toml',stage/'pyproject.toml')
copied = {'pyproject.toml':sha(ROOT/'pyproject.toml')}
for package in settings['packages']:
    relative = __import__('pathlib').Path(*package.split('.'))
    if package.startswith('xdiyo_analytics'): relative = __import__('pathlib').Path('src')/relative
    origin, destination = ROOT/relative, stage/relative
    destination.mkdir(parents=True,exist_ok=True)
    files = set(origin.glob('*.py'))
    for pattern in settings.get('package-data',{}).get(package,[]):files.update(origin.glob(pattern))
    for file in files:
        assert file.is_file() and file.parent == origin
        shutil.copy2(file,destination/file.name)
        copied[file.relative_to(ROOT).as_posix()] = sha(file)
result = subprocess.run([sys.executable,'-I','-B','-c','from setuptools.build_meta import build_wheel; import sys; print(build_wheel(sys.argv[1]))',str(output)],cwd=stage,text=True,capture_output=True)
(SCRATCH/'wheel_build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
assert result.returncode==0,result.stderr
wheels = list(output.glob('*.whl')); assert len(wheels)==1
required = ['xdiyo_analytics/splits/'+name+'.py' for name in ['__init__','core','temporal','cpcv']]
with zipfile.ZipFile(wheels[0]) as wheel:
    assert all(name in wheel.namelist() for name in required)
    wheel.extractall(unpacked)
probe = '''import importlib,json,pathlib,sys
import numpy as np
import pandas as pd
root=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
for name in ['xdiyo_analytics.splits','xdiyo_analytics.splits.core','xdiyo_analytics.splits.temporal','xdiyo_analytics.splits.cpcv','xdiyo_analytics.datasets']:
    assert pathlib.Path(importlib.import_module(name).__file__).resolve().is_relative_to(root)
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import *
base=2**63+101
meta=pd.DataFrame({'event_id':pd.Series([base+i for i in range(12)],dtype='uint64[pyarrow]'),'competition_id':17,'season_id':[2023]*6+[2024]*6,'round':list(range(1,7))*2,'kickoff_at':pd.date_range('2024-01-01',periods=12,tz='UTC')})
keys=('competition_id','season_id','event_id')
d=ModelDataset(pd.DataFrame({'x':np.arange(12,dtype=float)}),pd.DataFrame({'y':np.ones(12)}),meta,'match',keys,keys,'total')
assert len(TemporalSplit(3,test_size=2).folds(d))==4
assert len(MatchKFold(3).folds(d))==3 and len(GroupKFold(2).folds(d))==2
plan=create_split_plan(d,CPCV())
assert plan.get_n_splits()==15 and plan.paths.path_id.nunique()==5
values=[pd.DataFrame({'demo':f.test.astype(float)},index=f.test) for f in plan.folds]
paths=reconstruct_paths(plan,values)
assert len(paths)==60 and (paths.demo==paths.row_position).all()
assert plan.model_selector is None and plan.refit_policy is None
assert len(list(plan.split(d.X)))==15 and d.metadata.event_id.iloc[0]==base
import xdiyo_analytics.splits as api
assert set(api.__all__)=={'Fold','SplitPlan','MatchKFold','GroupKFold','TemporalSplit','CPCV','create_split_plan','reconstruct_paths'}
assert 'sklearn' not in sys.modules
print(json.dumps({'isolated_wheel_import':True,'exports':sorted(api.__all__),'all_families_and_paths':'passed','sklearn_imported':False}))
'''
result = subprocess.run([sys.executable,'-I','-B','-c',probe,str(unpacked)],cwd=unpacked,text=True,capture_output=True)
(SCRATCH/'wheel_import.log').write_text(result.stdout+result.stderr,encoding='utf-8')
assert result.returncode==0,result.stderr
assert all(sha(ROOT/name)==digest for name,digest in copied.items()) and guard()==source_hashes
record = {'wheel':wheels[0].relative_to(ROOT).as_posix(),'sha256':sha(wheels[0]),'source_file_count':len(copied),'source_sha256':copied,'required_members':required,'probe':json.loads(result.stdout),'no_installs':True}
save('wheel_check.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='source_sha256'},indent=2))
