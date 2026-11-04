"""Compare representative original schemas; missing evidence fails explicitly."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
root=Path(__file__).resolve().parents[1];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');a.output.mkdir(parents=True,exist_ok=True)
records=[]
for group in json.loads(a.manifest.read_text()):
 name=group['name'];folder=a.output/name;folder.mkdir(parents=True,exist_ok=True);jobs=folder/'jobs.json';jobs.write_text(json.dumps(group['jobs'],indent=2))
 for label,source in [('frozen',Path(group['source'])),('candidate',root)]:
  env['PYTHONPATH']=str(source/'src')
  cmd=[sys.executable,str(root/'tests/analytics/gat_legacy_inference_worker.py'),'--source',str(source),'--jobs',str(jobs.resolve()),'--output',str((folder/label).resolve()),'--device',a.device]
  if label=='candidate':cmd.append('--roundtrip')
  with (folder/(label+'.log')).open('w') as log:result=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(str(folder/(label+'.log')))
 left=json.loads((folder/'frozen/receipt.json').read_text());right=json.loads((folder/'candidate/receipt.json').read_text())
 for x,y in zip(left,right):
  assert y.pop('roundtrip') is True
  assert x==y,(name,x['id'])
  with np.load(folder/'frozen'/(x['id']+'.npz')) as aa,np.load(folder/'candidate'/(x['id']+'.npz')) as bb:
   for key in aa.files:np.testing.assert_array_equal(aa[key],bb[key],err_msg=x['id']+' '+key)
  records.append(dict(study=name,id=x['id'],passed=True,scope='four fixed original fixture probes',tolerance='exact',roundtrip=True))
 (a.output/'comparison.json').write_text(json.dumps(records,indent=2));print(name+': '+str(len(left))+' original schemas pass',flush=True)
