"""Bounded fitting parity, one declared seed/fold per retained family."""
import argparse,os,subprocess,sys,json
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];frozen=a.research/'experiments/gat_corners_propagation_20261101_01/shared/source_frozen'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
def exact(a,b,path='root'):
 if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
 elif isinstance(a,dict):
  assert list(a)==list(b),(path,'keys')
  for k in a:exact(a[k],b[k],path+'.'+str(k))
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
 else:assert a==b,(path,a,b)
a.output.mkdir(parents=True,exist_ok=True);receipt=[]
for family in ['time_B/N','time_C/E1','time_B/CD_DW']:
 folder=a.output/family.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
 for label,source in [('frozen',frozen),('candidate',root)]:
  env['PYTHONPATH']=str(source/'src');cmd=[sys.executable,str(root/'tests/analytics/gat_retained_training_worker.py'),'--source',str(source),'--research',str(a.research),'--output',str(folder/label),'--family',family]
  if label=='candidate':cmd.append('--candidate')
  with (folder/(label+'.log')).open('w') as log:result=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(str(folder/(label+'.log')))
 x=torch.load(folder/'frozen/trajectory.pt',weights_only=False);y=torch.load(folder/'candidate/trajectory.pt',weights_only=False);exact(x,y)
 receipt.append(dict(family=family,fold='fold_2',seed=17,updates=3,passed=True,tolerance='exact',state='full optimizer gradients RNG selection restoration'))
 (a.output/'comparison.json').write_text(json.dumps(receipt,indent=2));print(family+': exact three-update parity',flush=True)
