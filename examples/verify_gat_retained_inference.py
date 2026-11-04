"""All18 original exports versus their frozen runtime, without retraining."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
frozen=a.research/'experiments/gat_corners_propagation_20261101_01/shared/source_frozen'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
a.output.mkdir(parents=True,exist_ok=True)
for label,source in [('frozen',frozen),('candidate',root)]:
 out=a.output/label;env['PYTHONPATH']=str(source/'src')
 cmd=[sys.executable,str(root/'tests/analytics/gat_retained_inference_worker.py'),'--source',str(source),'--research',str(a.research),'--references',str(a.references),'--output',str(out),'--device',a.device]
 if label=='candidate':cmd+=['--roundtrip']
 with (a.output/(label+'.log')).open('w') as log:
  result=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(str(a.output/(label+'.log')))
 print(label+': all original exports evaluated',flush=True)
x=json.loads((a.output/'frozen/receipt.json').read_text());y=json.loads((a.output/'candidate/receipt.json').read_text())
assert len(x)==len(y)==18
records=[]
for original,candidate in zip(x,y):
 assert candidate.pop('roundtrip') is True
 assert original==candidate,original['name']
 name=original['name']
 with np.load(a.output/'frozen'/(name+'.npz')) as left,np.load(a.output/'candidate'/(name+'.npz')) as right:
  for key in left.files:np.testing.assert_array_equal(left[key],right[key],err_msg=name+' '+key)
 records.append(dict(name=name,fixtures=original['count'],passed=True,tolerance='exact',roundtrip=True))
(a.output/'comparison.json').write_text(json.dumps(records,indent=2))
print('All18: exact inputs, logits, probabilities, goal rates, identities and round trips',flush=True)
