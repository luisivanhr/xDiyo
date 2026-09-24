from datetime import datetime, timezone
import hashlib, re
from support import ROOT, guard, save, sha
source = guard()
namespace = {}; results = []; guides = {}
for guide in ['splits.md','splits_reference.md']:
    path = ROOT / 'docs/analytics' / guide
    for number, code in enumerate(re.findall(r'^```python\n(.*?)^```',path.read_text(encoding='utf-8'),re.M|re.S),1):
        exec(compile(code,f'{guide}:example_{number}','exec'),namespace)
        results.append({'guide':guide,'example':number,'code_sha256':hashlib.sha256(code.encode()).hexdigest()})
    guides[guide] = sha(path)
assert guard() == source
save('examples_check.json',{'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source,'example_count':len(results),'examples':results,'guide_sha256':guides})
print({'status':'passed','examples':len(results)})
