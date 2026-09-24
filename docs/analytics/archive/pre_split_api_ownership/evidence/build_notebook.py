from datetime import datetime, timezone
import os, re
import nbformat
from nbclient import NotebookClient
from support import ROOT, SCRATCH, guard, save, sha
source = guard()
target = ROOT / 'notebooks/07_splits_quickstart.ipynb'
assert not target.exists(), 'Preserve existing notebook.'
examples = re.findall(r'^```python\n(.*?)^```',(ROOT/'docs/analytics/splits.md').read_text(encoding='utf-8'),re.M|re.S)
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
notebook = nbformat.v4.new_notebook(cells=[
    md('''# Choose splits and inspect their membership

## Goal and setup
Load the pinned Premier League 2023/24 and 2024/25 publications, assemble a match
dataset, and inspect chronological and retrospective split plans. Use the existing
`misc314_py314` kernel. This notebook trains no model.

Default feature/split availability uses earlier kickoff as a retrospective proxy,
not measured result-publication time. The [guide](../docs/analytics/splits.md)
contains timing examples and equations; the [reference](../docs/analytics/splits_reference.md)
lists every setting and validation rule.'''),
    code(examples[0]),
    md('''## Steps: choose chronological membership
Start with 20 observed round blocks, hold out six at a time, and score after the
first held-out block. Earlier test rows stay held out. Keep the final short horizon.
Training eligibility can remove postponed or unavailable matches.'''),
    code(examples[1]),
    code(examples[2]),
    md('''## Compare other split families
Match/group K-fold and CPCV are retrospective. GroupKFold below holds competition-season
groups together. CPCV keeps kickoff ties together and records 15 folds and five
paths under its default configuration. Its default intervals are points at kickoff;
meaningful information-overlap purging requires justified interval timestamps.'''),
    code(examples[6] + '\n' + examples[7]),
    code('''choices = {'chronological': plan, 'match_kfold': match_plan,
           'competition_season': season_plan, 'cpcv': cpcv_plan}
assert all(p.model_selector is None and p.refit_policy is None for p in choices.values())
assert all(set(f.score) <= set(f.test) and set(f.train).isdisjoint(f.test)
           for p in choices.values() for f in p.folds)
pd.DataFrame([
    {'scheme': name, 'folds': p.get_n_splits(),
     'first_train': len(p.folds[0].train), 'first_test': len(p.folds[0].test),
     'first_score': len(p.folds[0].score),
     'paths': 0 if p.paths is None else p.paths.path_id.nunique()}
    for name, p in choices.items()
])'''),
    md('''## Checks and next steps
Use original positions with `.iloc`; preserve the same dataset order when passing
`plan.split()` as a future sklearn `cv` iterable. Match tuples are identities,
not ranker group sizes. The separate scoring subset is available on each Fold.

`model_selector=None` and `refit_policy=None` are stored options only. They disable
future selection/additional scheduled refits, without removing the need for a
future initial fit in each outer fold. This module invokes none of them.

CPCV paths share observations/models. Retrospective evaluation still needs
fold-aware feature/state preparation: interval purging alone does not remove
held-out outcomes already carried into Glicko or rolling features. Statistical
reports, inner selection, strategy execution and fitting remain later layers.
See the [verification evidence](../docs/analytics/splits_check.json).'''),
],metadata={'kernelspec':{'display_name':'Python (misc314_py314)','language':'python','name':'misc314_py314'},'language_info':{'name':'python','version':'3.14'}})
nbformat.validate(notebook)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
NotebookClient(notebook,timeout=180,kernel_name='misc314_py314',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(notebook)
assert guard() == source
assert not any(output.output_type=='error' for cell in notebook.cells if cell.cell_type=='code' for output in cell.outputs)
nbformat.write(notebook,target)
record = {'notebook':target.relative_to(ROOT).as_posix(),'cells':len(notebook.cells),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),'kernel':'misc314_py314','fresh_top_to_bottom_execution':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(target),'source_sha256':source,'error_outputs':0}
save('notebook_check.json',record)
print({k:v for k,v in record.items() if k!='source_sha256'})
