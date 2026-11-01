"""Read-only verification of existing private selections after catalogue repairs."""
import argparse
import hashlib
import json
from pathlib import Path
from xdiyo_analytics.data.catalogue import select_data,resolve_selection
from xdiyo_analytics.ui.recipe import prepare_recipe


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalogue',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.catalogue)
    paths=sorted(root.rglob('*'));before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    versions=json.loads((root/'import-receipt.json').read_text(encoding='utf-8'))['versions']
    union=select_data(root,snapshots=versions,mode='union')
    resolved=resolve_selection(root,union)
    assert len(resolved.matches)==len(union['target_ids'])==len(set(union['target_ids']))==14235
    recipes={}
    for name in ('single','extended'):
        recipe=json.loads((root/(name+'-recipe.json')).read_text(encoding='utf-8'))
        selected=resolve_selection(root,recipe['data_selection']['selection'])
        prepared=prepare_recipe(recipe)
        recipes[name]=dict(fingerprint=selected.provenance['data_selection']['fingerprint'],targets=len(prepared.dataset.X),
            folds=len(prepared.split_plan.folds),history=len(selected.provenance['data_selection']['history_ids']))
        assert len(prepared.dataset.X)==8
    after={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    assert before==after
    result=dict(versions=versions,union_targets=len(resolved.matches),union_fingerprint=union['fingerprint'],recipes=recipes,
        private_files_checked=len(before),private_files_unchanged=True,models_fitted=0)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
