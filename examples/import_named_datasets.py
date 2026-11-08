"""Private local import. This never rewrites legacy data or runs models."""
import argparse
import json
from pathlib import Path
import time
from xdiyo_analytics.data.catalogue import Registry, read_historical_results, read_results_workbook, combine_imports, publish_dataset, inspect_coverage
from xdiyo_analytics.data.catalogue.native import seed_comprehensive, review_premier_league, review_native_reserves
from xdiyo_analytics.data.catalogue.storage import file_hash
from xdiyo_analytics.data.catalogue import load_team_identity_decisions, apply_team_identity_decisions


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--results-txt',required=True);p.add_argument('--workbook',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output).resolve();legacy=Path(a.legacy_root).resolve()
    if output==legacy or output.is_relative_to(legacy):raise ValueError('New publications must be separate from legacy data')
    started=time.monotonic();before=[file_hash(a.results_txt),file_hash(a.workbook)]
    existing=Registry.load(output/'registry.json') if (output/'registry.json').exists() else None
    registry,native,checksums=seed_comprehensive(legacy,registry=existing)
    registry=apply_team_identity_decisions(registry,load_team_identity_decisions())
    review_premier_league(registry)
    review_native_reserves(registry)
    print('Native identities seeded; first reviewed squad batch: Premier League.',flush=True)
    txt=read_historical_results(a.results_txt,registry);print('Historical archive validated:',txt.summary,flush=True)
    workbook=read_results_workbook(a.workbook,registry);print('Workbook validated:',workbook.summary,flush=True)
    tables,summary=combine_imports([txt,workbook])
    output.mkdir(parents=True,exist_ok=True)
    registry.save(output/'registry.json')
    # Both snapshots use the same final identity registry, including new fixture IDs.
    versions={
        'comprehensive_dataset':publish_dataset(output,'comprehensive_dataset',native,registry,input_checksums=checksums),
        'international_dataset':publish_dataset(output,'international_dataset',tables,registry,input_checksums=before,summary=summary)}
    assert before==[file_hash(a.results_txt),file_hash(a.workbook)],'Original inputs changed'
    receipt=dict(versions=versions,input_checksums=before,elapsed_seconds=time.monotonic()-started,
      reviewed_scope='Premier_League senior squads; other competition/team/season identities require canonical review',
      redistribution='Local research publications only; rights not established',
      coverage={name:inspect_coverage(output,name,version) for name,version in versions.items()})
    (output/'import-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='coverage'},indent=2),flush=True)

if __name__=='__main__':main()
