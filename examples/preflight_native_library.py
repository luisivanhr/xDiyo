"""Write a native SofaScore query/readiness report without fitting or data writes."""
import argparse
import csv
import json
from pathlib import Path
from xdiyo_analytics.data import NativeLibrary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/xDiyo_data')
    parser.add_argument('--query',default='{}',help='JSON object with competitions/seasons/phases/teams/annual_groups filters')
    parser.add_argument('--require',nargs='*',choices=['countdown','standings','movements'],default=[])
    parser.add_argument('--availability',choices=['retrospective_finished','explicit'],default='retrospective_finished')
    parser.add_argument('--entry-information',choices=['retrospective_season_entry','explicit'],default='retrospective_season_entry')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve();root=Path(args.root).resolve()
    if output.is_relative_to(root):raise ValueError('Preflight reports must be outside the source dataset')
    report=NativeLibrary(root).preflight(query=json.loads(args.query),required_features=args.require,
        availability=args.availability,entry_information=args.entry_information)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    rows=[]
    for r in report['cohorts']:
        rows.append({**{k:r[k] for k in ('publication','competition_id','competition','season_id','season','phase_id','phase','membership','readiness')},
                     'countdown':r['countdown']['status'],'standings':r['standings']['status'],
                     'countdown_reason':r['countdown'].get('reason',''),'standings_reason':r['standings'].get('reason',''),
                     'unknown_movements':r['movement']['unknown_teams'],**r['counts'],
                     'blockers':' | '.join(r['blockers']),'assumptions':' | '.join(r['assumptions'])})
    with output.with_suffix('.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f"Wrote {len(rows)} phase/cohort rows to {output}")


if __name__=='__main__':main()
