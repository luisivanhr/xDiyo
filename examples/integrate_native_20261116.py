"""Plan/apply the audited native metadata migration; never call a collector.

One dataset, unchanged storage keys. Raw observations are shared, not copied.
Manifest changes intentionally invalidate old selection pins. Restore their
recorded Git revision for old experiments; refresh explicitly for new work.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.data.native_integration import POLICY, validate_integration

ROOT = Path('data/xDiyo_data')
OUT = Path('docs/analytics/native_integration_20261116')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rule_plan(row, frame):
    cohort = row['reconciliation']
    unsupported = dict(status='unsupported', reason='Independent table/calendar scope or competition format requires implementation/review')
    standings, countdown = deepcopy(unsupported), deepcopy(unsupported)
    tids = sorted(int(x) for x in frame.tournament_id.dropna().unique())
    teams = set(frame.home_id) | set(frame.away_id)
    n = len(row['source_teams'])
    pairs = Counter(tuple(sorted((a,b))) for a,b in zip(frame.home_id,frame.away_id))
    repeats = set(pairs.values())
    rounds = frame['round'].dropna()
    complete = (len(tids) == 1 and len(pairs) == n*(n-1)//2 and len(repeats) == 1
                and repeats <= {1,2} and len(teams) == n and cohort.get('schedule_verified') is True)
    # Pair completeness is cross-checked against the already reviewed provider
    # scope/format evidence, not treated as proof of independent split rules.
    phases = cohort.get('reviewed_phases', [])
    independent = (len(phases) == 1 and len(phases[0].get('table_ids', [])) == 1
        or cohort.get('selection') == 'all')
    if complete and independent and rounds.notna().all() and len(rounds) == len(frame):
        meetings = next(iter(repeats)); total = (n-1 if n%2 == 0 else n)*meetings
        if rounds.max() <= total and all(frame.loc[frame['round'].eq(r), ['home_id','away_id']].stack().duplicated().sum()==0 for r in rounds.unique()):
            countdown = dict(status='supported', format='double' if meetings == 2 else 'single',
                             total_rounds=total, team_count=n, tournament_ids=tids,
                             basis='verified_regular_scope_and_complete_pair_schedule')
            standings = dict(status='supported', table_scope=['competition_id','season_id'],
                             tournament_ids=tids, rule='independent_3_1_0', meetings_per_pair=meetings,
                             official_table=False, basis='sporting_result_form_ranks_not_administrative_final_table')
    # Previous supported formats keep their existing implementation and
    # phase safeguards; this migration never derives a new format from names.
    if not row['new']:
        standings = dict(status='legacy_verified', reason='Existing explicit DerivedStandings membership/schedule validation remains required')
        countdown = dict(status='legacy_verified', reason='Existing ID/phase-specific countdown checks remain required')
    if row['stem'] == 'Chile_Primera_Division_19_19':
        countdown = dict(status='supported', format='double', total_rounds=30, team_count=16,
                         tournament_ids=[64475], basis='planned_double_round_robin_curtailed_after_played_rounds')
        standings = dict(status='unsupported', reason='Curtailed season: 193 played fixtures versus 194 administrative table results; ruling availability is not established. No synthetic draw or final-table backfill.')
    # Official/table treatment differs from a sporting result stream even if
    # the fixture topology is otherwise ordinary.
    if any(r.get('standings_score') is not None or r.get('disposition') == 'retain_annulled'
           for r in cohort.get('reviewed_fixture_dispositions', [])):
        standings = dict(status='unsupported', reason='Administrative or annulled results require independently timed table/ruling scope; source sporting scores remain separate')
    return dict(enforce=True, standings=standings, countdown=countdown,
                format_sources=cohort.get('format_source', []),
                scope_warning='Annual grouping is not a standings-table or countdown-calendar identity')


def plan():
    inventory = json.loads((OUT/'inventory.json').read_text(encoding='utf8'))
    competitions, teams = {}, {}
    def latest(target, key, item):
        if key not in target or item['evidence']['observed_at'] > target[key]['evidence']['observed_at']:
            target[key] = item
    for row in inventory['publications']:
        if row['sofa']:
            latest(competitions, row['scope']['league_id'], row['sofa']['competition'])
            for key, item in row['sofa']['teams'].items(): latest(teams, key, item)
    plans, exceptions = [], []
    for row in inventory['publications']:
        stem = row['stem']; src = _open_source(ROOT, stem)
        if sha(src.path) != row['manifest_sha256']:
            raise ValueError('Baseline changed before planning: '+stem)
        cid = row['scope']['league_id']
        if cid not in competitions or row['sofa_season'] is None:
            exceptions.append(dict(stem=stem, reason='Missing captured native competition/season identity'))
            continue
        p = pq.ParquetFile(src.table_path('matches'))
        cols = ['event_id','competition_id','season_id','tournament_id','round','home_id','away_id',
                'home_name','away_name','kickoff_utc','is_awarded','status','home_score_current','away_score_current']
        frame = p.read(columns=cols).to_pandas()
        if frame.duplicated(['competition_id','season_id','event_id']).any():
            raise ValueError('Duplicate native fixture identity: '+stem)
        selected_teams = {k:dict(teams[k]['value'], evidence=teams[k]['evidence'],
                               aliases=inventory['team_aliases'][k]) for k in row['source_teams'] if k in teams}
        missing = sorted(set(row['source_teams']) - set(selected_teams))
        if missing: exceptions.append(dict(stem=stem, reason='Missing captured team display names', team_ids=missing))
        cohort = row['reconciliation']
        metadata = dict(policy=POLICY, competition=competitions[cid]['value'],
            season={k:row['sofa_season']['value'][k] for k in ('id','name','year')},
            evidence=dict(competition=competitions[cid]['evidence'], season={k:v for k,v in row['sofa_season'].items() if k!='value'}),
            teams=selected_teams, tournaments={k:v['value'] for k,v in (row['sofa'] or {}).get('tournaments', {}).items()},
            storage_alias=dict(stem=stem, league_key=row['scope']['league_name'], reason='Stable compatibility key, not the provider display name'),
            rules=rule_plan(row,frame),
            coverage=dict(kind='curtailed' if stem == 'Chile_Primera_Division_19_19' else 'completed_match_snapshot' if cohort.get('snapshot') else 'published_scope',
                          played_rows=len(frame), source_snapshot_flag=cohort.get('snapshot'),
                          note='Coverage applies to the selected native stages only; see archive_cohort'),
            availability=dict(classification='conditional_retrospective',
                results='Explicit result bounds take precedence; otherwise earlier finished kickoff is a retrospective proxy, not verified publication evidence',
                movements='Retrospective season-entry review; no historical publication times established',
                schedule='Revised exported schedule; historical issuance schedule availability is not established',
                administrative_outcomes='Ruling date does not establish exact historical publication time'),
            previous_revision=dict(git_commit=inventory['base_revision'], manifest_sha256=row['manifest_sha256'],
                                   publication_sha256=row['publication_sha256'], tables=row['tables']), rulings=[])
        for review in cohort.get('reviewed_fixture_dispositions', []):
            matches = frame.loc[frame.event_id.eq(review['event_id'])]
            if matches.empty: continue
            r=matches.iloc[0]
            if review.get('standings_score') is not None or review['disposition']=='retain_annulled':
                score=review.get('standings_score')
                evidence=dict(event_id=int(r.event_id),home_id=int(r.home_id),away_id=int(r.away_id),kickoff_utc=float(r.kickoff_utc),
                    source=review.get('source'),reason=review['reason'],
                    play_status='played_annulled' if review['disposition']=='retain_annulled' else 'played_administrative_award')
                if score is not None:
                    evidence.update(is_awarded=True,administrative_home_score=score[0],administrative_away_score=score[1])
                if int(r.event_id)==14115336:
                    evidence['ruling_date']='2025-09-16'
                metadata['rulings'].append(evidence)
        validate_integration(metadata, src.manifest['scope'])
        plans.append(dict(stem=stem, before_manifest=row['manifest_sha256'], before_publication=row['publication_sha256'], metadata=metadata))
    result=dict(schema_version=1,inventory_sha256=sha(OUT/'inventory.json'),plans=plans,exceptions=exceptions)
    (OUT/'plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(plans=len(plans),exceptions=len(exceptions),
        countdown=Counter(x['metadata']['rules']['countdown']['status'] for x in plans),
        standings=Counter(x['metadata']['rules']['standings']['status'] for x in plans))))
    return result


def apply():
    from xdiyo_analytics.data.native_publication import publish_integration, materialize_awarded_rulings
    plans=json.loads((OUT/'plan.json').read_text(encoding='utf8'))
    receipts=[];awards=[]
    for item in plans['plans']:
        receipts.append(publish_integration(ROOT,item['stem'],item['metadata'],
            expected_manifest=item['before_manifest'],expected_publication=item['before_publication']))
        result=materialize_awarded_rulings(ROOT,item['stem'])
        if result['changed']:
            awards.append(result)
            receipts[-1].update(manifest_sha256=sha(_open_source(ROOT,item['stem']).path),
                publication_sha256=sha(ROOT/(item['stem']+'.manifest.json')),observation_tables_rewritten=True)
    (OUT/'publication.json').write_text(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),receipts=receipts),indent=2)+'\n',encoding='utf8')
    if awards:
        (OUT/'awarded_materialization.json').write_text(json.dumps(awards,indent=2)+'\n',encoding='utf8')
    print('Published',len(receipts),'metadata updates; reviewed awarded flags materialized:',sum(len(r['changed']) for r in awards))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    apply() if args.apply else plan()
