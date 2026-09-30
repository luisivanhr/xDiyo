"""Read-only study inventory audit; writes grouped docs, never fits a model.

Run from the repository root with its analytics Python environment.
"""
from pathlib import Path
import ast
from collections import defaultdict
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tests/analytics')]
import pandas as pd
from xdiyo_analytics.features import FeatureBankPreset, SeededEMA, Hard, evaluate_features, RestDays
from xdiyo_analytics.features.inventory import summarize_feature_columns
from xdiyo_analytics.ratings import GlickoTransition
from xdiyo_analytics.splits import TemporalSplit
from xdiyo_analytics.data import load_seasons
from transition_samples import history_from_games, warm_games

OUT = Path(__file__).parent
notebook = ROOT / 'notebooks/20_xgboost_classifier_total_corners.ipynb'
cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
source = next(''.join(cell['source']) for cell in cells
              if 'FEATURE_SELECTION_PROPORTION =' in ''.join(cell['source']))
# Execute configuration assignments only, omitting plotting/printing/run calls.
assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)]
env = dict(ROOT=ROOT, SeededEMA=SeededEMA, Hard=Hard,
           GlickoTransition=GlickoTransition, TemporalSplit=TemporalSplit)
exec(compile(ast.Module(body=assignments, type_ignores=[]), str(notebook), 'exec'), env)
summaries = list((ROOT / 'experiments/total_corners_xgboost_classifier').glob('*/runs/*/classifier_summary/summary.json'))
completed = []
for path in summaries:
    meta = json.loads((path.parents[1] / 'run.json').read_text(encoding='utf-8'))
    if meta['status'] == 'complete':
        completed.append((meta['created_at'], path, meta))
_, summary_path, run = max(completed, key=lambda item: item[0])
run_path = summary_path.parents[1]
summary = json.loads(summary_path.read_text(encoding='utf-8'))
fold = json.loads((run_path / 'training.json').read_text(encoding='utf-8'))[0]
selected = fold['feature_columns']
manifest = pd.read_csv(summary_path.parent / 'feature_manifest.csv').feature.tolist()
assert len(selected) == summary['selected_feature_count']
assert len(manifest) == summary['features']
assert set(selected).issubset(manifest)
selected_export = pd.read_csv(summary_path.parent / 'selected_features.csv').feature.tolist()
assert selected_export == selected
provenance = dict(run_id=run['run_id'], created_at=run['created_at'], name=run['name'],
    source=str(run_path.relative_to(ROOT)), rows=summary['rows'], fit_rows=summary['fit_rows'],
    score_rows=summary['test_rows'], prepared_columns=len(manifest), selected_columns=len(selected),
    selection=dict(proportion=summary['feature_selection_proportion'],
                   correlation=summary['feature_selection_correlation']),
    historical_expressions=summary['historical_expressions'])
actual = dict(provenance=provenance, prepared=summarize_feature_columns(manifest),
              selected=summarize_feature_columns(selected))

print('Reading selected development exports for current statistic availability...', flush=True)
development = env['SEASONS'][:-1]  # selected five chronological seasons in this notebook
data = load_seasons(env['DATA_ROOT'], development, leagues=env['LEAGUES'], tables=('statistics',))
identities = list(data.statistics[['period', 'group_name', 'key']].drop_duplicates().itertuples(index=False, name=None))
del data
options = dict(windows=env['FEATURE_WINDOWS'], lags=env['FEATURE_LAGS'], spans=env['FEATURE_EMA_SPANS'],
    periods=env['FEATURE_PERIODS'], include_ratings=env['INCLUDE_RATINGS'],
    include_loo=env['INCLUDE_LOO'], loo_windows=env['LOO_WINDOWS'], loo_reducers=env['LOO_REDUCERS'],
    include_h2h=env['INCLUDE_H2H'], h2h_windows=env['H2H_WINDOWS'], h2h_reducers=env['H2H_REDUCERS'],
    warm_policy=env['WARM_FEATURE_POLICY'], warm_stat_keys=env['WARM_STAT_KEYS'],
    rating_warm_policy=env['WARM_RATING_POLICY'])
preset = FeatureBankPreset(**options)
definitions = preset.build(identities)
# Tiny synthetic rows only expand the public evaluator's multi-output names.
# No historical measurements, correlations, selection or model fit are computed.
history = history_from_games(warm_games()).iloc[:2].copy()
history.attrs['stat_columns'] = {}
synthetic_columns = {}
for index, (period, group, key) in enumerate(identities):
    if (group, key) not in preset.stats or period not in preset.periods:
        continue
    for role in ('team', 'opponent'):
        column = f'stat_{index}_{role}'
        synthetic_columns[column] = 1.0
        history.attrs['stat_columns'][column] = dict(period=period, group_name=group,
            key=key, field='value', role=role)
attributes = dict(history.attrs)
history.attrs = {}
history = pd.concat([history, pd.DataFrame(synthetic_columns, index=history.index)], axis=1)
history.attrs = attributes
values = evaluate_features(history, {**definitions, 'rest_days': RestDays()})
history_columns = list(values)
columns = [f'{side}::{name}' for side in ('home', 'away') for name in history_columns]
columns.extend(preset.postassembly_definitions(history_columns, columns))
columns.extend(preset.calendar_definitions())
# Same outer-training league vocabulary as the verified saved study. Current
# source labels are obtained separately from the match exports below.
league_data = load_seasons(env['DATA_ROOT'], development, leagues=env['LEAGUES'], tables=('matches',))
columns.extend(f'league::{league}' for league in sorted(league_data.matches.source_league.unique()))
del league_data
current = dict(provenance=dict(notebook=str(notebook.relative_to(ROOT)),
    status='configured schema, not a new fitted run', seasons=env['SEASONS'],
    development_seasons=development, holdout=env['SEASONS'][-1],
    discovery='current development export statistic identities; synthetic rows expand output names only',
    selection_proportion=env['FEATURE_SELECTION_PROPORTION'],
    selection_correlation=env['FEATURE_SELECTION_CORRELATION'],
    historical_expressions=len(definitions), grid=env['GRID'],
    expressions_equal_completed_run={key: repr(value) for key, value in definitions.items()} == run['config']['preparation']['feature_definitions']),
    prepared=summarize_feature_columns(columns))
current['provenance']['ordered_columns_equal_completed_run'] = (
    current['prepared']['ordered_columns_sha256'] == actual['prepared']['ordered_columns_sha256'])
for filename, payload in [('completed_run.json', actual), ('current_configuration.json', current)]:
    (OUT / filename).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

def tree(inventory, baseline=None):
    lines = []
    for stage, count in inventory['stage_counts'].items():
        denominator = '' if baseline is None else f" / {baseline['stage_counts'].get(stage, 0):,} available"
        lines.append(f'- **{stage}: {count:,}{denominator} columns**')
        groups = [group for group in inventory['groups'] if group['stage'] == stage]
        families = defaultdict(list)
        for group in groups:
            families[(group['variant'], group['family'])].append(group)
        for (variant, family), members in families.items():
            lines.append(f'  - {variant} / {family}: {sum(group["count"] for group in members):,}')
            for group in members:
                window = '' if group['window'] is None else f', window {group["window"]}'
                period = '' if group['period'] is None else f', {group["period"]}'
                roles = '' if not group['roles'] else f'; sides {"/".join(group["roles"])}'
                venues = '' if not group['venues'] else f'; venues {"/".join(group["venues"])}'
                fields = '' if not group['fields'] else f'; fields {"/".join(group["fields"])}'
                lines.append(f'    - {group["operation"]}{window}{period}: {group["count"]} columns{roles}{venues}{fields}')
                if group['statistics']:
                    lines.append('      - Stats/identities: ' + ', '.join(f'`{name}`' for name in group['statistics']))
    return '\n'.join(lines)

text = f'''# Classifier feature inventory

Verified 28 September 2026; no model was trained to create this inventory.

## Latest completed run: what the fitted winner actually used

Run `{run['run_id']}`, created `{run['created_at']}`. This is newer than the old
1,589-column result still displayed in notebook20. Source: `{provenance['source']}`.
The fitted fold's feature_columns, selected_features.csv and full feature manifest
agree: **{len(selected):,} selected from {len(manifest):,} prepared columns**, with
TopK **Spearman**, proportion **{summary['feature_selection_proportion']}**
(ceil(0.8 × 2,681)=2,145). The completed search had {summary['candidates']} candidates.
Winner: `{run['name']}`.

| Family of columns | Actually selected | Prepared |
|---|---:|---:|
{chr(10).join(f'| {stage} | {actual["selected"]["stage_counts"].get(stage, 0):,} | {count:,} |' for stage, count in actual['prepared']['stage_counts'].items())}

This tree counts the actual selected columns. A group can include different
subsets for home/away or for/against; listed dimensions are observed members,
not a claim that every Cartesian combination survived selection. The grouped
JSON retains exact counts by computation/window/period. Ordered-name hashes
allow comparison with the underlying saved schema without dumping thousands of rows.

<details>
<summary>Expand the selected feature hierarchy: computation → window → period → statistics</summary>

{tree(actual['selected'], actual['prepared'])}

</details>

## Current saved notebook configuration: available inputs if run now

The current source requests **no feature selector** (`None`); all **{len(columns):,}**
assembled columns would be offered to the model. Its grid contains 54 candidates.
This is a configured schema, not a newly evaluated result. Export availability
was read only from 20_21–23_24; 24_25 remains held out. There are
**{len(definitions):,} historical expressions**, plus rest/calendar/identity and
post-assembly arithmetic columns. Baseline and warmed variants coexist.

Current expression definitions equal the latest completed run's prepared bank:
**{current['provenance']['expressions_equal_completed_run']}**. Ordered column names also match:
**{current['provenance']['ordered_columns_equal_completed_run']}**. The differing
selector/grid therefore matters even when the feature bank matches.

<details>
<summary>Expand the complete current configured feature hierarchy</summary>

{tree(current['prepared'])}

</details>

## Reading the hierarchy

- `direct` contains home/away versions of team-history expressions. `for` means
  the focal team's statistic, `against` its opponent's; venue is independent.
- `sum` and `difference` combine same-match home/away historical values.
  `trend` is each perspective's mean3 minus mean20.
- League/LOO windows count completed rounds. H2H windows count prior encounters.
- Warmed columns use the configured prior-seeded EMA or rating transition;
  they do not replace their baseline counterparts. No movement evidence table
  is supplied, so promotion/relegation is not inferred from new appearance.
- Calendar has month sine/cosine, weekday and round. Identity columns represent
  leagues only; there are no team dummy variables.
- Feature selection is fitted inside each training fold. This actual-winner
  inventory describes the final outer-training fit; inner selected sets can differ.
- [Completed grouped JSON](completed_run.json), [current grouped JSON](current_configuration.json),
  and [read-only reproducer](build_inventory.py).
'''
(OUT / 'README.md').write_text(text, encoding='utf-8')
print(json.dumps(dict(actual_columns=len(selected), prepared_columns=len(manifest),
    current_columns=len(columns), current_expressions=len(definitions),
    unclassified_actual=actual['selected']['unclassified_columns'],
    unclassified_current=current['prepared']['unclassified_columns'],
    definitions_match=current['provenance']['expressions_equal_completed_run']), indent=2))
