# Inspect match results and predictions

`MatchResultReporter` turns retained predictions into an offline fixture view.
Each row shows the home and away teams, observed result, prediction and, for a
numeric label, signed error. Colors show agreement. A team-level dataset places
the home and away labels beside each other. Optional probabilities appear in the
same group. Status remains in exported data; the compact fixture view has no
Status column or textual “within tolerance” cells.

Use this after training with an existing `TrainingResult`. The examples below
construct **artificial retained observations and predictions** to demonstrate the
reporter without fitting or calling a model. Team names and optional badges come
from the local catalog; the illustrated fixtures and numbers are synthetic.

See the [API reference](match_results_reference.md),
[equations](match_results_equations.md), [coverage checklist](match_results_documentation_checklist.md)
and [verification record](match_results_check.json). The short
[notebook 12](../../notebooks/12_match_results_quickstart.ipynb) has a
[saved HTML view](../../notebooks/outputs/match_results_quickstart.html).

## 1. Set up an offline demonstration

Run these Python blocks in order from the repository root using the existing
`misc314_py314` environment. Outputs go under `experiment/match_results_demo/guide`.
In normal use, keep the result returned by your training runner and skip the
`retained_result` helper. Its `model=None` is only a placeholder for this display
demonstration. The two artificial folds deliberately repeat the same observations
so the examples can show fold switching and explicit pooling.

```python
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root = Path.cwd()
if root.name == 'notebooks':
    root = root.parent
sys.path.insert(0, str(root / 'src'))
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import MatchResultReporter, TeamCatalog
from xdiyo_analytics.training import FoldResult, TrainingResult

output_dir = root / 'experiment/match_results_demo/guide'
output_dir.mkdir(parents=True, exist_ok=True)
catalog = TeamCatalog.from_json(root / 'docs/analytics/team_assets/catalog.json')
match_columns = ('competition_id', 'season_id', 'event_id')

def retained_result(metadata, observed, outputs, *, layout='match'):
    positions = observed.index.to_numpy()
    folds = [FoldResult(
        fold_id=fold_id, model=None, train_positions=np.array([], dtype=int),
        test_positions=positions.copy(), score_positions=positions.copy(),
        feature_columns=(), target_columns=tuple(observed.columns),
        predictions={name: frame.copy() for name, frame in outputs.items()},
        y_true=observed.copy(), metadata=metadata.copy(),
    ) for fold_id in (0, 1)]
    identity = match_columns + (('team_id',) if layout == 'team_match' else ())
    return TrainingResult(folds, layout, identity, match_columns,
                          'team' if layout == 'team_match' else 'total')
```

## 2. Inspect numeric and categorical labels

For numeric values, error is prediction minus result. With `tolerance=0.25`,
absolute error at or below 0.25 is green; larger errors are red. Missing values
are neutral. `comparison='categorical'` makes integer classes use exact class
agreement. This explicit choice is useful for a custom numeric class definition.

```python
rows = pd.Index(range(6), name='row_position')
metadata = pd.DataFrame({
    'competition_id': [10, 10, 20, 10, 20, 20], 'season_id': 2025,
    'event_id': [101, 102, 103, 104, 105, 106],
    'source_league': ['Demo A', 'Demo A', 'Demo B', 'Demo A', 'Demo B', 'Demo B'],
    'source_season': '2025/26', 'round': [1, 1, 2, 1, 3, 3], 'stage': 'Example',
    'home_id': [1652, 8, 1665, 1652, 11, 24338],
    'away_id': [2671, 11, 24338, 8, 2671, 1665],
}, index=rows)
observed = pd.DataFrame({'corners': [8., 10., 6., 9., 5., 7.],
                         'over_line': [0, 1, 0, 1, 0, 0]}, index=rows)
predicted = pd.DataFrame({'corners': [8., 9.75, 6.5, 8.5, np.nan, 7.1],
                          'over_line': [0, 1, 1, 0, 0, 0]}, index=rows)
training = retained_result(metadata, observed, {'predict': predicted})
report = PostTrainingAnalysis({
    'Corner totals': MatchResultReporter(type='per_fold', target='corners',
        tolerance=.25, catalog=catalog, show_badges=True, page_size=2),
    'Custom integer classes': MatchResultReporter(type='per_fold', target='over_line',
        comparison='categorical', catalog=catalog, show_badges=True, page_size=2),
}, title='Synthetic fixture inspection').run(training)
report.to_html(output_dir / 'fixtures.html')
```

Open [the generated fixture report](../../experiment/match_results_demo/guide/fixtures.html).
Use its fold selector, League/Season/Team/Round controls, sorting and pagination.
Team filtering matches either side using the stable team ID. Two teams may have
the same display name without becoming the same identity. `target=None` produces
one panel per target; a sequence selects a subset in that order. Use separate
reporters when targets need different comparison settings, as above.

## 3. Start filtered and keep the complete population

`None` means All for each initial filter. Zero is a real selection. With repeated
held-out observations, an overall view requires an explicit pooling choice.
Here `first` keeps the first selected fold's prediction for each row position.

```python
filtered = PostTrainingAnalysis({
    'Troyes in round one': MatchResultReporter(type='overall', pooling='first',
        target='corners', tolerance=.25, catalog=catalog, show_badges=True,
        league='Demo A', season='2025/26', team=1652, round=1, page_size=1),
}).run(training)
filtered.to_html(output_dir / 'initially_filtered.html')
full_table = filtered.studies[0].result.tables['matches::corners']
print(len(full_table))  # 6 observations retained; the initial display matches 2.
```

The [filtered view](../../experiment/match_results_demo/guide/initially_filtered.html)
starts with two fixtures and one fixture per page. **Download filtered CSV**
exports both matching observations across all pages. The Data table and its CSV
retain all six. Reset filters restores the whole scope. Filtering, sorting and
pagination never fit a model or recalculate predictions.

## 4. Put home and away observations on one fixture row

For `team_match`, each input row is a team observation. Metadata supplies `side`,
`team_id` and `opponent_id`; rows are paired by the declared match identity and
fold. The table displays independent home and away Result/Prediction/Error groups.

```python
paired_meta = metadata.iloc[[0, 0, 1, 1, 2, 2]].reset_index(drop=True)
paired_meta.index.name = 'row_position'
paired_meta['side'] = ['home', 'away'] * 3
paired_meta['team_id'] = [1652, 2671, 8, 11, 1665, 24338]
paired_meta['opponent_id'] = [2671, 1652, 11, 8, 24338, 1665]
paired_y = pd.DataFrame({'corners': [3., 5., 6., 2., 4., 1.]}, index=rows)
paired_prediction = pd.DataFrame({'corners': [3., 5.4, 5.8, 2., 4., np.nan]}, index=rows)
paired_training = retained_result(paired_meta, paired_y,
    {'predict': paired_prediction}, layout='team_match')
paired_report = PostTrainingAnalysis({
    'Corners by team': MatchResultReporter(type='per_fold', tolerance=.25,
        catalog=catalog, show_badges=True, page_size=2),
}).run(paired_training)
paired_report.to_html(output_dir / 'paired.html')
```

The [paired report](../../experiment/match_results_demo/guide/paired.html) colors
each side independently. Any incorrect side makes the row marker red. Both sides
must be present and correct for a green row marker; otherwise it is neutral.
A missing side displays dashes. Duplicate same-side rows or conflicting fixture
context are errors. The CSV still has one row per team observation.

## 5. Show probabilities and explicitly choose a decision rule

An output with `(target, class)` columns is a probability output. It shows the
class probabilities and leaves the Prediction blank by default. `argmax` and a
binary threshold are optional, explicit decisions. Their colors compare the
chosen class with the observed class. The threshold uses an inclusive boundary.

```python
probability_y = pd.DataFrame({'won': [1, 0, 1, 0, 1, 0]}, index=rows)
probabilities = pd.DataFrame([
    [.7, .3], [.2, .8], [.5, .5], [np.nan, np.nan], [.6, .4], [.4, .6],
], index=rows, columns=pd.MultiIndex.from_product([['won'], [1, 0]], names=['target', 'class']))
probability_training = retained_result(metadata, probability_y,
    {'predict_proba': probabilities})
probability_options = dict(type='overall', pooling='first', output='predict_proba', catalog=catalog)
probability_report = PostTrainingAnalysis({
    'Probabilities only': MatchResultReporter(**probability_options),
    'Largest probability': MatchResultReporter(**probability_options, decision='argmax'),
    'Positive probability at least 0.6': MatchResultReporter(**probability_options,
        threshold=.6, positive_class=1),
}).run(probability_training)
probability_report.to_html(output_dir / 'probabilities.html')
```

In the [probability report](../../experiment/match_results_demo/guide/probabilities.html),
the tied row selects class 1 under argmax because class 1 is stored first.
The 0.6 row selects class 1 at the threshold boundary. The missing vector stays
neutral. Finite values in `[0,1]` must sum to one within absolute tolerance
`1e-6`; invalid vectors are unavailable and are never renormalized. Probability
display does not imply that a regression model provides probabilities.

## 6. Supply names or optional local badges

Catalogs map IDs to a name string or a record with `name` and optional
`badge_path`. A supplied nonempty catalog name takes precedence over metadata;
otherwise metadata names are used, then `Team <ID>`. Names are independent of
badge availability. JSON badge paths resolve beside the catalog file.

```python
simple_catalog = TeamCatalog({1652: 'Troyes', 2671: {'name': '1. FC Köln'}})
named_matches = metadata.copy()
named_matches['home_name'] = [catalog.display(key)[0] for key in metadata.home_id]
named_matches['away_name'] = [catalog.display(key)[0] for key in metadata.away_id]
names_from_matches = TeamCatalog.from_matches(named_matches)
print(simple_catalog.display(1652))  # ('Troyes', None, None); no badge read.
```

The [optional asset catalog](team_assets/README.md) contains 289 named IDs and
286 local badges. Three IDs have names only. `show_badges=False` is the default;
when enabled, accepted bytes are embedded once per team per reporter execution.
Saved reports work offline. Missing, oversized or rejected files leave the name
and a diagnostic note. The catalog is repository data, separate from the wheel.

## Notebook use and boundaries

Call `report.to_notebook()` in a trusted HTML-capable notebook to display the
same report in an isolated iframe. `to_html(path)` saves a standalone view.
Notebook 12 was executed in a fresh kernel and its saved iframe interactions were
checked. Live Jupyter frontend behavior remains unverified.

Reporting consumes the retained test or score population. It does not choose
training rows, tune a model, fill missing results, infer historical crest versions
or certify predictive performance. The [reference](match_results_reference.md)
describes scope, schema, fallback rules and rendering helpers in detail.
