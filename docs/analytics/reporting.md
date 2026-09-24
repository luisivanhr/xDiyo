# Inspect data and compose feature selections before training

`PreTrainingAnalysis` runs the studies you name against explicit row scopes.
Its `AnalysisReport` contains reusable numerical tables, selected feature names
and an interactive offline viewer. Running or displaying a report does not fit
a predictive model. An empty reporter mapping produces an empty report.

Start with the [minimal notebook](../../notebooks/08_reporting_quickstart.ipynb).
The [API and helper reference](reporting_reference.md) describes every setting;
the [coverage checklist](reporting_documentation_checklist.md) records verification.

## Prepare an explicit dataset and split plan

This example uses two pinned Premier League publications. Corners provide a small
working example of the reusable interfaces; this does not select a pilot recipe.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import IsHome, Lag, Stat, evaluate_features
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.splits import TemporalSplit, create_split_plan
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import (
    FeatureDistributionReporter, CorrelationAnalysis, FeatureTimeline,
    TopKCorrelationSelector, FeatureSelector, FeatureSelection, vote_selections,
    Artifact, StudyResult,
)

root = Path('C:/Users/luisi/Documents/Programming/Python/xDiyo')
loaded = load_seasons(
    root / 'data/xDiyo_data', ['23_24', '24_25'], leagues='Premier_League',
    tables=['matches', 'statistics'], verify_hashes=True,
    record_dir=root / 'experiment/initial_population/selections',
)
history = build_team_history(select_stats(
    loaded, stats=[('ALL', 'Match overview', 'cornerKicks')],
))
corners = Stat('ALL', 'Match overview', 'cornerKicks')
features = evaluate_features(history, {
    'venue': IsHome(), 'previous': Lag(corners), 'previous_2': Lag(corners, 2),
}, keyed=True)
labels = create_labels(history, {'corners': MatchTotal(corners)})
dataset = assemble_dataset(features, labels['corners'], layout='match')
plan = create_split_plan(dataset, TemporalSplit(
    20, test_size=8, step=20, allow_partial_test=True,
))
```

The dataset declares `match` or `team_match`. Reporters count both observations
and distinct match tuples. They give every supplied row equal statistical weight:
two team rows are two observations, even when a feature happens to be identical.
Display grouping does not change that statistical unit. Inputs must retain their
original aligned order after the split plan is created.

## Choose execution and partition separately

Every reporter requires `type` and `partition`; there is no implicit study set.

| Execution type | `all` | `train`, `test` or `score` |
| --- | --- | --- |
| `per_fold` | That fold's train/test union; gaps remain excluded. | That fold's named positions. |
| `overall` | All supplied dataset rows. | Unique union across the selected folds, in original input order. |
| `timeline` | All dataset rows, sorted by UTC kickoff. | The same unique union, sorted by UTC kickoff. |

Only `FeatureTimeline` and suitably declared custom reporters support `timeline`.
Without a plan, only `overall/all` or supported `timeline/all` can run. `fold_ids`
selects distinct zero-based folds; omitted means every fold. An empty selected
fold list cannot serve a scope that requires folds. A fold's empty `score`
partition remains an empty context. Equal kickoff times retain original order.

For selected folds \(\mathcal F\), overall training uses the population

\[
U_{\mathrm{train}}=\bigcup_{f\in\mathcal F}T_f.
\]

An observation appearing in several training folds contributes once to this
overall population. Overall coefficients are calculated from these rows together;
they are not averages of fold coefficients. Fold consensus is a separate operation.

## Compose named studies and a training selection

Dependencies appear before their consumers. `source` reuses already computed
correlations for identical rows, fold and partition. `features_from` applies an
earlier selection's column names to a later reporter's own row scope.

```python
report = PreTrainingAnalysis({
    'training distributions': FeatureDistributionReporter(
        type='per_fold', partition='train',
        features=['home::previous', 'away::previous'], bins=10,
    ),
    'training correlations': CorrelationAnalysis(
        type='per_fold', partition='train', methods=['pearson', 'spearman'],
    ),
    'selected in each fold': TopKCorrelationSelector(
        type='per_fold', partition='train', k=2, source='training correlations',
    ),
    'test feature distributions': FeatureDistributionReporter(
        type='per_fold', partition='test', features_from='selected in each fold',
    ),
    'pooled training correlations': CorrelationAnalysis(
        type='overall', partition='train', methods='spearman',
    ),
    'fold consensus': TopKCorrelationSelector(
        type='overall', partition='train', across_folds=True,
        k=2, fold_k=1, source='training correlations',
    ),
}, title='Descriptive pre-training inspection').run(
    dataset, split_plan=plan, fold_ids=[0, 1],
)
```

The test distributions use names chosen from each fold's training rows. They do
not recalculate selection scores from test labels. A later reporter without
`features_from` still receives all input features. An explicitly named overall
selection can be applied to folds, but such reuse does not create a nested
evaluation. A matching same-fold selection takes precedence over an overall one.

### Retained studies from fitted preparation

When these reporters are supplied as a candidate's `pre_analysis`, their results
are retained in `training.fitted_report`. This includes descriptive correlation
studies as well as selectors, even when no selector is consumed by the candidate.
The combined `ExperimentResult.report` displays them with the `fitted/` prefix
alongside `pre/` exploration and `post/` diagnostics.

These studies use the candidate's actual fitting rows. If a validation tail is
reserved, validation rows are excluded along with held-out test rows. Saved row
positions refer to the original assembled dataset, including after selection
works on a local subset. Retained study objects are separate from the selection
record attached to a fitted model.

For a holdout search, the final report includes preparation studies recalculated
for the selected candidate's outer fit. For nested selection, it includes the
evaluated winner in each outer fold, with a fold selector for those outer folds.
Even an `overall` study inside one such fit is shown as an outer-fold-specific
view; its scope label records that original meaning. Inner-trial studies remain
in trial evidence and are not repeated in the final report.

New recovery bundles retain these studies and their tables without refitting.
Older bundles lacking `fitted_report` still load with that field set to `None`;
loading alone cannot reconstruct diagnostics that were never saved.

```python
selected_columns = {
    fold: selection.columns
    for fold, selection in report.selections['selected in each fold'].items()
}
selected_columns
```

Selections retain complete ranking tables. No feature columns are removed from
the source dataset. If no scores are defined, selection returns an empty tuple;
downstream built-in reporters show a no-features note when their feature selector
is left as `None`.

## Include observed labels in distribution studies

`FeatureDistributionReporter` can plot selected observed label columns alongside
the selected features:

```python
FeatureDistributionReporter(
    type='per_fold', partition='train',
    features=['home::previous', 'away::previous'],
    targets=['corners'], bins=10, bandwidth='scott',
)
```

`features=None` selects all input features. Labels are opt-in:
`targets=None` omits label plots, while `targets=['corners']` includes that
observed label. In the builder this control is named **Labels**. An enabled
empty selection (`[]`) is an error; disable the control to use its default.
Run reports the affected reporter and field before preparation or fitting,
while Prepare/Discover remain available to load the choices.

Feature and label plots use the same selected rows, fold/overall scope,
partition, histogram bins and KDE settings. Each variable has its own plot;
label titles start with **Label ·**. A feature and label sharing a name remain
separate. Existing `summary`, `histogram` and `kde` tables retain their feature
schemas; `label_summary`, `label_histogram` and `label_kde` use a `label` column.
Labels can also be plotted when the supplied context has no feature columns.

Nonfinite and nonnumeric values are excluded separately for each variable,
with finite/excluded counts retained. Constant labels or fewer than two finite
values receive a histogram and an unavailable-KDE note; a label with no finite
values receives an explanatory plot annotation. The KDE of a discrete label is
a visual density guide. This study displays observed labels, not predictions;
prediction-versus-label distributions belong to post-training diagnostics.

## Display once, navigate without recalculation

```python
html_path = root / 'notebooks/outputs/reporting_guide.html'
html_document = report.to_html(html_path)
assert html_path.exists()
```

The standalone file embeds Plotly once and works offline. Use the sidebar/search,
fold selector, collapsible studies and artifact panels, sortable headers, and
entity dropdowns. Chart controls support hover, zoom and SVG export. CSV links
include every computed table row and original scope position, even when a table
preview stops at 200 rows. Sorting preserves exact integer IDs.

In a trusted notebook, leave `report` as the last expression of a code cell for
automatic rich display. Explicit display is also available:

```python
from IPython.display import IFrame
inline_view = report.to_notebook(height=700)
assert isinstance(inline_view, IFrame)
# In a notebook, display inline_view, or call report.show(height=700).
```

The native `IFrame` embeds the same document through `srcdoc`, isolating its CSS
and JavaScript from other cells. Rendering does not rerun analysis. Embedded
Plotly makes saved notebook output larger. Frontends must trust and permit
HTML/JavaScript; restrictive viewers can use the standalone HTML export.

## Interpret distributions

Non-numeric and nonfinite values are excluded with explicit counts. For \(n\)
finite observations and bin \([b_j,b_{j+1})\), with the final right edge included,
the displayed height is

\[
d_j=\frac{n_j}{n(b_{j+1}-b_j)},\qquad
\sum_jd_j(b_{j+1}-b_j)=1.
\]

Explicit edges must cover all finite values and increase strictly. Unequal widths
are respected. These are densities, not heights equal to bin probabilities.
[NumPy histogram definition](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html).

The Gaussian overlay is

\[
\widehat f(x)=\frac{1}{nh\sqrt{2\pi}}
\sum_{i=1}^{n}\exp\!\left(-\frac{(x-x_i)^2}{2h^2}\right),
\qquad h=b\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}.
\]

Scott uses \(b=n^{-1/5}\); SciPy's Silverman choice uses
\(b=(3n/4)^{-1/5}\). A positive finite scalar supplies the covariance factor
directly. `None` also means Scott. The finite display grid spans the histogram
edges, so its visible KDE area need not equal one. The estimator itself extends
beyond that grid. Fewer than two observations or constant values omit the KDE;
singular covariance is recorded. A smooth curve over count data is a visual
guide, not a discrete probability mass function.
[SciPy Gaussian KDE](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).

## Interpret association columns

Each numeric feature/target pair uses its own jointly finite observations.
The `coefficients` table retains `n` and `status` for every requested cell.
Constants and insufficient pairs produce missing estimates; missing cells never
become zero. No p-values, causal effects or model performance are reported.

Pearson is

\[
r=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}}.
\]

Spearman applies this expression to average tied ranks of each variable.
[Pearson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html),
[Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
Kendall tau-b counts concordant pairs \(P\), discordant pairs \(Q\), and ties
only in each variable \(T_x,T_y\):

\[
\tau_b=\frac{P-Q}{\sqrt{(P+Q+T_x)(P+Q+T_y)}}.
\]

Pairs tied in both variables contribute to neither tie-only count.
[SciPy Kendall tau-b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).

MCC requires both columns to be explicitly categorical or thresholded. Thresholds
use strict greater-than; equality maps to zero. Native classes need deliberate
correspondence between feature and target labels. Relabelling one side can change
MCC. There is no automatic binning or learned threshold.

```python
mcc_report = PreTrainingAnalysis({
    'explicit binary association': CorrelationAnalysis(
        type='overall', partition='all', features=['home::previous'], methods='mcc',
        feature_thresholds={'home::previous': 4},
        target_thresholds={column: 8 for column in dataset.y.columns},
    ),
}).run(dataset)
```

For binary counts,

\[
\mathrm{MCC}=\frac{TP\,TN-FP\,FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
\]

For a shared multiclass vocabulary, let \(C_{ij}\) count paired classes,
\(p_i=\sum_jC_{ij}\), \(t_j=\sum_iC_{ij}\), \(s=\sum_{ij}C_{ij}\), and
\(c=\sum_iC_{ii}\). Then

\[
\mathrm{MCC}=\frac{cs-\sum_kp_kt_k}
{\sqrt{(s^2-\sum_kp_k^2)(s^2-\sum_kt_k^2)}}.
\]

This library keeps a zero denominator missing. The independent sklearn
comparison applies to nondegenerate cases.
[MCC reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html).

### Percentiles and pooled magnitude

Percentile means a feature's rank by absolute association within one target/metric
column. It is a descriptive ordering, not a fifth estimator. Among \(m\) finite
coefficients, if \(L_j\) are smaller and \(E_j\) equal in magnitude, then

\[
\mathrm{percentile}_j=100\frac{L_j+(E_j+1)/2}{m}.
\]

Leaderboard order uses the sum of absolute coefficients over selected target and
method columns:

\[
A_j=\sum_{(t,m)\in D_j}|r_{jtm}|,
\]

where \(D_j\) contains defined cells. `n_coefficients` records coverage;
\(A_j\) remains missing when \(D_j\) is empty. Unequal coverage can affect order.
Green/coral bars retain coefficient sign while ordering uses magnitude.

## Three selection modes share one contract

`FeatureSelector` supplies the execution rules independently of the selection
algorithm: local per-fold selection, one calculation over pooled unique rows,
or separate local selections followed by an explicit consensus policy.

`TopKCorrelationSelector` uses one requested metric and sums absolute values
across selected target columns. It selects up to the requested number of defined scores; ties retain
requested/input feature order. Reusing a source with missing pairs, targets or
the wrong metric fails explicitly.

### Counts or proportions

Both `k` and consensus `fold_k` accept a positive whole-number count or a
proportion strictly between zero and one. `1` and `1.0` both mean **one feature**,
not 100%. For an input population of \(n\) feature columns, the requested count is

\[
K(k,n)=
\begin{cases}
\lceil kn\rceil, & 0<k<1,\\
k, & k\in\{1,2,3,\ldots\}.
\end{cases}
\]

For example, `k=0.5` requests three of five inputs, and `k=0.14` requests seven
of fifty. Decimal multiplication avoids accidentally rounding that latter
request up to eight because of floating-point representation. Zero, negative
values, booleans, nonfinite values, and non-whole values greater than one are
invalid.

The denominator is the input feature count **after** `features_from` and an
explicit `features` subset have restricted the population. It includes inputs
whose association is undefined; those inputs count toward the request but are
ineligible to be selected. The selector returns fewer features when too few
have defined scores. Thus a chained 50% selection applies to the first
selector's output, not the original dataset. Empty input stays empty.

```python
TopKCorrelationSelector(type="per_fold", partition="train", k=0.25)
TopKCorrelationSelector(
    type="overall", partition="train", across_folds=True,
    k=0.5, fold_k=0.2,
)
```

The second example asks each fold to nominate 20% of its supplied inputs,
rounded up, then asks the consensus for 50% of its final input population.
Only nominated features can survive, so a consensus may return fewer than its
requested count. `fold_k=None` uses `k` independently for each fold.

Custom selectors can opt into the public
`resolve_selection_count(k, n_features)` helper. Their own `k` arguments do not
automatically gain this behavior merely by inheriting `FeatureSelector`.
`vote_selections` uses the helper for count/proportion requests and also accepts
`k=None` to retain all nominees.

For consensus, each fold nominates up to `fold_k` features (`None` uses `k`). With
nomination sets \(N_f\), feature votes and score tie-breaks are

\[
v_j=\sum_{f\in\mathcal F}\mathbf 1\{j\in N_f\},\qquad
\bar A_j=\frac{1}{|F_j|}\sum_{f\in F_j}A_{fj},
\]

where \(F_j\) includes every fold with a finite score for that feature, including
folds that did not nominate it. Sort by decreasing votes, then decreasing mean
score, then input order. Only nominated features can be selected. Vote fraction
is \(v_j/|\mathcal F|\). Repeated observations in overlapping folds cast repeated
votes intentionally; votes are not independent evidence.

The generic `vote_selections` helper can omit numerical scores entirely. Custom
selectors may also implement another `combine` policy. This target-independent
example selects columns with enough finite observations:

```python
from dataclasses import dataclass

@dataclass(kw_only=True)
class CoverageSelector(FeatureSelector):
    min_present: int = 20

    def select(self, context):
        finite = context.X.apply(pd.to_numeric, errors='coerce').replace(
            [np.inf, -np.inf], np.nan,
        ).notna().sum()
        ranking = finite.rename('present').rename_axis('feature').reset_index()
        columns = tuple(finite.index[finite >= self.min_present])
        return StudyResult('Feature coverage', tables={'coverage': ranking},
                           selection=FeatureSelection(columns, ranking))

    def combine(self, context, fold_results):
        selection = vote_selections(
            fold_results, feature_order=context.X.columns, k=None,
        )
        return StudyResult('Coverage nominations', selection=selection)

coverage = PreTrainingAnalysis({
    'local': CoverageSelector(type='per_fold', partition='train'),
    'pooled': CoverageSelector(type='overall', partition='train'),
    'consensus': CoverageSelector(type='overall', partition='train', across_folds=True),
}).run(dataset, split_plan=plan, fold_ids=[0, 1])
```

Override `for_fold()` when local configuration differs from final configuration.
Local children receive their own fold metadata, definitions and previous-result
copies. Shared mutable state in custom reporter objects remains the author's
responsibility; the framework isolates the supplied contexts.

## Follow team or league timelines

```python
timeline = PreTrainingAnalysis({
    'team history': FeatureTimeline(
        type='timeline', partition='all', features='home::previous',
        teams=sorted(set(dataset.metadata.home_id))[:2], max_points=80,
    ),
}).run(dataset)
```

Team-match rows use `team_id`. Match features prefixed `home::` or `away::` map
to the corresponding team ID, or supply `team_column` explicitly. Grouping uses
competition plus team identity across seasons. Missing values remain gaps;
there is no fill or smoothing. `max_points` samples only the display, retaining
endpoints and extra missing breakpoints, so the display can exceed that limit.
Full point tables remain downloadable.

League mode collapses agreeing finite values at each competition/kickoff. For
different team/LOO values, choose team mode or explicitly select `mean`/`median`:

```python
league_view = PreTrainingAnalysis({
    'mean of supplied home values': FeatureTimeline(
        type='timeline', partition='all', entity='league',
        features='home::previous', league_aggregation='mean',
    ),
}).run(dataset)
```

This example averages the selected rows at a kickoff. It does not calculate a
season/round league benchmark. With no aggregation, missing values are ignored
when checking agreement; an entirely missing coordinate stays missing.

## Add a reporter with its own artifact arrangement

```python
@dataclass
class ScopeSummary:
    type: str
    partition: str
    supported_types = ('per_fold', 'overall', 'timeline')

    def run(self, context):
        table = pd.DataFrame({'rows': [len(context.X)], 'matches': [context.n_matches]})
        return StudyResult('Population',
            artifacts=[Artifact('text', 'Counts follow the declared dataset unit.'),
                       Artifact('table', table, 'Scope counts')],
            tables={'counts': table})

summary = PreTrainingAnalysis({'population': ScopeSummary('overall', 'all')}).run(dataset)
empty_report = PreTrainingAnalysis({}).run(dataset)
assert empty_report.studies == []
```

Subclassing or registration is unnecessary for reporters. Built-in artifact kinds
are `text`, `table`, `plotly`, `leaderboard` and trusted `html`. Additional kinds
use `report.to_html(renderers={kind: callable})`; that callable receives an
`Artifact` and returns HTML. Ordinary labels/text are escaped. HTML artifacts and
custom renderer output are trusted local code.

## Limits before a predictive experiment

These are descriptive summaries of supplied features and targets. Selection on
overall data or across these folds needs separate outer evaluation. Statistical
dependence between team rows or overlapping folds is not removed by sorting,
deduplication of repeated positions, or percentile ranks.

Reports do not reconstruct rolling/Glicko states separately for each fold.
Retrospective CV still needs an explicit rule for held-out information carried
into later features. Model adapters, training, calibration, post-training metrics
and multiplicity corrections remain separate work. The report/artifact contracts
can support later post-training consumers without introducing a model dependency.
