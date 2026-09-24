# Choose whole-match splits and inspect held-out rows

The `xdiyo_analytics.splits` module prepares fold membership from a `ModelDataset`.
It supports chronological windows, match/group K-fold and retrospective CPCV.
It returns positions for the original dataset; it does not train, search, refit,
recompute features or calculate statistical/strategy reports.

Use the [minimal notebook](../../notebooks/07_splits_quickstart.ipynb),
[complete API/helper reference](splits_reference.md) and
[coverage checklist](splits_documentation_checklist.md).

## Load and assemble a pinned example

```python
from pathlib import Path
import numpy as np
import pandas as pd
from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import IsHome, Lag, Stat, evaluate_features
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.splits import (
    TemporalSplit, MatchKFold, GroupKFold, CPCV,
    create_split_plan, reconstruct_paths,
)

root = Path('C:/Users/luisi/Documents/Programming/Python/xDiyo')
loaded = load_seasons(
    root / 'data/xDiyo_data', ['23_24', '24_25'], leagues='Premier_League',
    tables=['matches', 'statistics'],
    record_dir=root / 'experiment/initial_population/selections', verify_hashes=True,
)
history = build_team_history(select_stats(
    loaded, stats=[('ALL', 'Match overview', 'cornerKicks')],
))
corners = Stat('ALL', 'Match overview', 'cornerKicks')
features = evaluate_features(
    history, {'venue': IsHome(), 'previous_corners': Lag(corners)}, keyed=True,
)
labels = create_labels(history, {'corners': MatchTotal(corners)})
dataset = assemble_dataset(features, labels['corners'], layout='match')
```

The same splitters accept the assembler's `team_match` layout. Both team rows
always travel together. Exact match tuples distinguish repeated event IDs across
partitions. Match tuples identify observations/groups; they are not ranker group
sizes. See the [assembly contract](datasets_reference.md).

## Inspect expanding chronological folds

```python
scheme = TemporalSplit(
    train_size=20, test_size=6, score_start=1, allow_partial_test=True,
)
plan = create_split_plan(dataset, scheme)
fold_counts = pd.DataFrame([
    {'fold': i, 'train': len(f.train), 'test': len(f.test), 'score': len(f.score),
     'fit_at': f.metadata['fit_at']}
    for i, f in enumerate(plan.folds)
])
fold_counts.head()
```

Here `train_size` counts the initial observed round blocks, `test_size` counts
held-out blocks, and the default `step=None` advances by `test_size`. Expanding
windows keep the beginning of the calendar. `window='sliding'` keeps only the
most recent `train_size` candidate blocks. Gaps and timing checks can reduce the
actual training population below the candidate span. Final short tests are
omitted unless `allow_partial_test=True`.

Round blocks default to `(season_id, round)` and are ordered by their earliest
observed kickoff, even across seasons. Missing rounds are not fabricated.
Postponed fixtures in an earlier nominal round are excluded from training if
their actual supplied kickoff/availability is too late. Such a fixture can
become eligible in a later fold while its block remains in the training span.

## Slice with original positions

```python
fold = plan.folds[0]
train_X = dataset.X.iloc[fold.train]
test_X = dataset.X.iloc[fold.test]
score_y = dataset.y.iloc[fold.score]
assert set(fold.train).isdisjoint(fold.test)
assert set(fold.score) <= set(fold.test)
dataset.metadata.iloc[fold.test].head()
```

`train`, `test` and `score` are integer positions, not DataFrame index labels.
They refer to the input dataset's order, including when its rows are shuffled.
Keep `X`, `y` and metadata aligned and use `.iloc`. Returned membership arrays
are sorted by original position; `plan.row_order` separately records chronological
positions, stable for kickoff ties. A plan cannot detect a later same-length
reordering of your input.

## Separate held-out membership from scoring

Let \(T_f\), \(H_f\) and \(S_f\) be a fold's training, held-out test and scoring
position sets. Then

\[
T_f\cap H_f=\varnothing,\qquad S_f\subseteq H_f,\qquad
(H_f\setminus S_f)\cap T_f=\varnothing.
\]

`score_start` skips that many held-out blocks for scoring only. `score_rounds`
selects an inclusive round-number range in every held-out season; either bound
can be `None`. The two rules intersect. An empty score set is allowed, including
when a short final horizon contains only skipped blocks.

```python
later_season_score = create_split_plan(
    dataset, TemporalSplit(1, unit='seasons', score_rounds=(6, None)),
)
season_fold = later_season_score.folds[0]
assert dataset.metadata.iloc[season_fold.score]['round'].min() >= 6
assert set(season_fold.test) - set(season_fold.score)
assert set(season_fold.train).isdisjoint(season_fold.test)
```

That example holds out the next season and scores its later rounds. The early
test-season matches remain held out. This choice differs from a block `gap`
between candidate training and testing, a `gap_time` before the fitting boundary,
or feature-owned warm-up/history rules. The splitter does not change feature
histories or create a refit schedule.

## State prediction and availability times

For match \(m\), let \(t_m\) be kickoff, \(c_m\) the earliest supplied prediction
cutoff among its rows, and \(a_m\) the latest availability among its rows.
Any missing explicit availability makes that match unavailable for training.
For held-out matches \(\mathcal H_f\), candidate training matches \(\mathcal C_f\),
duration gap \(\delta\), and complete-target indicator \(O_m\), the actual rule is

\[
u_f=\min_{m\in\mathcal H_f}c_m,\qquad b_f=u_f-\delta,
\]
\[
\mathcal T_f=
\{m\in\mathcal C_f:t_m<b_f,\ a_m\le b_f,\ O_m=1\}.
\]

Completeness means every selected target cell in every row of the match is
nonmissing. Missing feature values do not filter membership. Test/scoring targets
are retained even when unknown. Training at the same kickoff as the boundary is
excluded; availability at the boundary is permitted if kickoff is earlier.

```python
prediction_cutoffs = dataset.metadata['kickoff_at'] - pd.Timedelta('2D')
lead_plan = create_split_plan(
    dataset, TemporalSplit(20, test_size=6, allow_partial_test=True),
    cutoffs=prediction_cutoffs,
)
lead_plan.folds[0].metadata['excluded_train'][:10]
```

Both timing arguments accept a metadata column name or row-aligned datetime
values. A Series must have the metadata index/order. Inputs are converted to UTC;
numeric epochs need explicit conversion first. `cutoffs=None` uses kickoff.
`available_at=None` also uses kickoff as a retrospective proxy; it does not
establish historical result publication or completion. When verified availability
exists, pass it through `available_at`. Missing explicit availability excludes
training matches, including the all-missing case; empty training raises an error.

The split cutoff does not retrospectively recompute a feature evaluated at a
later cutoff. Feature construction must use compatible issuance and availability
rules before a chronological evaluation can make a deployment claim.

## Choose calendars deliberately

`unit='seasons'` pools leagues by the shared `source_season` label. Different
native `season_id` values for the same source season therefore belong to one
season block. `unit='league_seasons'` uses separate `competition_id` calendars
and native `season_id` blocks. **Migration:** use `league_seasons` to retain the
former separate-league behavior of `seasons`.

Rounds still default to separate competition calendars. Kickoff mode defaults
to one pooled calendar, with simultaneous matches in the same block.
`calendar_by=()` explicitly pools calendars. A name or tuple of names can select
another grouping; `block_by` replaces the default block identity. Explicit
overrides take precedence over the unit's defaults.

```python
pooled_kickoffs = create_split_plan(
    dataset, TemporalSplit(20, test_size=8, step=50, unit='kickoffs'),
)
explicit_rounds = create_split_plan(
    dataset, TemporalSplit(
        20, test_size=6, calendar_by=('competition_id',),
        block_by=('season_id', 'round'),
    ),
)
assert explicit_rounds.get_n_splits() > 0
```

For multi-stage competitions, include the actual stage column in `block_by`.
For synchronized cross-league rounds, supply deliberate shared season/round
identifiers; unrelated season IDs or round numbers do not define a shared
calendar. Separate-calendar folds evaluate only that calendar's matches.
Their order follows first calendar appearance, then that calendar's block order;
the complete fold list is not globally interleaved by fitting time.

## Use a round gap inside a season test window

```python
scheme = TemporalSplit(
    train_size=4, test_size=1, unit='seasons',
    gap=10, gap_unit='rounds',
)
```

This trains on four pooled seasons and tests the next season after excluding
its first ten observed round blocks **separately in each league**. With ordinary
round numbering, testing starts at round 11 and ends at that season's end.
The gap matches do not enter training. A postponed match belonging to an excluded
early round stays excluded even if played after round 11. Missing round numbers
are not fabricated: the gap counts observed blocks, not a numerical round label.
Use `unit='league_seasons'` for the same operation in separate league folds.

`gap_unit=None` inherits `unit`: it retains the ordinary behavior of skipping
whole window blocks between training and testing. Setting a finer `gap_unit`
trims the beginning of the nominal test span instead; it does not move the
nominal season end or extend training. `gap_unit='kickoffs'` skips the first
unique kickoff batches pooled within the current calendar's test span (unlike
the round gap, which counts separately per competition). A gap that
consumes the entire test span raises a clear error.

The fitting time is the earliest prediction cutoff among the **retained** test
matches. Existing kickoff, target completeness and result-availability checks
still filter the candidate training rows. `score_start` and `score_rounds` can
further restrict scoring, but scoring always stays within the retained test
rows. Feature histories and warm-up remain governed by feature construction;
this gap does not reset them or add a refitting schedule.

## Retrospective match and group K-fold

```python
match_plan = create_split_plan(dataset, MatchKFold(5, shuffle=True, random_state=41))
season_plan = create_split_plan(dataset, GroupKFold(2))
external_groups = pd.Series(
    list(zip(dataset.metadata.source_league, dataset.metadata.source_season)),
    index=dataset.metadata.index, dtype=object,
)
external_plan = create_split_plan(dataset, GroupKFold(2), groups=external_groups)
assert sum(len(f.test) for f in match_plan.folds) == len(dataset.X)
```

`MatchKFold` partitions unique matches in first-appearance order, optionally
shuffling reproducibly. `GroupKFold` defaults to competition-season tuples.
It keeps every group in one test fold and balances match counts greedily, largest
groups first. Shuffling changes equal-size tie order only; it is not sklearn's
exact allocation rule. External hashable tuple groups are supported and must
agree across rows of a match. The [sklearn GroupKFold reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
describes the general non-overlapping-group interface.

These schemes are retrospective: their training data can come after held-out
data. Neither K-fold family automatically filters missing targets or applies
temporal availability checks. Group separation alone does not establish an
out-of-team or past-only forecasting experiment.

## Combinatorial purging and embargo

CPCV splits the sorted unique kickoff batches into `n_blocks` consecutive blocks.
Block sizes differ by at most one **batch**, with larger blocks first; match/row
counts can differ more because ties stay together. With \(N\) blocks and \(k\)
held-out blocks per fold,

\[
F=\binom{N}{k},\qquad
\phi=\frac{k}{N}\binom{N}{k}=\binom{N-1}{k-1}.
\]

Each block occurs in \(\phi\) held-out folds. Assigning its successive fold
occurrences to successive paths yields one copy of each block in every path.
For \(N=6\), \(k=2\), this gives 15 folds and five paths. This is the combinatorial
construction in [López de Prado's exposition, pages 13–14 and Exhibit 8](https://smallake.kr/wp-content/uploads/2018/07/SSRN-id3104816.pdf).

For a match's information interval \(I_i=[s_i,e_i]\), purge a candidate training
match if it overlaps any held-out interval, including touching endpoints:

\[
I_i\cap I_j\ne\varnothing
\quad\Longleftrightarrow\quad s_i\le e_j\ \text{and}\ s_j\le e_i.
\]

Disjoint held-out intervals retain their intervening gaps. For each contiguous
run \(R\) of held-out block numbers, let \(E_R\) be the latest interval end in
that run. A duration embargo \(\eta\) additionally removes candidate match \(i\)
when

\[
E_R<t_i\le E_R+\eta
\quad\text{for at least one held-out run }R.
\]

Within paired rows, starts use the minimum and ends the maximum. Default starts
and ends are point intervals at kickoff, and `embargo=None` is zero duration.
Use `information_start` and `available_at` with justified information/label
times for a substantive purging experiment. This implementation uses durations,
not a finance-specific percentage of price bars. See the broader
[purging/CPCV methodology](https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html).

```python
cpcv = CPCV(n_blocks=6, n_test_blocks=2)
cpcv_plan = create_split_plan(dataset, cpcv)
assert cpcv_plan.get_n_splits() == 15 and cpcv.n_paths == 5
cpcv_plan.paths[['path_id', 'block_id', 'fold_id']].head(8)
```

CPCV does not automatically filter missing targets. Empty training after purging
or embargo raises; the fold is not silently skipped.

## Reconstruct held-out values without fitting

The following values are alignment sentinels, not forecasts. A future fitting
layer must supply genuinely held-out predictions with its own provenance.

```python
demonstration_values = [
    pd.DataFrame({
        'example_value': f.test.astype(float),
        'example_missing': np.where(f.test % 7 == 0, np.nan, 1.0),
    }, index=f.test)
    for f in cpcv_plan.folds
]
paths = reconstruct_paths(cpcv_plan, demonstration_values)
assert paths.groupby('path_id').size().eq(len(dataset.X)).all()
assert paths['example_missing'].isna().any()
paths.head()
```

Supply one numeric DataFrame per fold, indexed by exactly its original held-out
positions. The frame order may differ; duplicate, missing, extra or training
positions are rejected. All folds need identical ordered output columns. NaNs
remain missing, including for multiple outputs. The result contains `path_id`,
`row_position`, `fold_id` and the output columns, sorted by path and kickoff with
stable ties. The [reference](splits_reference.md#prediction-path-contract) lists
the complete contract and reserved names.

## Connect a future fitting layer

```python
outer_cv = plan.split()
train_positions, test_positions = next(outer_cv)
assert plan.get_n_splits() == len(plan.folds)
assert np.array_equal(train_positions, plan.folds[0].train)
assert np.array_equal(test_positions, plan.folds[0].test)
```

`scheme.split(dataset)` is a convenience iterator. `plan.split()` can be supplied
as sklearn's `cv` iterable when the later caller uses the same dataset order.
No sklearn dependency or private sklearn base class is introduced here. Only
training/test membership is yielded; a later reporter must apply `fold.score`
explicitly. These are outer folds. Optional future inner selection must remain
inside each outer training population.

Split plans contain membership and ordering only, so reporters can inspect folds
without any fitting configuration. `SplitPlan` and `create_split_plan` do not own
`model_selector` or `refit_policy`.

The future training runner will own those options, both defaulting to `None`.
That will disable selection/search and additional scheduled refits; it will not
remove the need for an initial fit in each outer fold. No initial fit, selector
call or scheduled refit runs in this module.

## Interpretation and remaining work

CPCV paths reuse observations and fitted models; they are not independent evidence.
Purging supplied intervals does not automatically remove held-out outcomes that
later Glicko ratings or rolling features carry forward. Retrospective evaluation
needs a future layer to reconstruct or restrict feature/state preparation for
each fold, with a clear held-out information rule. Ordinary chronological
features must also respect the evaluation's actual issuance times.

The reference notebook `C:/Users/luisi/Escritorio/dsr_pbo_cpcv.ipynb` was inspected
read-only. Its combinatorial incidence idea is the reference; its finance returns,
variance, Sharpe, DSR/PBO and private sklearn code were not adopted or executed.
Fold-aware preparation, selection corrections, DSR/PBO/FDR reporting, strategy
execution and betting cash accounting remain later layers.

Verification evidence is in [splits_check.json](splits_check.json). It establishes
membership, alignment and preservation, not predictive benefit or actual
historical publication times.
