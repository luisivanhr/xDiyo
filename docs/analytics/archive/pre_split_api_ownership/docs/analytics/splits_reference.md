# Splits and CV API and helper reference

Import public APIs from `xdiyo_analytics.splits`. The
[practical guide](splits.md) provides executable examples and LaTeX equations;
the [coverage checklist](splits_documentation_checklist.md) records verification.

## Shared dataset and timing contract

All built-in splitters consume an assembled `ModelDataset`. `X`, `y` and metadata
must have equal lengths and the same index/order. Data must be nonempty, and
every `match_columns` value must be nonmissing. Exact `dataset.groups` tuples
identify matches; neither float-converted IDs nor a plain event ID replace them.
The complete pair and layout contract comes from [assembly](datasets_reference.md).
Changing or manually constructing a container remains the caller's responsibility.

Every membership array contains distinct integer original-row positions in
ascending original order. Paired rows and partition identities remain whole.
Training and test membership are disjoint; scoring is a subset of testing.
All built-in families raise on empty training/test folds. Empty scoring is allowed.
Inputs are not sorted, filtered or mutated in place.

Temporal/CPCV require nonmissing, match-constant `kickoff_at`. Timestamp arguments
accept a metadata column name, list/array or Series with one value per row. A
Series must have the metadata index/order. Numeric dtypes are rejected rather
than guessing epoch units. UTC conversion uses `pandas.to_datetime(..., utc=True)`;
invalid timestamp strings raise. Naive values are interpreted as UTC by that
conversion. Missing cutoff/information-start/CPCV availability raises; missing
explicit temporal availability instead excludes the match from training.

Metadata grouping columns must exist, be nonmissing and agree across all rows of
each match. A grouping specification accepts a column name or tuple/sequence of
names. Exact matching and equality apply; names do not imply domain relationships.

## Fold and SplitPlan records

```text
Fold(train, test, score, metadata=<new empty dict>)
SplitPlan(folds, n_rows, row_order, paths=None,
          model_selector=None, refit_policy=None)
create_split_plan(dataset, splitter, *, model_selector=None, refit_policy=None,
                  **split_options) -> SplitPlan
```

| Member | Meaning |
| --- | --- |
| `Fold.train` | Original positions eligible for this fold's training membership. |
| `Fold.test` | All held-out original positions, including unscored observations. |
| `Fold.score` | Original positions to score; does not move other test rows into training. |
| `Fold.metadata` | Scheme, timing/grouping assumptions and family-specific information below. Default is a separate empty dictionary per record. |
| `SplitPlan.folds` | Prepared list of `Fold` objects. |
| `SplitPlan.n_rows` | Original dataset row count. |
| `SplitPlan.row_order` | Stable kickoff-sorted original positions if `kickoff_at` exists, otherwise original position order. It is not a change to the dataset. |
| `SplitPlan.paths` | CPCV path mapping DataFrame; otherwise `None`. |
| `SplitPlan.model_selector` | Supplied object retained by reference; `None` disables future search/selection. Never invoked here. |
| `SplitPlan.refit_policy` | Supplied object retained by reference; `None` disables additional scheduled refits, not a future initial fit per outer fold. Never invoked here. |

`create_split_plan` calls `splitter.folds(dataset, **split_options)` and, when
present, `splitter.path_map(folds)`. Options therefore belong to the selected
splitter's `folds` signature. It creates no inner folds, features, fitted state,
selector run or model fit. Custom splitter objects can use that same interface
but must uphold the membership contract themselves. The mutable dataclass
constructors do not independently validate manually supplied fold/path arrays.

### Iterator convenience and sklearn use

```text
splitter.folds(dataset, **family_options) -> list[Fold]
splitter.split(dataset, **family_options) -> iterator[(train, test)]
splitter.get_n_splits(dataset, **family_options) -> int
plan.split(X=None, y=None, groups=None) -> iterator[(train, test)]
plan.get_n_splits(X=None, y=None, groups=None) -> int
```

Splitter methods compute membership on each call; `get_n_splits(dataset)` also
builds/validates the folds. Plan methods reuse prepared folds. Both iterators
yield copies of the membership arrays. `plan.split` checks supplied `X` length
but cannot verify its order; `y` and `groups` are unused compatibility parameters.
`plan.get_n_splits` returns the stored fold count and ignores those parameters.

Use `plan.split()` as a `cv` iterable for a later sklearn call with the original
dataset order. No sklearn import, private base class or fit is needed to prepare
it. This iterator exposes train/test only; scoring subsets require separate use
of `fold.score`. All plans are outer evaluations; future inner selection belongs
within the corresponding outer training data.

## TemporalSplit settings and eligibility

```text
TemporalSplit(train_size, test_size=1, step=None, window='expanding', unit='rounds',
              gap=0, gap_time=None, calendar_by=None, block_by=None,
              score_start=0, score_rounds=None, allow_partial_test=False)
TemporalSplit.folds(dataset, *, cutoffs=None, available_at=None)
```

| Parameter | Default, meaning and restrictions |
| --- | --- |
| `train_size` | Required positive integer. Initial/minimum candidate block span for expanding windows, fixed candidate span for sliding. |
| `test_size` | Positive integer, default 1 held-out block per horizon. |
| `step` | `None` means `test_size`; otherwise a positive integer count of blocks between fold starts. Tests may overlap if the step is shorter than the horizon. |
| `window` | `expanding` or `sliding`; default expanding from calendar start. |
| `unit` | `rounds`, `seasons` or `kickoffs`; controls default calendar/block keys. |
| `gap` | Nonnegative integer, default 0 candidate blocks skipped between training and testing. |
| `gap_time` | Nonnegative `pandas.Timedelta`-convertible duration; `None` is zero. Applied before both kickoff and availability eligibility. Prefer explicitly unit-bearing values such as `2D`. |
| `calendar_by` | `None` means `('competition_id',)` for rounds/seasons and `()` for kickoffs. Empty tuple pools. Other names group independent calendars. |
| `block_by` | `None` means `('season_id','round')`, `('season_id',)` or `('kickoff_at',)` according to unit. Explicit nonempty names replace that identity. Include stage when rounds repeat across stages. |
| `score_start` | Nonnegative integer, default 0. Skip held-out blocks only for scoring. Must be less than configured `test_size`, even when the final horizon is shorter. |
| `score_rounds` | `None` or inclusive `(low, high)` round-number bounds; either may be `None`. Requires numeric-convertible, match-constant nonmissing `round`. Lower bound cannot exceed upper bound. Applied in every held-out season and intersected with `score_start`. |
| `allow_partial_test` | Default `False`; true includes a final shorter horizon. A short horizon may have no scoring rows. |
| `cutoffs` | `None` uses kickoff. Explicit per-row datetimes reduce to the minimum within each match; the effective match cutoff must not exceed kickoff. Fit time is the earliest test-match cutoff. |
| `available_at` | `None` uses kickoff as a retrospective proxy. Explicit per-row datetimes reduce to the maximum within a match only when every row is nonmissing; otherwise the match is unavailable. |

Integer settings reject booleans and nonintegral values. Duration settings reject
negative or missing durations. Dataclass construction stores settings; these
validations run when folds are requested. `allow_partial_test` is intended as a
boolean switch; it is not separately type-validated.

Observed blocks are sorted by their minimum supplied kickoff, not by numerical
round/season labels. First-appearance order breaks equal block-time ties.
Calendar order follows first appearance in the original dataset. Default kickoff
blocks keep all equal times together; a custom `block_by` defines a different
blocking rule, so the caller owns its meaning. Same-time candidate training still
fails the strict temporal kickoff check.

After candidate block selection, retain a match only when its kickoff is strictly
before `fit_at-gap_time`, its availability is at or before that boundary, and
every selected target cell across the match is nonmissing. This excludes
postponed and unknown-target training matches. Missing features are retained.
This is an `isna` check, not a finiteness conversion for manually altered targets.
Test/score targets are never filtered automatically. No fitting or feature warm-up
is performed. See the [eligibility equations](splits.md#state-prediction-and-availability-times).

No complete horizon raises unless partial horizons are enabled and one fits.
Any horizon with no eligible training raises; it is not silently omitted.

### Temporal fold metadata

| Key | Value |
| --- | --- |
| `scheme`, `retrospective` | `temporal`, `False`; describes membership direction, not proof of input-feature timing. |
| `calendar`, `calendar_by`, `block_by` | Calendar tuple and resolved grouping/block-column tuples. |
| `unit`, `window` | Requested span unit and expanding/sliding mode. |
| `fit_at`, `training_boundary` | Earliest held-out effective cutoff and that time minus the duration gap. |
| `train_blocks`, `test_blocks` | Ordered candidate training and held-out block tuples; train blocks precede eligibility pruning. |
| `excluded_train` | Original positions removed from the candidate training population by timing/target eligibility. Does not include skipped gap blocks. |
| `availability`, `cutoffs` | `kickoff_proxy`/`explicit` and `kickoff`/`explicit`, respectively. |

## MatchKFold and GroupKFold

```text
MatchKFold(n_splits=5, shuffle=False, random_state=None)
MatchKFold.folds(dataset)
GroupKFold(n_splits=5, group_by=('competition_id','season_id'),
           shuffle=False, random_state=None)
GroupKFold.folds(dataset, *, groups=None)
```

| Parameter | Contract |
| --- | --- |
| `n_splits` | Integer at least 2; cannot exceed match count for MatchKFold or distinct group count for GroupKFold. Booleans/nonintegral values are rejected. |
| `shuffle` | Default `False`. MatchKFold permutes unique match IDs; GroupKFold randomizes equal-size group ordering only. Intended boolean switch, not separately type-validated. |
| `random_state` | Default `None`; forwarded to `numpy.random.default_rng` only when shuffling. Use an integer seed for reproducibility. It is not sklearn's RandomState contract. |
| `group_by` | GroupKFold's nonempty metadata column name(s), default competition-season. Ignored when external `groups` is supplied. |
| `groups` | Optional row-aligned hashable values, including tuples; Series index/order must match metadata. Values must be nonmissing and identical for every row of a match. |

Unshuffled MatchKFold uses unique matches in first-appearance order, with fold
match counts differing by at most one; larger folds come first. GroupKFold sorts
groups by descending match count, then repeatedly assigns the next group to the
currently lightest fold. Stable ties use original group order or its shuffled
order; equal fold loads choose the lowest fold number. It balances matches,
preserves whole groups and is not sklearn-identical.

Both schemes test each match exactly once and score every held-out row. Metadata
contains `scheme` (`match_kfold` or `group_kfold`) and `retrospective=True`.
GroupKFold additionally records `test_groups`, a tuple of the selected group keys.
Neither scheme filters missing targets, enforces chronology, or checks result
availability. They need no kickoff column to build folds; creating a plan uses
it for `row_order` if present.

## CPCV configuration, intervals and paths

```text
CPCV(n_blocks=6, n_test_blocks=2, embargo=None)
CPCV.folds(dataset, *, information_start=None, available_at=None)
CPCV.n_paths -> int
CPCV.path_map(folds) -> pandas.DataFrame
```

| Parameter | Contract |
| --- | --- |
| `n_blocks` | Integer at least 2, default 6; cannot exceed the number of distinct kickoff batches. |
| `n_test_blocks` | Positive integer smaller than `n_blocks`, default 2. All combinations are used in lexicographic block order. |
| `embargo` | Nonnegative duration convertible with `pandas.Timedelta`; default `None` is zero. Removes training kickoffs strictly after each held-out run's latest end through the inclusive embargo endpoint. |
| `information_start` | Optional timestamp column/vector. Default kickoff; minimum across all rows of each match when supplied. Missing timestamps raise. |
| `available_at` | Optional timestamp column/vector. Default kickoff; maximum across all rows of each match when supplied. Missing timestamps raise. |

Counts reject booleans/nonintegral values. Effective interval starts must not
exceed ends; effective availability ends must not precede kickoff. There is no
additional requirement that the supplied start equal or precede kickoff.
The [guide equations](splits.md#combinatorial-purging-and-embargo) define closed
overlap, run embargo, fold count and path count. Intervals are checked against
each held-out event; gaps between disjoint intervals are not filled. Embargo is
applied after each contiguous run of selected block numbers, using the latest
event end anywhere in that run. Empty training raises. Targets are not filtered.

`n_paths` validates block-count settings and returns the combinatorial path count
without inspecting data. `path_map` consumes the complete generated CPCV fold
collection. It assigns successive occurrences of each block to zero-based path
IDs, then sorts by path/block. Missing occurrences raise. Preserve the generated
fold order and metadata; arbitrary edited or manually built fold collections are
not a separately validated protocol.

The path table has `path_id`, `block_id`, `fold_id`, and `rows` (a copied NumPy
array of original positions). Each generated path covers every block once and
each fold/block incidence is used once. Repeated rows across different paths are
expected; paths are not independent samples.

### CPCV fold metadata

| Key | Value |
| --- | --- |
| `scheme`, `retrospective` | `cpcv`, `True`. |
| `test_blocks` | Selected zero-based block-number tuple. |
| `block_rows` | Mapping from selected block number to original position arrays. |
| `purged_train` | Candidate training positions whose intervals overlap held-out intervals. |
| `embargoed_train` | Positions excluded by embargo that were not already purged. |
| `information_start`, `availability` | `kickoff_proxy` or `explicit` for each interval input. |
| `embargo` | Resolved nonnegative pandas Timedelta. |

## Prediction path contract

```text
reconstruct_paths(plan, predictions) -> pandas.DataFrame
```

`plan` must have CPCV paths. `predictions` is a sequence or integer `fold_id`
mapping with one frame per fold, accessible by every zero-based fold ID.
Each frame must be a numeric pandas DataFrame with a unique integer index equal
to that fold's held-out **original positions**, in any order. Full-dataset,
training, duplicate, missing or extra positions are invalid. A malformed mapping
with absent fold IDs raises on lookup.

All frames must have identical ordered, unique, nonempty output columns, excluding
`path_id`, `row_position` and `fold_id`. The implementation accepts column labels
supported by pandas; use descriptive names. Numeric nullable/Arrow types and
missing predictions are supported. Non-numeric dtypes are rejected. It does not
validate probabilities, finiteness, prediction quality, provenance or how a
numeric value was produced.

Output columns are `path_id`, `row_position`, `fold_id`, followed by prediction
columns. Rows are sorted by path then `plan.row_order` and receive a fresh
RangeIndex. Values and missingness are copied; inputs are unchanged. No NaN is
interpreted as a loss, zero, absent trade or removed path row. A future reporting
layer must state its own handling of missing predictions.

## Internal helpers and error boundaries

These are internal helpers, not extra package exports. They live in `splits.core`
except `CPCV._check`, which belongs to `splits.cpcv`:

| Helper | Responsibility |
| --- | --- |
| `Splitter` | Shared `split` and `get_n_splits` convenience methods over a subclass's `folds`. |
| `_integer(value, name, minimum=1)` | Reject booleans, nonintegral values and counts below the threshold. |
| `_duration(value, name)` | Resolve `None` to zero and validate a nonnegative, nonmissing duration. |
| `_times(dataset, value, name)` | Validate row count/Series index, reject numeric epochs and convert to UTC. |
| `_Matches(dataset)` | Validate aligned/nonempty data, factorize exact match tuples and store reversible original-position membership. |
| `_Matches.rows(matches)` | Expand match codes to sorted original row positions. |
| `_Matches.constant(values, name)` | Require one common value per match and return its representative value. |
| `_Matches.columns(names)` | Resolve names and validate nonmissing, match-constant metadata groups. |
| `_Matches.times(value, name, reduction=None)` | Require nonmissing datetimes, then match-constant times or per-match min/max reduction. |
| `_fold(matches, train, test, metadata, score=None)` | Expand match membership, reject empty train/test and default score to all test rows. |
| `CPCV._check()` | Validate block counts for fold generation and `n_paths`. |

Wrong dataset type raises `TypeError`. Invalid settings, alignment, time values,
missing grouping values, insufficient groups/blocks, empty folds and malformed
prediction frames generally raise `ValueError`; missing metadata column names
raise `KeyError`. Unsupported keyword options raise the receiving method's
`TypeError`. Timestamp/duration conversion errors retain pandas' exception details.

The module has no file I/O, sklearn dependency, private sklearn inheritance,
statistical reports, selection runner, refit executor or model training.
Retrospective fold-aware feature/state reconstruction remains a separate layer;
purging alone does not remove held-out outcomes carried by earlier feature
computations. See [interpretation limits](splits.md#interpretation-and-remaining-work).
