# Match-result reporting reference

This reference describes the independently verified source recorded in
[match_results_check.json](match_results_check.json). Start with the
[executable guide](match_results.md). [Post-training reporting](post_training_reference.md)
defines the shared orchestration, pooling and report contracts.

## Public entry points

`MatchResultReporter` and `TeamCatalog` are exported from
`xdiyo_analytics.reporting`. `team_key` is available from
`xdiyo_analytics.reporting.teams`. The renderer lives in
`xdiyo_analytics.reporting.match_view`; callers normally use `AnalysisReport`.

```text
MatchResultReporter(*, type, partition='score', pooling=None, target=None,
    output='predict', comparison='auto', tolerance=0.0,
    league=None, season=None, team=None, round=None,
    catalog=None, show_badges=False, page_size=25,
    league_column=None, season_column=None,
    probability_output='predict_proba', decision=None,
    threshold=None, positive_class=None)
MatchResultReporter.run(context) -> StudyResult
TeamCatalog(entries)
TeamCatalog.from_json(path) -> TeamCatalog
TeamCatalog.from_matches(matches) -> TeamCatalog
TeamCatalog.display(key, *, badges=False) -> (name, image_uri, note)
team_key(value) -> str
render_match_results(artifact) -> str
```

## All reporter options

| Option | Meaning and validation |
|---|---|
| `type` | Required: `per_fold` or `overall`. `timeline` is unsupported. |
| `partition='score'` | `score` selects retained score positions; `test` selects all retained test positions. Train/model/experiment scopes are unsupported. |
| `pooling=None` | Inherited repeated-row policy: `occurrences`, `first`, `last` or `mean`. Required when an overall scope repeats row positions. No repeated rows need no policy. |
| `target=None` | All target columns. A string selects one; a sequence selects distinct columns in the given order. Empty/duplicate selections fail; unknown targets fail. Each target gets its own panel. |
| `output='predict'` | Named retained output. Flat columns are point predictions. A two-level `(target, class)` MultiIndex selects probability mode. Unknown outputs fail. |
| `comparison='auto'` | `auto`, `numeric` or `categorical`. Auto is categorical for probability output, `Outcome`/`Above`/`BetOption` label definitions, or a nonnumeric target dtype; numeric otherwise. Explicit settings override auto. Use categorical for custom integer classes. |
| `tolerance=0.0` | Finite, nonnegative real, excluding bool. Numeric green means absolute signed error is at most this value. Validated even for categorical reports; it has no categorical effect. |
| `league=None` | Initial league display filter; `None` means All. It does not reduce the stored table. |
| `season=None` | Initial season display filter; `None` means All. |
| `team=None` | Initial stable-ID filter matching either home or away. Names are display text, never identity keys. |
| `round=None` | Initial round filter; `None` means All. Zero is valid. Integral numeric values normalize consistently, e.g. `1.0` and `1`. |
| `catalog=None` | A `TeamCatalog` or ID-to-name/record mapping. `None` uses metadata names or `Team <ID>`. |
| `show_badges=False` | Opt into embedding accepted local badges. Rendering later uses the already embedded bytes. |
| `page_size=25` | Positive built-in integer, excluding bool. Counts fixture occurrences, not team observations. |
| `league_column=None` | Explicit metadata column, otherwise `source_league` if present, else `competition_id`. The selected column is required. |
| `season_column=None` | Explicit metadata column, otherwise `source_season` if present, else `season_id`. The selected column is required. |
| `probability_output='predict_proba'` | Optional companion output beside flat point predictions. Read if that name is present. Set `None` to disable. Invalid companion vectors leave their probability cell unavailable; valid point comparisons still work. |
| `decision=None` | No implicit probability decision. `argmax` chooses the first stored class attaining the maximum. Requires probability mode. Mutually exclusive with `threshold`. |
| `threshold=None` | Optional binary positive-class threshold in `[0,1]`, inclusive. Requires probability mode, exactly two distinct classes and a declared positive class. Class schema is checked even with no rows or all-invalid vectors. |
| `positive_class=None` | The class selected when its probability is at least `threshold`; the other class is selected below it. Required and present in the class columns for threshold mode. Alone it supplies no decision rule. |

Configuration is validated during execution. `supported_types` is the class-level
tuple `('per_fold', 'overall')`, not a constructor option. Defaults do not enable
the reporter automatically; register a named instance in `PostTrainingAnalysis`.

## Inputs, scopes and identity

`run(context)` consumes a `PostTrainingContext` whose metadata, targets and chosen
output have the same `(fold_id, row_position)` index. It reads the declared
`match_columns`, target definitions, layout and retained outputs. It never calls
a model. `PostTrainingAnalysis.run(training, fold_ids=...)` prepares this context
and retains scope identities in the resulting `StudyRun`.

| Layout | Required fixture metadata | Presentation |
|---|---|---|
| `match` | Declared match columns, `home_id`, `away_id`, league and season columns | One observation per fixture/fold; Home, Away, Result, Prediction, optional Error and Probabilities. |
| `team_match` | Declared match columns, `team_id`, `opponent_id`, `side` equal to `home` or `away`, league and season columns | Pair at most one observation per side, under separate Home label and Away label groups. |

`round` and `stage` may be absent. A non-None initial round filter requires `round`.
Optional metadata names are `home_name`/`away_name`, or
`team_name`/`opponent_name` for a team layout. Optional
`settlement::<target>` supplies `push`, `void` or `missing` neutrality.

Fixture identity combines the fold ID with all declared match keys. Repeated
fixtures from different folds stay separate under `occurrences`. Pairing requires
agreement in home/away IDs, league, season, round and stage. Duplicate fixture/side
observations fail. A single retained side is valid and leaves the other side blank.

For overall scopes, `first`/`last` use selected fold order per original row
position. `mean` combines numeric output cells, assigns fold ID `-1`, and leaves
a cell missing if any contributing occurrence is missing. Do not use mean to
average integer class decisions. The shared orchestrator applies pooling to all
retained named outputs; a probability-only result avoids averaging class decisions.
Conflicting outcomes or metadata for the same repeated row position fail.

## Comparison and probabilities

Point comparisons require both values after missing/nonfinite normalization.
Numeric comparison requires real scalar labels and computes prediction minus
result. Categorical comparison uses equality. `push`, `void` and `missing`
settlements take precedence and remain neutral. A valid vector with an observed
class and no decision has internal status `no decision`; the Prediction shows a
dash. See the [equations](match_results_equations.md).

Probability frames need aligned rows and exactly two column levels. Each target
must have at least two distinct class columns. Values must be finite, lie in
`[0,1]`, and sum to one with `atol=1e-6, rtol=0`. Invalid numeric vectors become
unavailable. Malformed schemas or nonnumeric conversion failures raise errors.
Missing class columns in a pooled fold stay missing; there is no filling or
renormalization. Argmax uses stored class order even if the browser displays
integer-like class keys in a different JSON property order.

## Returned tables and artifacts

Each selected target produces `result.tables['matches::<target>']` with the full
reporter population and an `Artifact(kind='match_results', ...)` over that table.
Initial filters change neither its length nor the enclosing study's scope.

| Fields | Meaning |
|---|---|
| Declared match columns | Original match identity values. |
| `fixture` | Internal serialized fold-plus-match identity used for grouping. |
| `fold_id`, `row_position` | Retained occurrence provenance; mean-pooled fold is `-1`. |
| `league`, `season`, `round`, `stage` | Competition context; missing values remain missing. |
| `home_id`, `away_id`, `home`, `away` | Stable IDs and resolved display names. |
| `side` | `home`/`away` for team observations, otherwise missing. |
| `target`, `result`, `prediction`, `error` | Selected target, observed value, optional prediction/decision and numeric signed error. |
| `status` | `correct`, `incorrect`, `within tolerance`, `outside tolerance`, `unavailable`, `no decision`, `push`, `void` or `missing`. Exported and used for colors, without a fixture-table Status column. |
| `settlement` | Original normalized settlement value, if supplied. |
| `probabilities` | JSON class-label-to-probability string for a valid vector, otherwise missing. |

Artifact options are `teams` (ID to resolved `name` and embedded `badge`), `layout`,
`page_size`, `numeric`, `probabilities`, `tolerance` and `initial` (the four
normalized filters). Notes describe population retention, signed errors, neutral
settlement behavior, missing names and rejected badge files. Rendering consumes
these computed options and table values.

## TeamCatalog and exact IDs

- `TeamCatalog(entries)` canonicalizes ID keys and shallow-copies each record.
  A name string becomes `{'name': ...}`. Extra provenance fields are retained.
- `from_json(path)` reads an ID-to-record JSON mapping using UTF-8 with optional
  BOM. Relative `badge_path` values resolve beside that JSON file; absolute local
  paths are accepted. The method performs no download.
- `from_matches(matches)` accepts one DataFrame or an iterable. Each frame needs
  both ID/name column pairs. Frames are processed in order, home observations
  first and away observations second. The last encountered nonmissing name for
  each ID wins. This is not a chronological “latest name” resolver. Missing IDs
  or names are skipped; equal names never merge different IDs.
- `display(key, badges=False)` returns `(str_name, None, None)` without reading
  badge files. With badges enabled, it returns a data URI or a failure note.
  Missing/falsy names fall back to `Team <ID>` at this helper level.
- In a reporter, a nonempty catalog name wins. Otherwise the first encountered
  metadata name for the team is used; if unavailable, `Team <ID>` and a note are
  retained. Resolution and badge reading happen once per team per `run` call.
- `team_key(value)` preserves integer IDs exactly as decimal strings. Finite
  real values equal to an integer lose the `.0`; other values use `str(value)`.
  It does not validate IDs or repair already rounded upstream floats.

The [repository catalog](team_assets/README.md) is optional: 289 IDs, 286 badges
(284 SVG and two PNG), three name-only IDs. Its source/identity/asset records are
kept separately from model output. The wheel does not bundle these optional files.

## Local badge acceptance

`_badge(path)` accepts local SVG, PNG, JPEG (`.jpg`/`.jpeg`) and WebP files up to
2,000,000 bytes. It returns the original bytes as a base64 data URI. SVG is parsed
as UTF-8 with an optional BOM and checked for the following supported subset:

- SVG root, no document/entity declarations, script, foreignObject, iframe,
  object, embed or event-handler attributes.
- `href`/`src` references must be local fragments or embedded base64 PNG/JPEG
  tiles with the expected signature. ASCII whitespace within base64 is accepted.
  Remote references and nested SVG data URLs are rejected.
- Style text/attributes cannot use `@import`; `url(...)` references must start
  with a local fragment after stripping surrounding whitespace and quotes.

`_local_reference(value)` implements the fragment/raster-reference check.
`display` catches file, value and XML parse errors, retaining the name and note.
Top-level raster acceptance uses suffix/size; a browser may still reject corrupt
image bytes. Its image-error handler removes the image while retaining the name.
The independently audited catalog's active PNGs were decoded as well as hashed.

## Renderer and browser helpers

`render_match_results(artifact)` serializes the computed table into an escaped
JSON script block and returns its HTML container. No filesystem, network or model
access occurs here. IDs/context values become strings; integral result/prediction
values beyond JavaScript's exact integer range also become strings. Scalar
missing values become JSON null. Titles are HTML-escaped; JSON escapes `<`, `>`
and `&`. Browser text is inserted through `textContent`.

`MATCH_CSS` and `MATCH_JS` supply the shared viewer's fixture style and behavior:

| Helper or control | Behavior |
|---|---|
| `matches`, `labelFor`, filter controls | Match all four selections, with either-side team membership. Each option list reflects the other selections; an unknown initial choice remains selectable and can yield zero rows. |
| `refresh`, `valueFor` | Filter full observation rows, group fixture occurrences and sort by original order, team name or the selected side's result/prediction/error. Missing values sort last. |
| `draw` | Paginate fixtures, retain competition/stage/round/fold headings and align their spans with optional numeric/probability columns. Empty views show a usable zero-result message. |
| `mood`, `outcomeCells` | Apply good/bad/neutral colors independently to label groups. Either bad side gives a bad row; both present/good sides are required for a good paired row. Settlement words replace the displayed observed value. |
| `format`, `teamCell` | Dashes for unavailable values, up to four decimal places for numeric display, optional 28-pixel embedded badge beside the full name. Probabilities show one decimal percentage. |
| `text`, `addButton` | Construct DOM text and typed buttons with listeners. Names and labels are not interpreted as markup. |
| Reset filters | Clear four selections and return to page one; retain the chosen sort/direction. Each target panel and fold view has independent state. |
| Download filtered CSV | Export every matching observation across all pages in original observation order, with UTF-8 BOM and quoted fields. Omit the internal `fixture` field; retain IDs, status and full numeric precision. Sorting changes display order only. |

The shared Data table download remains the full stored population. No Date or
Status column is added to the compact fixture table; context appears in group
headings. Wide paired tables scroll inside their panel on narrow screens.

## Internal calculation helpers and shared integration

`_plain(value)` converts pandas missing sentinels and nonfinite floats to `None`
and NumPy scalars to Python scalars. `_probability_rows(frame, target, index)`
validates the class schema and yields ordered dictionaries or `None` per row.
`_columns` is the existing shared target-selection validator. These helpers are
implementation details, not separate user configuration objects.

The shared `render_report` recognizes `match_results` alongside existing artifact
kinds. `AnalysisReport.to_html(path=None, renderers=None)` returns standalone HTML
and optionally writes it. `to_notebook(height=800, renderers=None)` returns an
IPython iframe; `show(...)` displays it once and returns `None`. Automatic rich
display uses the same isolated document. Rendering never reruns the study.

The [verification record](match_results_check.json) covers fresh notebook execution
and saved HTML interactions. Live Jupyter frontend behavior remains unverified.
