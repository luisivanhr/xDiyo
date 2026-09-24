# Dataset assembly documentation and verification coverage

Status: verified. Focused/full tests, real-data reconciliation, all guide examples,
the fresh notebook, isolated wheel import, rendering and preservation audit pass.
Results are recorded in [datasets_check.json](datasets_check.json).

## Complete API coverage

| API | Import/signature and contract | Equations or practical use |
| --- | --- | --- |
| `evaluate_features(..., keyed=False)` | [Full signature and identity option](datasets_reference.md#feature-identity-option) | [Keep keys through shuffling and explicit columns](datasets.md#keep-feature-identities-attached) |
| `assemble_dataset` | [Every argument, default and validation rule](datasets_reference.md#assembly-entry-point) | [Team and match alignment equations](datasets.md#choose-the-layout-explicitly) |
| `ModelDataset` | [Every field, copying and output schema](datasets_reference.md#modeldataset-fields-and-groups) | [Metadata, settlement and grouping](datasets.md#metadata-settlement-and-downstream-groups) |
| `ModelDataset.groups` | [Exact tuple Series and adapter boundary](datasets_reference.md#modeldataset-fields-and-groups) | [Group identity equation](datasets.md#metadata-settlement-and-downstream-groups) |

## Behavioral coverage

- [x] Explain unchanged default feature values, order and index; keyed output is optional.
- [x] List available identity levels, required event/team/side fields and attrs metadata.
- [x] Accept named MultiIndex or explicit columns; reject anonymous positional alignment.
- [x] Explain exact IDs, partition collisions, reordered levels/rows and MultiIndex precedence.
- [x] Document unique/nonmissing identity checks, permitted extra rows and missing-record errors.
- [x] Require one selected LabelData and an explicit matching observation layout.
- [x] Explain complete team pairs, side agreement and reversed opponent identities.
- [x] Explain independent home/away matching, home-first column order and away target perspective.
- [x] Distinguish historical for/against features from current home/away assembly blocks.
- [x] Describe selectors, column order, unchanged dtypes and fresh shared output indices.
- [x] Write alignment, whole-match filtering and group identity equations in LaTeX.
- [x] Keep prediction fixtures by default; optional missing-target filtering preserves entire pairs.
- [x] Distinguish missing feature values from missing records; filter selected targets only.
- [x] Preserve finite push/void encodings and explain manually supplied infinite targets.
- [x] Keep selected settlement categories in metadata, with alignment and collision checks.
- [x] Explain exact group tuples, empty outputs, deep-copied definitions and cleared frame attrs.
- [x] Provide seven executable guide examples and a separate minimal sixth notebook.
- [x] Preserve the previous five notebooks, prior tests/evidence, source exports and archived docs.
- [x] Finish isolated wheel import, rendered math, navigation and stable-source evidence audit.

## Independent evidence and limits

The focused suite has 96 assembly/keyed-feature cases; all 490 analytics tests
pass, retaining the previous 394. The first focused run exposed two incorrect
settlement expectations in the test fixture: line 10 produced win/push/win.
Using line 11 supplies the intended push/loss/win cases. The fixture was corrected;
no implementation defect, library edit or callback was required.

One pinned Premier League 2024/25 publication supplies 380 matches and 760 team
rows. Four feature columns, four label definitions, keyed/explicit inputs and
both missing-target settings produce 16 assembly variants. Independent dictionary
joins reconcile 48,640 feature cells and 16,720 target cells exactly. Synthetic
tests cover missing outcomes, pair filtering, malformed identities, partition
collisions and exact large IDs not exercised by that finished real season.

Seven guide examples execute in the existing environment. The separate notebook
has nine cells, five containing code, and ran top-to-bottom in a fresh kernel.
The bounded wheel used 72 source files and imported the new APIs in an isolated
interpreter without installing packages. All 13 LaTeX expressions, including
four display equations, render without errors or overflow; eight sections were
visually reviewed. The final audit checks 51 local links, 29 stable source files,
141 preserved earlier files and five unchanged prepared-source/selection files.
The five earlier notebooks and seven archived original documents are unchanged.

These checks establish assembly behavior; temporal splitting, model fitting,
pilot selection and general orchestration remain separate work. Source
availability caveats from feature construction continue to apply.
