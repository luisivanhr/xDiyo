# Experiment builder browser verification

Verified on 19 September 2026 through the Codex in-app browser against the local Python server. All fitted examples used the isolated synthetic recipe at `.pytest_tmp/ui_browser/recipe.json`.

## Observed interactions

- The real data folder discovery listed 135 published season files. Premier League 2024/25 inspection listed 138 statistic identities; no real-data fitting was performed.
- The recursive feature form exposed `RollingMean` with numeric window 5 and its nested `Stat` parameters.
- Importing and reopening the synthetic recipe worked. Preparation returned exact large event/team identifiers and one fold with 16 training, 7 test and 7 scoring rows.
- Fixed Ridge fitting completed and displayed the existing report viewer inline, including readable team names and match filters.
- Enabling selection, adding a numeric grid `[0.1, 1]`, renaming its key to `alpha`, and clicking Run completed a two-candidate search. The report displayed both parameter values, scores and the selected candidate, without internal ID/hash columns in the comparison table.
- The initial grid-key rename interaction exposed a stale form value. Changing the handler to update on input corrected it; the entire browser search was then repeated successfully.
- Feature forms were visually inspected at the default narrow in-app width and a temporary 1400-by-900 desktop viewport. The desktop document width was 1385 pixels within the 1400-pixel viewport. The viewport override was reset afterward.
- The final inspected browser session reported no console errors.

The report heading initially duplicated the selected grid parameter. The recipe compiler retains distinct readable candidate names, and the experiment display-name function adds only parameters not already present in that name. This presentation follow-up is covered by the targeted Python verification record.

## Scope

The API/workflow tests separately cover persistence, recovery, prediction alignment, native CPU boosting, parallel folds and exported code. Browser evidence here does not establish native model CUDA operation, mid-fit checkpoint recovery or a live Jupyter frontend. The launcher notebook is deliberately saved unexecuted; its cells and structure were validated. No package was installed and no production model was trained.
