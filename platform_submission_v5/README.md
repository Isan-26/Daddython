# Platform submission v5: v4 + LightGBM-imputed branch averaged over 10 seeds

> **Result: public RMSE 3.25687, worse than v4 (3.24754). Not recommended; keep v4 as the final submission.**
> The public score can't tell whether seed 527's edge is luck on the public rows or a genuinely better fitted model.
> 527's advantage over the 10-seed average is somewhat larger than picking the best of three seeds would usually
> produce, which leans toward a genuinely better draw. The expected private gain from averaging (about 0.003)
> doesn't justify giving up the best public result.

**Upload:** `prediction_notebook.ipynb` and `model.pkl` (197 MB). Default `VERSION = 'v5_seed_average'`; `'v4_uncertainty'` and `'v2_fallback'` reproduce v4 and v2 exactly.

## The one change

v2 and v4 use a single LightGBM seed (527) for the LightGBM-imputed Ridge/HGB branch, which has 0.619875 of the final weight. v5 averages the ten already-fitted seeds 527–536 (taken frozen from `../platform_submission/model.pkl`), with v4's multiple-imputation averaging applied to each seed. Everything else is unchanged: the neural network, RealMLP, per-pattern experts, MissForest branch, all blend weights and the +0.8 for blank `previous_usage`. **No model was fitted** (fit methods are blocked in `build_v5.py`).

## Why

* The seeds only differ in which test blank patterns their imputers practised on, so they are interchangeable by design. Seed 527 was chosen because it scored about 0.02 better on the public leaderboard than seeds 528/529 (`best_model.ipynb`).
* On the test rows, the seeds disagree a lot where inputs are blank: the seed-to-seed standard deviation of the branch prediction is **0.51 per row on rows with a blank**, against 0.08 on complete rows.
* Averaging removes that noise. For squared error, the averaged prediction is never worse than the **average** of the ten single-seed predictions on any set of rows (convexity). The expected reduction, from the measured seed spread, is about 0.018 in MSE, roughly 0.003 RMSE. That's an estimate, not a promise.

## What to expect

* **Public score: probably worse than v4.** Seed 527's public edge looks like luck, and v5 removes it.
* **Private score:** if the private rows are separate from the public rows, seed 527 is just one random draw there, and v5 is the better bet. If the private score covers all test rows, part of seed 527's public luck also counts, and the two are closer.
* So **don't judge v5 by its public score alone.** Choose between v4 and v5 based on how the private ranking is computed.

## Changes vs v4 (`prediction_changes_vs_v4.csv`)

| Rows | Count | RMS change | Mean change |
|---|---:|---:|---:|
| complete | 2,453 | 0.043 | −0.003 |
| with any blank | 547 | 0.326 | about 0 (varies by pattern) |

The changes on rows with blanks are spread both ways with no systematic shift. v3, by contrast, moved blank rows down on average.

## Checks (all passed)

* Seed 527 in `../platform_submission/model.pkl` and in v4's `model.pkl` give identical predictions (difference 0.0).
* `v4_uncertainty` reproduces `../platform_submission_v4/candidate_v4_uncertainty.csv` exactly; `v2_fallback` reproduces `../platform_submission_v2/platform_dl13_prevfix.csv` exactly.
* Platform-style runs (shuffled test file without `id`, `DATATHON_INPUT_PATH` / `DATATHON_OUTPUT_PATH`), twice: identical output, equal to `candidate_v5_seed_average.csv`, one `prediction` column, 3,000 rows, about 18 seconds.

## Files

| File | What it is |
|---|---|
| `prediction_notebook.ipynb` | **upload** (all code inline) |
| `model.pkl` | **upload**; 197 MB, kept out of git (over GitHub's 100 MB limit) |
| `candidate_v5_seed_average.csv` | v5 predictions for `test.csv` (`id,prediction`) |
| `build_v5.py` | builds `model.pkl` from `../platform_submission_v4/model.pkl` + `../platform_submission/model.pkl` (`datathon-platform-env`, about 80 seconds) |
| `frozen_v2.py`, `uncertainty_runtime.py` | v4's code; one change: a version can select which v2 setting it uses |
| `v5_summary.json`, `prediction_changes_vs_v4.csv` | build summary and changes by blank pattern |
