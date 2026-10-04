# Platform submission v3

This is an additive, training-only feature and imputation experiment. The files in
`../platform_submission_v2/` are inputs and are never overwritten. The v2 public
leaderboard score is not used to select any new setting.

Run from the repository directory using the existing platform environment:

```sh
/Users/ethanis/Downloads/datathon-platform-env/bin/python platform_submission_v3/feature_imputation_upgrade.py
```

The default is **FULL**: outer and inner 5-fold shuffled KFold, seed 2026, five
LightGBM imputer seeds 527–531, 700 estimators per imputer, and 80 ExtraTrees trees.
`--fast` uses three folds, three seeds, 300 estimators, and 40 ExtraTrees trees;
it is a development mode, not the final configuration. `--check-only` reproduces
the saved v2 submission without fitting anything.

## What is preserved

The cheap architecture remains Ridge + residual histogram gradient boosting,
missing-pattern Ridge/boosting experts, and the original v3 notebook expert.
The frozen v2 functions are copied into `legacy_runtime.py` and
`legacy_training.py` so the same-fold baseline uses the actual original recipe.

The five pretrained sklearn networks keep their **40 inputs**: four numeric
values, two expected values, five event/deviation values, weekend, six daily
Fourier terms, two monthly terms, twelve building indicators, and eight type
indicators. Their original imputer, scaler, target normalization, and weights
are retained. RealMLP keeps its **15 numeric inputs**, original category coding,
building-hour median filling, and all ten saved sets of weights.

Effective final weights stay at **0.8265 non-DL + 0.0435 sklearn MLP + 0.13
RealMLP**. Existing DL OOF artifacts use seed-0 folds, not the requested seed-2026
folds, so they are not mixed into the new CV results. Reported CV scores evaluate
the non-DL branch; the final full ensemble has no claimed OOF RMSE.

## New branch

All new imputer training uses training inputs only, never energy labels, IDs,
public-test rows, or public-test frequency weights. Original numeric columns are
predicted from the other inputs using multiple LightGBM seeds, ExtraTrees, and
smoothed hierarchical medians. Coarse convex weights are selected independently
for the four variables. Dispersion between seeds and imputation methods provides
uncertainty features; this dispersion is not a calibrated prediction interval.

Inside every outer fold, a second KFold produces cross-fitted imputer predictions
for training energy models. Every source row and all its artificial copies stay
in the same fold. Imputation blend weights for a training origin are selected
without that origin's reconstruction targets. Hierarchy statistics are fitted
only on the relevant training partition. Final inference imputers are fitted on
all training rows. Observed inputs are never overwritten, and surprise features
are zero for originally missing values.

The core adds four original missing flags, count, bitmask/one-hot missing pattern,
and OOF expected values. Six small ablations test uncertainty, weekday/building
interactions, smooth weather, hierarchical deviations/surprises, physical cross
features, and their combination. A family is retained only for a meaningful
gain, improvement in most inner folds, stable fold variation, and safe pattern
errors. The residual boosting model uses the same core inputs in all ablations,
so these ablations isolate the Ridge feature changes.

Mask augmentation is half sampled from natural training patterns and half from
generic patterns. New training weights are 1 for original rows and 0.25 for each
masked copy. All 16 patterns are supported. Two-pass imputation updates only
originally missing variables, simultaneously, and is enabled per pattern only
if its downstream inner-OOF result passes the gain/consistency screen.

Pattern strategies use a coarse 0.1 direct/imputed grid. Missing-previous bias
corrections are learned from OOF residuals and screened against no correction;
contextual corrections shrink toward the global value. There is no hard-coded
`+0.8` correction in the new inference path.

## Evaluation and selection

The current non-DL recipe is refitted on the same outer splits. Its original
transductive imputers and test-mix weights are kept **inside that legacy baseline**.
New models never use that test distribution. Outer validation labels do not
select their feature family, imputer weights, strategy, or bias correction.

Two scores are reported explicitly:

- **Natural OOF RMSE:** each held-out training row with its original missingness.
- **Balanced stress RMSE:** 50% natural rows plus 50% uniform weighting of the
  fifteen nonzero artificial patterns on complete validation rows. This is a
  training-only robustness objective, not an estimate of the hidden test mix.

The upgrade must improve mean balanced RMSE by at least 0.004, improve most outer
folds, keep natural RMSE within 0.005, limit major pattern regressions to 0.15,
and avoid a large increase in fold standard deviation. Otherwise the artifact
uses the legacy non-DL expert. Its optional learned fallback bias is separately
reported as a secondary, leave-fold-out calibration diagnostic, not as a fully
nested CV result. No leaderboard improvement is promised.

## Outputs

- `candidate_feature_upgrade.csv`: exactly `id,prediction`, 3,000 rows in test order.
- `feature_upgrade_cv_results.csv`: outer-fold RMSE, MAE, R², and pattern diagnostics.
- `imputation_cv_results.csv`: outer reconstruction scores, including actual
  training-selected ensemble weights rather than validation-fitted weights.
- `missing_pattern_results.csv`: before/after errors and final pattern choices.
- `feature_family_results.csv`: inner ablations and rejection decisions.
- `feature_upgrade_summary.txt` and `.json`: metrics, configuration, limitations,
  hashes, selected settings, and runtime.
- `feature_upgrade_model.pkl` and `prediction_notebook.ipynb`: the upload pair.
- `run_full.log`: progress from the first full run in this chat.

`cache_upgrade/` stores compressed cross-fitted predictions and decisions, not
hundreds of fitted model copies. Cache keys include data, source, configuration,
and library version hashes. The final pickle stores each fixed legacy expert
once. Stale configurations are not silently reused.

## Submission

Upload `prediction_notebook.ipynb` and `feature_upgrade_model.pkl` together.
The notebook contains all inference functions, uses only platform libraries,
reads `DATATHON_INPUT_PATH`, and writes one `prediction` column to
`DATATHON_OUTPUT_PATH`. It does not need IDs, training data, helper Python files,
or an internet connection. The local candidate CSV includes IDs for auditing.

## Checks

```sh
/Users/ethanis/Downloads/datathon-platform-env/bin/python -m unittest discover -s platform_submission_v3 -p 'test_upgrade.py' -v
```

These checks cover target exclusion, original flags, safe surprises, observed
value preservation, hierarchy fallbacks, complete mask coverage, and keeping
all copies of a source row in its held-out partition. The full run also checks
v2 prediction reproduction, frozen DL predictions, finite outputs, IDs, row
counts, and column order.

After building the artifact, validate the notebook independently:

```sh
/Users/ethanis/Downloads/datathon-platform-env/bin/python platform_submission_v3/validate_submission.py
```

This runs the notebook in a temporary directory containing only the model and
the inference inputs. It checks an exact candidate match, repeated inference,
shuffled rows without IDs, and all sixteen missing patterns.

## Completed FULL run

The upgrade passed the nested-CV acceptance screen. Natural non-DL OOF RMSE
changed from **3.071177 to 3.068562**; mean balanced stress RMSE changed from
**4.034054 to 3.836695**. All five balanced folds improved. The largest gains
were in patterns missing occupancy and previous usage together. The extra
feature families were rejected; the final feature family is `core`.

The isolated notebook checks and all five unit checks passed. See
`IMPLEMENTATION_REPORT.md` for fold and pattern tables, imputation scores,
retained settings, artifact sizes, and the actual runtime.

The first run needed one numerical Ridge SVD repair. The fallback uses LAPACK
`gesvd` with the same RidgeCV objective and alpha grid when `gesdd` fails to
converge. Its equivalence is covered by a unit check. Previously completed cache
entries were copied only after checking that this was the sole training change;
their byte hashes and the source transition are recorded in
`cache_upgrade/solver_repair_manifest.json`. The successful retry's 9.3-minute
runtime includes cached work; it is not a cold-training benchmark.
