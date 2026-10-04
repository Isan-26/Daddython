# V4: frozen v2 with multiple-imputation inference

The requested uncertainty-averaging experiment is implemented here. **No model is retrained.** V4 uses the original v2 model objects, preprocessing, features, neural networks, ensemble weights and missing-previous +0.8 correction. Original v2 files remain untouched.

The main candidate is experimental. There is no validated new energy RMSE or evidence yet that it beats v2 on the hidden test labels.

## Files to use

For the platform notebook workflow, upload **prediction_notebook.ipynb and model.pkl together**. The default notebook setting is v4_uncertainty and matches candidate_v4_uncertainty.csv.

For CSV submission, each candidate has 3,000 rows, columns id,prediction and original test order:

| Configuration | CSV | Behavior |
|---|---|---|
| v4_uncertainty (default) | candidate_v4_uncertainty.csv | Average 64 sampled fills through the frozen imputed energy branch |
| v4_conservative | candidate_v4_conservative.csv | Apply 25% of the averaging change; this is a fixed optional setting, not a validated winner |
| v2_fallback | candidate_v2_fallback.csv | Exact original platform v2 predictions |

To select another notebook configuration, change VERSION in the final cell. The standalone notebook reads DATATHON_INPUT_PATH, writes one prediction column to DATATHON_OUTPUT_PATH, and needs no training data, helper modules or network access.

## What changes

1. Build an empirical uncertainty table from complete **training input** rows. The energy label, test inputs and leaderboard scores are excluded from the table and all configuration choices.
2. For each original missingness pattern, use the frozen v2 imputers to calculate donor reconstruction errors. Find 64 nearby feature donors, preferring the same building and using calendar context plus originally observed numeric inputs.
3. Jointly sample centered donor error vectors around the existing imputer's point estimates. Use positive/negative pairs to preserve each point estimate as the sample mean. Only original blanks receive perturbations; draws stay within symmetric training-support bounds.
4. Feed each draw through the **existing** Ridge and HGB energy estimators. Their expected-occupancy and expected-previous features continue to use original raw inputs. Average the energy predictions, not the imputed input values.
5. Replace only the LightGBM-imputed energy branch's prediction with its average. Its final ensemble weight remains 0.619875. Direct pattern experts, the MissForest branch and all neural predictions retain original inputs and predictions. Original missingness decisions and the +0.8 shift are preserved. Complete rows remain exactly v2.

Sampling is deterministic by row contents, with no dependence on IDs, labels, row order or batching. Fixed-size draw blocks control memory usage.

## Limits

The donor errors are **in-sample auxiliary residuals** from the frozen imputers, so their spread can be optimistic. The centered symmetric bootstrap is an empirical approximation, not a calibrated posterior. Self donors are excluded from sampling diagnostics, but the underlying imputers still saw those input values during their original training.

All saved energy models were trained on the original training labels. Without saved matching fold models or permission to refit, evaluating them on those labels cannot provide an honest new OOF RMSE. Therefore this experiment does not select a setting using training energy errors and makes no leaderboard improvement claim.

The main candidate actively changes incomplete-row predictions. The fallback is available explicitly; passing inference checks does not automatically mean the main candidate should replace the best submission.

## Rebuild and verify

Use the original model-compatible Python environment:

```sh
/Users/ethanis/Downloads/datathon-platform-env/bin/python platform_submission_v4/build_submission.py
/Users/ethanis/Downloads/datathon-platform-env/bin/python -m unittest discover -s platform_submission_v4 -p 'test_uncertainty.py' -v
/Users/ethanis/Downloads/datathon-platform-env/bin/python platform_submission_v4/validate_submission.py
```

The builder blocks estimator fit calls, verifies the original model's full serialized-state hash before and after inference, and checks original-file hashes. Cached sampling tables are keyed by original model, training data, configuration and source hashes.

See v4_summary.json for provenance and measured prediction changes, sampling_diagnostics.csv for auxiliary sampling checks, sampling_convergence.csv for Monte Carlo stability, prediction_changes_by_pattern.csv for affected rows, and validation_results.json for isolated-notebook checks.
