# Frozen v2 weather experiment

This additive folder tests exactly three interactions: occupancy × temperature,
occupancy × humidity, and occupancy × max(temperature − 27, 0). A separate,
regularized Ridge learns a correction to the v2 ensemble's OOF residuals.

All deployed v2 models, imputers, feature schemas, ensemble weights and the +0.8
missing-previous correction remain unchanged. The added model updates complete
rows only. Every originally incomplete row always receives exactly v2's
prediction. Zero correction is the default.

Run with the existing platform environment:

```sh
/Users/ethanis/Downloads/datathon-platform-env/bin/python platform_submission_v2_weather/train_weather_correction.py
```

`--check-only` reproduces v2 without fitting anything. The full run reuses saved
sklearn MLP and RealMLP OOF predictions and fits only temporary legacy non-DL
validation models and small Ridge correction models. No neural network is
trained. Frozen inference functions are copied directly from the v2 notebook.

Validation uses the original five shuffled folds with seed 0 to align the saved
OOF experts. Legacy non-DL components are refitted with the actual platform v2
recipe on those folds. Test-pattern preprocessing exists only inside the old
baseline recipe; correction training, selection and scoring use naturally
observed training rows, without test-distribution weights or leaderboard scores.

The saved RealMLP OOF files cover five seeds, while deployment uses ten. The
script reports this as a full-ensemble OOF approximation and disables the
correction while deployment-seed coverage is incomplete. Existing caches also
lack original runtime/hash metadata; truth ordering and notebook provenance are
checked. Correction fitting and shrinkage selection are held out by fold, but
calibration of an existing OOF table is not an end-to-end nested refit of every
baseline model. No exact ten-seed full-pipeline OOF score is claimed.

The correction uses only three standardized inputs, Ridge penalty 100, and a
coarse shrinkage grid of 0, 0.1, 0.2, 0.5 and 1. Predeclared acceptance requires
at least 0.005 complete-row gain, 0.004 overall gain, improvement in at least
four folds, and no fold regression greater than 0.005. A failed check exports
zero correction and a CSV identical to v2.

Outputs are `candidate_v2_weather.csv`, `weather_summary.json`,
`weather_cv_results.csv`, `weather_weight_trials.csv`, and
`weather_oof_predictions.csv`. `prediction_notebook.ipynb` and `model.pkl` are
the standalone platform upload pair. The notebook needs no training data or
helper files and respects `DATATHON_INPUT_PATH` / `DATATHON_OUTPUT_PATH`.

Checks:

```sh
/Users/ethanis/Downloads/datathon-platform-env/bin/python -m unittest discover -s platform_submission_v2_weather -p 'test_weather.py' -v
/Users/ethanis/Downloads/datathon-platform-env/bin/python platform_submission_v2_weather/validate_submission.py
```

Caches are local, excluded from Git, and keyed by data, code, model, fold and
library-version hashes. The v2 source files and model are never overwritten.
See `weather_summary.json` and `run.log` for the completed experiment's decision
and runtime.

## Completed result

The correction was rejected: every nonzero tested weight worsened the available
OOF result. Exported weight is zero, so the candidate is identical to the saved
v2 platform CSV. Six unit checks and the isolated notebook checks passed.
See EXPERIMENT_REPORT.md for scores, limitations and the preservation checks.
