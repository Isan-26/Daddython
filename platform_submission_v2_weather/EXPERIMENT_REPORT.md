# Completed v2 weather experiment

Implemented the three requested interactions in a separate Ridge correction:
occupancy × temperature, occupancy × humidity, and occupancy × max(temperature − 27, 0).
The v2 deployed models, imputers, feature schemas, ensemble weights and +0.8 correction
remain frozen. Only originally complete rows can receive a correction.

**Result: the addition did not pass validation.** Every nonzero tested weight
worsened the available OOF score, with no positive fold gains. The selected and
exported correction weight is **0**. The generated candidate_v2_weather.csv is
exactly identical to platform_submission_v2/platform_dl13_prevfix.csv; all
3,000 IDs, predictions and their order match. Continue using the existing v2
submission; this trial provides no new prediction improvement to submit.

## Validation evidence

| Correction weight | Natural OOF proxy RMSE | Complete-row OOF proxy RMSE |
| --- | --- | --- |
| 0.0 | 3.070753 | 3.023578 |
| 0.1 | 3.071004 | 3.023848 |
| 0.2 | 3.071263 | 3.024127 |
| 0.5 | 3.072089 | 3.025017 |
| 1.0 | 3.073628 | 3.026677 |

Ridge penalty was fixed at 100. Correction fitting and weight selection used
held-out folds. The predeclared screen required 0.005 complete-row gain, 0.004
overall gain, at least four improving folds, and no fold regression above 0.005.
All outer calibration folds selected zero correction.

The OOF comparison combines the platform v2 non-DL recipe, five saved sklearn
MLP networks, the average of five saved RealMLP seeds, and the original +0.8
missing-previous shift. It uses the original seed-0 fold partition to align
cached experts. Deployment retains all ten RealMLP seeds. **These scores are
full-ensemble OOF approximations, not an exact ten-seed end-to-end nested CV.**
The existing OOF caches also lack original runtime/hash metadata; target ordering,
source notebooks, fold recipes and current file hashes were checked. The
additional deployment-seed coverage guard also prevents activation.

## Preservation and checks

- No deep learning model was retrained.
- Temporary legacy non-DL models were fitted only for validation.
- Only the new small correction Ridge models were added.
- Six unit checks passed, including the nonzero-correction path, all missing
  patterns, source/target isolation, negligible-gain rejection and input preservation.
- The isolated upload notebook passed exact candidate matching, repeated inference,
  shuffled input without IDs, and all sixteen missing patterns.
- Every missing-data case matched the original v2 predictions.
- Hash checks confirmed the original v2 model and notebook were untouched.

The training/validation/export script ran in **1.7 minutes**,
reusing saved neural OOF predictions. Implementation and notebook checks were
additional work. No leaderboard feedback was used for this experiment.

## Files and commands

train_weather_correction.py builds and validates the trial; weather_runtime.py
contains the three features and the optional correction; frozen_v2.py contains
inference functions copied directly from the v2 notebook; baseline_training.py
contains the unchanged cheap-model recipe. Detailed results are in
weather_weight_trials.csv, weather_cv_results.csv, weather_oof_predictions.csv
and weather_summary.json.

    /Users/ethanis/Downloads/datathon-platform-env/bin/python "/Users/ethanis/Downloads/Track 1 Dataset/platform_submission_v2_weather/train_weather_correction.py"
    /Users/ethanis/Downloads/datathon-platform-env/bin/python "/Users/ethanis/Downloads/Track 1 Dataset/platform_submission_v2_weather/validate_submission.py"

The standalone upload pair is prediction_notebook.ipynb and model.pkl.
Its correction is disabled after this trial. The notebook needs no helper
Python files, training data or internet connection.
