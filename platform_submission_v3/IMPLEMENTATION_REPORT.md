# Platform submission v3 — completed implementation

Created the new folder without changing `platform_submission_v2`. FULL mode completed with five outer folds, five inner folds, random state 2026 and imputer seeds 527–531. No public leaderboard feedback was used.

## Files and outputs

- `feature_imputation_upgrade.py`: nested training, selection, cache and export entry point.
- `upgrade_runtime.py`: new imputation, feature and ensemble inference functions.
- `legacy_runtime.py`, `legacy_training.py`: frozen v2 inference and inexpensive training recipes.
- `test_upgrade.py`, `validate_submission.py`: leakage and isolated inference checks.
- `prediction_notebook.ipynb`, `feature_upgrade_model.pkl`: standalone upload pair.
- `candidate_feature_upgrade.csv`: exactly 3,000 rows and `id,prediction`, in original test order.
- `feature_upgrade_cv_results.csv`, `imputation_cv_results.csv`, `missing_pattern_results.csv`, `feature_family_results.csv`: detailed validation tables.
- `feature_upgrade_summary.txt`, `feature_upgrade_summary.json`, `validation_results.json`, this report and `README.md`: results and usage.
- `requirements-image.txt`, `.gitignore`, `run_full.log`, `validation.log`: environment, cache exclusions and execution logs.

## Exact reproduction and preserved architecture

The saved v2 `dl13_prevfix` submission was reproduced exactly at its six-decimal export precision: maximum prediction difference **0.0**.

The original non-DL branch is 0.0625 V3 + 0.1875 direct missing-pattern experts + 0.75 LightGBM-imputed Ridge/residual-HGB. V3 uses the existing slim MissForest/Ridge/residual-HGB recipe. Pattern experts retain their Ridge/residual-HGB architecture. The new branch keeps 0.0625 V3 and replaces the remaining 0.9375 with coarse, pattern-specific direct/imputed blends.

The final ensemble keeps **0.8265 non-DL + 0.0435 sklearn MLP + 0.13 RealMLP**. All five pretrained sklearn MLPs retain their 40-input schema, imputer, scaling and target normalization. All ten RealMLP weight sets retain their 15 numeric inputs, category coding and median filling. Their predictions were checked unchanged after the cheap models were fitted.

Retrained models: twenty LightGBM imputers (four targets × five seeds), four ExtraTrees imputers, training-only smoothed median tables, the selected Ridge/residual-HGB energy model and sixteen Ridge/HGB pattern experts. Cheap legacy models were also refitted for the same-fold baseline. **No deep learning model was retrained.**

## Features and imputation

The retained core adds OOF expected temperature, humidity, occupancy and previous usage, the expected occupancy/previous type interactions, four original missing flags, missing count, bitmask and pattern indicators. Existing cyclic, event, building/type, interaction and typical-value features remain.

Each source row and all artificial copies share a held-out fold. Its own auxiliary target and energy label are excluded from the generating imputer; reconstruction-weight selection also excludes that source's targets. Hierarchy tables are fold-specific. New model fitting and selections use training data only; the legacy expert/baseline retains its original preprocessing.

LightGBM means/stds, ExtraTrees and hierarchical medians with smoothing 20 were implemented. The selected final convex weights are:

| Feature | LightGBM | ExtraTrees | Hierarchy |
| --- | --- | --- | --- |
| temperature | 1.000000 | 0.000000 | 0.000000 |
| humidity | 0.400000 | 0.600000 | 0.000000 |
| occupancy | 0.600000 | 0.400000 | 0.000000 |
| previous_usage | 0.600000 | 0.400000 | 0.000000 |

The hierarchy was evaluated but received zero final imputation-blend weight. Two-pass imputation was retained for patterns **5, 6, 9, 10, 11, 12, 13, 14**; all other patterns use one pass. Second pass updates only originally missing inputs. The arbitrary +0.8 correction was removed. CV selected **no additional missing-previous bias correction** (offset 0).

The extra uncertainty, weekday/building, smooth weather, hierarchical deviation/surprise, physical-interaction and combined feature families were tested and rejected by the improvement/consistency screen. Some had small gains below the 0.004 balanced-RMSE threshold. Final selection diagnostics, before strategy optimization, are:

| family | balanced_rmse | natural_rmse | eligible |
| --- | --- | --- | --- |
| core | 3.854977 | 3.070358 | True |
| uncertainty | 3.856175 | 3.068757 | False |
| weekday | 3.858698 | 3.073605 | False |
| weather | 3.853045 | 3.068336 | False |
| deviations | 3.855525 | 3.072227 | False |
| physical | 3.853701 | 3.069371 | False |
| combined | 3.859290 | 3.074191 | False |

## Honest non-DL OOF comparison

Natural OOF RMSE: **3.071177 → 3.068562** (gain 0.002615). This is a small improvement on the original rows.

Mean balanced stress RMSE: **4.034054 → 3.836695**. Balanced fold standard deviation: **0.110813 → 0.095642**. Natural fold standard deviation: **0.102008 → 0.107162**. Three natural folds and all five balanced folds improved.

Balanced stress weights are 50% natural validation rows and 50% uniformly weighted artificial patterns 1–15 on complete validation rows. They test missing-data robustness and do not estimate the hidden test distribution. Feature/weight decisions for every outer fold come from its training partition. The upgrade passed the predefined acceptance screen.

| Fold | Natural baseline | Natural upgrade | Balanced baseline | Balanced upgrade |
| --- | --- | --- | --- | --- |
| 1 | 3.077791 | 3.073123 | 4.035107 | 3.834081 |
| 2 | 3.031081 | 3.036465 | 3.994364 | 3.787970 |
| 3 | 3.172660 | 3.171789 | 4.144524 | 3.936824 |
| 4 | 2.917846 | 2.902918 | 3.870156 | 3.705081 |
| 5 | 3.149731 | 3.151030 | 4.126119 | 3.919521 |

## Missing-pattern comparison

Pattern 0 contains original complete validation rows. Other patterns combine naturally missing validation rows with artificially masked complete validation rows. Counts therefore include repeated source rows across patterns. Final strategy columns show the all-training inference choice; outer-fold scores use the choice learned within each outer training partition. Direct, one-pass and two-pass expert errors are also in `missing_pattern_results.csv`.

| Pattern | Missing inputs | Rows | Baseline RMSE | Upgrade RMSE | Final passes | Final direct weight |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | none | 7537 | 3.029269 | 3.022511 | one_pass | 0.500000 |
| 1 | temperature | 7651 | 3.376009 | 3.376446 | one_pass | 0.200000 |
| 2 | humidity | 7648 | 3.033292 | 3.024843 | one_pass | 0.600000 |
| 3 | temperature + humidity | 7540 | 3.468825 | 3.446472 | one_pass | 0.200000 |
| 4 | occupancy | 7654 | 3.720371 | 3.673292 | one_pass | 0.100000 |
| 5 | temperature + occupancy | 7538 | 4.010845 | 3.897803 | two_pass | 0.100000 |
| 6 | humidity + occupancy | 7540 | 3.732292 | 3.649162 | two_pass | 0.000000 |
| 7 | temperature + humidity + occupancy | 7537 | 4.089746 | 3.980481 | one_pass | 0.200000 |
| 8 | previous_usage | 7647 | 3.911610 | 3.880290 | one_pass | 0.600000 |
| 9 | temperature + previous_usage | 7538 | 4.680509 | 4.610208 | two_pass | 0.400000 |
| 10 | humidity + previous_usage | 7539 | 3.935354 | 3.878970 | two_pass | 0.600000 |
| 11 | temperature + humidity + previous_usage | 7537 | 4.961699 | 4.754503 | two_pass | 0.600000 |
| 12 | occupancy + previous_usage | 7538 | 6.176864 | 5.429452 | two_pass | 0.400000 |
| 13 | temperature + occupancy + previous_usage | 7537 | 6.949311 | 6.150239 | two_pass | 0.400000 |
| 14 | humidity + occupancy + previous_usage | 7537 | 6.161137 | 5.436829 | two_pass | 0.400000 |
| 15 | temperature + humidity + occupancy + previous_usage | 7537 | 7.224073 | 6.294295 | one_pass | 0.400000 |

## Imputation reconstruction comparison

Mean of five outer-validation reconstruction RMSEs on artificially masked, originally known values. The ensemble uses weights selected from outer training data, never validation truth. Two-pass column below is the LightGBM component alone; downstream pattern selection assesses the full convex blend.

| feature | lightgbm | extratrees | hierarchical | ensemble | two_pass_lightgbm |
| --- | --- | --- | --- | --- | --- |
| humidity | 4.582888 | 4.557123 | 5.005839 | 4.536330 | 4.588742 |
| occupancy | 25.751325 | 25.593183 | 32.616830 | 25.233462 | 25.444250 |
| previous_usage | 8.324012 | 8.348398 | 9.696531 | 8.208452 | 8.230803 |
| temperature | 0.935432 | 0.981288 | 1.273919 | 0.935432 | 0.937045 |

## Checks and practical limits

All five unit checks passed: auxiliary-target exclusion, original missingness/surprises/observed preservation, hierarchy fallback, complete mask coverage/source-fold isolation, and Ridge SVD fallback equivalence.

The notebook passed isolated execution with only the saved model and test inputs: exact candidate match, repeated predictions, shuffled rows without IDs and all sixteen patterns. Predictions are finite, IDs are unique and preserved, and column/order/count checks passed. Model size is **111.8 MB**.

Local inference emitted numerical-library warnings, but every output was finite and the repeat/shuffle checks matched the candidate exactly. These warnings are retained in `validation.log` for inspection.

The reported OOF scores cover the **non-DL branch only**. Existing DL OOF artifacts use different folds (seed 0), so no valid full-ensemble OOF result or DL-weight optimization is claimed. No leaderboard improvement is promised.

## Runtime and commands

Observed wall time through artifact export was approximately **25.4 minutes**, including the initial attempt and a numerical Ridge SVD repair. The successful retry took **9.3 minutes with verified completed-fold caches**; this is not a cold-run benchmark. The repair retries the same RidgeCV alpha grid/objective with LAPACK `gesvd` if `gesdd` fails. The cache transition is documented in `cache_upgrade/solver_repair_manifest.json`.

Run the complete default FULL workflow from any directory:

```sh
"/Users/ethanis/Downloads/datathon-platform-env/bin/python" "/Users/ethanis/Downloads/Track 1 Dataset/platform_submission_v3/feature_imputation_upgrade.py"
```

Use `--fast` only for development (three folds and three imputer seeds). Use `--check-only` for legacy reproduction without training.

Run isolated submission checks after training:

```sh
"/Users/ethanis/Downloads/datathon-platform-env/bin/python" "/Users/ethanis/Downloads/Track 1 Dataset/platform_submission_v3/validate_submission.py"
```

Exact local submission filename: **`candidate_feature_upgrade.csv`**. For the platform notebook workflow, upload **`prediction_notebook.ipynb` and `feature_upgrade_model.pkl` together**. The notebook reads `DATATHON_INPUT_PATH`, writes one prediction column to `DATATHON_OUTPUT_PATH`, and needs no training files, helper modules or network connection.
