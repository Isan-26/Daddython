# Third-round candidate: V6

Your confirmed competition best is **`candidate_v5.csv`, RMSE 3.31**. The next file to try is **`candidate_v6.csv`**. Its competition score is unknown; 3.20 has not been achieved in a verified submission.

## What changed

V6 estimates missing occupancy and previous usage using separate models for each combination of available numeric inputs. There are eight models for each field, covering the availability of the other three fields. Each model sees only the numeric fields available for its pattern, together with building identity and calendar features. This avoids asking a single imputer to cover every combination of blanks.

These CatBoost imputers use depth 4, 750 iterations, learning rate 0.05, and leaf regularization 20. Their training targets are observed occupancy and previous usage from train and test inputs. Test energy targets are never used. Temperature and humidity keep the V5 imputers. Expected occupancy and previous usage also replace the V5 typical-load regression features; the energy regression still uses labelled training rows.

V6 blends this component with the exact confirmed V5 file. The blend depends on the number of missing numeric inputs:

| Missing inputs | V5 weight | New component weight |
|---|---:|---:|
| None | 100% | 0% |
| One | 62.925% | 37.075% |
| Two or more | 38.642% | 61.358% |

The weights were fitted from development predictions made on rows excluded from each base model's training fold, minimizing weighted squared error under constraints from zero to one. They were fixed before the third-round holdout. The development scores used to select models and weights are selection results, not independent estimates of performance.

## Results and limitations

This round split the labelled data into 6,400 development and 1,600 reserved rows using seed 2028, stratified by building. Development used three folds and four draws of actual test missing patterns per fold. The reserved check used twelve mask draws on its 1,491 complete rows. Reserved rows were excluded from this round's selection, but some labels may have been used in earlier rounds.

| RMSE check | Reproduced V5 | V6 |
|---|---:|---:|
| Development, ordinary | 3.0807 | 3.0783 |
| Development, test missing patterns | 3.2232 | 3.2430 |
| Development, building/event weighted | 3.4232 | 3.4065 |
| Third-round holdout, ordinary | 3.0721 | 3.0759 |
| Third-round holdout, test missing patterns | 3.2316 | 3.2417 |
| Third-round holdout, building/event weighted | 3.4280 | 3.3961 |

Lower is better. **This change improves the building/event weighted check and worsens the unweighted missing-pattern check.** The test file has a different building/event mix, so the weighted check is the selection objective. That weighting is an estimate based on inputs, rather than a measured competition score. The official scoring subset and hidden targets remain unknown.

A paired bootstrap resampled original holdout rows, after averaging their errors over the twelve masks. Its 95% interval for weighted RMSE improvement was **0.0055 to 0.0647**. The unweighted interval was **−0.0178 to −0.0017**, favoring V5. These intervals are conditional on the chosen settings and estimated distribution weights; they do not account for earlier rounds' selection or uncertainty in the weighting method.

Other approaches tested this round included training the energy model on synthetic missing patterns, raw missing flags, weather-event probability features, different distribution weights, deeper conditional imputers, averaging several random seeds, global blends, and a small missing-pattern bias correction. The bias correction did not improve the selected development objective, so it was omitted. Results are retained under `experiments/round3/`.

## Files and verification

`candidate_v6.csv` contains 3,000 rows in the original `id,prediction` format and test ID order. The conditional component was refitted on all 8,000 labelled rows. The 2,453 complete-row predictions retain V5, and 547 incomplete rows receive the new blend. V5 is preserved as the confirmed best.

Submission generation checks the V5 file hash, input observation preservation, complete-row prediction preservation, finite outputs, and prediction consistency with repeated row indexes. `submission_v6_audit.json` records the data and output hashes and prediction changes.

`energy_model_v6.ipynb` shows the saved results and can regenerate the submission. `energy_model_v6.py` is its entry point, `model_v6_config.json` contains the fixed settings, and `validate_energy_model_v6.py` reproduces the reserved comparison. Keep the supporting model code in `experiments/round3_models.py`, `experiments/round2_models.py`, and the existing V4/V5 modules with the project. The existing `.venv` and `requirements_v5.txt` contain the required CatBoost dependency.
