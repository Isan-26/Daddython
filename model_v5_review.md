# Second-round improvement: candidate v5

Your confirmed competition best is **`candidate_v4.csv`, RMSE 3.37**. The next file to try is **`candidate_v5.csv`**. Its competition score is not yet known; a score of 3.20 has not been verified.

## What changed

Four CatBoost models estimate missing temperature, humidity, occupancy, and previous usage from the other available inputs, building identity, and calendar. They learn from the observed input values in both train and test. Their targets are these input values, rather than test energy usage.

The imputer training includes three copies of examples with missing patterns sampled from the actual test file. Building/event weights move its training distribution toward the test mix, where unusual events are more frequent. The selected exponent is 1.5.

Typical previous-usage and occupancy estimates also become additional energy-regression features, including building-type interactions. The energy model fits labelled training rows using the v3 feature design plus these typical-load features, with an 80/20 ridge/residual-tree blend.

The result is blended **50/50 with your exact 3.37 submission file**. A stored hash checks that the baseline file matches the confirmed version. Each imputer uses 1,100 CatBoost iterations and depth 5.

## Results

Settings were selected on 6,400 development rows using three folds and four masking trials per fold. The remaining 1,600 rows were excluded from second-round selection. Some were used during the first round; this is a second-round holdout rather than a new untouched dataset.

| Check | Current 3.37 model, reproduced locally | V5 |
|---|---:|---:|
| Development, ordinary | 3.1121 | 3.0884 |
| Development, test missing patterns | 3.2355 | 3.2285 |
| Development, building/event weighted | 3.4635 | 3.4036 |
| Second-round holdout, ordinary | 3.0578 | 3.0546 |
| Second-round holdout, test missing patterns | 3.1937 | 3.1995 |
| Second-round holdout, building/event weighted | 3.3751 | 3.3174 |

Lower is better. The gain is concentrated in the check accounting for the different test building/event mix. Ordinary holdout RMSE is essentially unchanged; missing patterns alone are slightly worse. The weighted check is a distribution-shift diagnostic, **not a promised competition score**.

The holdout mask check uses 1,510 complete rows and twelve draws of actual test missing patterns. A paired bootstrap resampled rows to account for repetition. Its 95% interval for weighted RMSE improvement was 0.010 to 0.125, conditional on selected settings and estimated weights. The missing-pattern-only improvement interval included zero.

Other approaches tried included sharing building slopes by type, CatBoost corrections to supervised pattern models, direct imputation without typical-load features, different tree depths, and different distribution weights. Scripts and development results are under `experiments/round2/`.

## Use the result

`candidate_v5.csv` has 3,000 rows, the same `id,prediction` format, and the original test ID order. The new component was fitted using all 8,000 labelled training rows. It is ready for the next competition submission.

`energy_model_v5.ipynb` shows the results and regenerates the file. `energy_model_v5.py` is the entry point. Supporting model code is in `experiments/round2_models.py`; keep it with the project. `validate_energy_model_v5.py` reproduces the fixed-settings holdout comparison.

CatBoost 1.2.10 was installed in the project's `.venv`; `requirements_v5.txt` records that additional dependency. Its handling of numeric missing values is described in the [official documentation](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing).
