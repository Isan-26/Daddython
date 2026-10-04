# Final submission: v6 (public RMSE 3.24618)

## Upload checklist

| Platform card | File |
|---|---|
| 1. Prediction notebook | `prediction_notebook.ipynb` |
| 2. Trained model | `model.pkl` (87 MB) |
| 3. requirements.txt | none needed (only pre-installed libraries are used) |
| Technical report | `technical_report.pdf` (1 page, 5 KB) |

**Reference CSV:** keep the platform's default, the latest valid submission #4003 (3.24618). The notebook reproduces it exactly.

## Checks (all passed)

* Run like the platform does, in a folder containing **only** `prediction_notebook.ipynb` and `model.pkl`: shuffled test file without `id`, paths from `DATATHON_INPUT_PATH` / `DATATHON_OUTPUT_PATH`. Two runs gave identical output: one `prediction` column, 3,000 rows, about 16 seconds.
* Un-shuffled, that output equals `reference_predictions.csv` (= submission #4003) exactly.
* The notebook's code is the same as `../platform_submission_v6/prediction_notebook.ipynb`, split into commented sections (plus two import lines).
* `technical_report.pdf`: 1 page, US Letter, 5 KB; follows the organizers' `report_format.pdf` (five sections).

## Files

* `prediction_notebook.ipynb`, `model.pkl`: upload pair (inference only; no training cells).
* `technical_report.pdf`: report upload; rebuilt by `report_source/make_report.py` (needs `reportlab`).
* `reference_predictions.csv`: expected predictions for `test.csv` (`id,prediction`).

How the model was built and validated: `../platform_submission_v6/README.md`, and v4/v2 in their own folders.
