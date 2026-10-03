# Platform submission v2: `dl13_prevfix`

**Upload these two files:** `prediction_notebook.ipynb` and `model.pkl` (76 MB). No `requirements.txt` is needed: everything runs on the platform's pre-installed libraries (`requirements-image.txt`: numpy, pandas, scikit-learn 1.5.2, lightgbm 4.5.0, joblib).

## What the model is

| Part | Weight | Code from |
|---|---|---|
| best model: 25% v4 + 75% LightGBM-imputed model (seed 527) | 0.95 × 0.87 | `best_model.ipynb` |
| scikit-learn neural network (5 networks averaged) | 0.05 × 0.87 | `neural_net_model.ipynb` |
| RealMLP (10 seeds averaged); trained with PyTorch, prediction step in numpy | 0.13 | `deep_learning_model.ipynb` |
| +0.8 on rows whose `previous_usage` is blank | | real blanks hide higher-than-typical usage (`deep_learning_model.ipynb`) |

`model.pkl` also has version `best` (25% v4 + 75% LightGBM-imputed model); set `VERSION` in the notebook to switch.

## Run it on another laptop

1. **Platform environment** (Python 3.12): `python3.12 -m venv datathon-platform-env && datathon-platform-env/bin/pip install -r requirements-image.txt ipykernel`
2. **Predict:** open `prediction_notebook.ipynb` with that environment and Run All. It reads `test.csv` from this folder (copy `../test.csv` here) or the file in `DATATHON_INPUT_PATH`, and writes `predictions.csv` (or `DATATHON_OUTPUT_PATH`).
3. **Rebuild `model.pkl` from scratch (optional, about 1 minute):** run `2_build_model_pkl.ipynb` with the same environment. It trains v4, the LightGBM-imputed model and the neural network from `../train.csv` / `../test.csv`, and loads RealMLP's weights from `realmlp_parts/`.
4. **Retrain RealMLP (optional):** needs PyTorch, so a second environment: `pip install -r requirements-dl-env.txt`, then run `1_export_realmlp.ipynb` (about 30 seconds per seed; set `RETRAIN = True`).

## Files

| File | Run with | What it does |
|---|---|---|
| `prediction_notebook.ipynb` | platform environment | **upload**: loads `model.pkl`, predicts the test file |
| `model.pkl` | | **upload** (76 MB, zlib-compressed) |
| `2_build_model_pkl.ipynb` | platform environment | trains every part from `train.csv` / `test.csv` and writes `model.pkl`, then checks it |
| `1_export_realmlp.ipynb` | `requirements-dl-env.txt` | trains RealMLP (10 seeds), saves the weights to `realmlp_parts/`, checks numpy vs PyTorch |
| `platform_dl13_prevfix.csv` | | what the prediction notebook writes for `test.csv` (with ids added) |
| `realmlp_numpy.py`, `export_realmlp.py`, `realmlp_parts/` | | numpy RealMLP code, the exporter and the exported weights |

**Note on v3's MissForest imputer:** the original recipe uses 100 trees, leaf size 3 and 8 rounds, which makes `model.pkl` about 280 MB. Like the previous platform `model.pkl`, this build uses 50 trees, leaf size 10 and 4 rounds (it reproduces that model's v4 to within 3e-10). The final prediction changes by only about 0.005 per row between the two.

## Checks (all passed)

* numpy RealMLP vs PyTorch: largest difference 2.4e-5 per seed, 7e-6 for the 10-seed average (11,000 rows).
* Neural network vs the one in `candidate_nn_w05.csv`: 1e-12. LightGBM-imputed model vs the one in `candidate_v8.csv`: 7e-14.
* Platform-style run (shuffled test file without `id`, paths from `DATATHON_INPUT_PATH` / `DATATHON_OUTPUT_PATH`), three times: identical output every run (single-threaded math, rounded to 6 decimals), one `prediction` column, 3,000 rows.
* `platform_dl13_prevfix.csv` vs `../candidate_dl13_prevfix.csv`: typical difference 0.037, because the platform's v4 is retrained with the platform library versions (the CSV used `candidate_v4.csv`).
