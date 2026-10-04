# Platform submission v6: v4 + engineered-input RealMLP ensemble (15%)

> **Result: public RMSE 3.24618, the best so far** (v4: 3.24754, v2: 3.24883). Recommended final submission.

**Upload:** `prediction_notebook.ipynb` and `model.pkl` (87 MB). Default `VERSION = 'v6_eng'`; `'v4_uncertainty'` and `'v2_fallback'` reproduce v4 and v2 exactly. Only pre-installed platform libraries are used: the new networks run as a numpy forward pass.

## What's new

v6 = **0.85 × v4 + 0.15 × average of 10 new RealMLP networks**. Nothing in v4 changes.

The new networks get the pipeline's **engineered inputs** instead of the crude median fills the existing RealMLP uses:
- the seed-527 LightGBM-imputed values, the expected previous usage and occupancy, deviations from typical, heat/rain flags and time curves (the 20 numeric inputs of `neural_net_model.ipynb`'s network);
- plus the four "was blank" flags, with building, building type and weekday as categories.

5 networks predict `energy_usage` (`eng`) and 5 predict the change from the filled `previous_usage` (`eng_delta`); the two kinds make different errors.

## Cross-validation evidence (same 5 folds as all earlier experiments)

Added to the v2-like ensemble (out-of-fold predictions), with the weight learned on 4 folds and scored on the 5th:

| New networks (in CV: 3 seeds each) | Alone: natural rows | Alone: simulated blanks | Gain: natural | Gain: simulated blanks | Gain: test-weighted | Weight per fold |
|---|---:|---:|---:|---:|---:|---|
| `eng` | 3.151 | 3.375 | +0.0010 | +0.0006 | +0.0085 | 0.09–0.13 |
| `eng_delta` | 3.233 | 3.461 | +0.0015 | +0.0009 | +0.0097 | 0.07–0.12 |
| **both (used)** | **3.135** | **3.366** | **+0.0024** | **+0.0012** | **+0.0142** | **0.14–0.18** |

For comparison, the existing RealMLP (5 seeds) scores 3.181 / 3.556 alone. The natural-row gain is positive in **all five folds** (+0.0011, +0.0007, +0.0030, +0.0022, +0.0050). The weight 0.15 is a round value inside the per-fold range; it wasn't tuned on the leaderboard.

**Caveats:** the CV ensemble is v2-like (v4's multiple imputation isn't in the out-of-fold predictions). The simulated-blank and test-weighted gains rely on randomly simulated blanks, which proved unreliable for v3. The final model averages 10 networks where CV used 6. No leaderboard improvement is promised.

## Checks (all passed)

* Network inputs computed by the platform code equal the training inputs (difference 0.0).
* numpy vs PyTorch: each network ≤ 2e-5; 10-network average 4e-6.
* `v4_uncertainty` and `v2_fallback` reproduce v4's and v2's CSVs exactly.
* Platform-style runs (shuffled test file without `id`, `DATATHON_INPUT_PATH` / `DATATHON_OUTPUT_PATH`), twice: identical output, equal to `candidate_v6_eng.csv`, about 16 seconds.
* Changes vs v4: RMS 0.13 on complete rows and 0.20 on rows with blanks; no systematic shift (mean change about +0.005).

## Files and rebuild

| File | Run with | What it does |
|---|---|---|
| `prediction_notebook.ipynb`, `model.pkl` | platform | **upload** |
| `candidate_v6_eng.csv` | | v6 predictions for `test.csv` (`id,prediction`) |
| `cv/nn_features_folds.py` | `datathon-platform-env` | engineered inputs per fold + final (`../experiments/nn_features_folds.npz`) |
| `cv/realmlp_variants.py` (+ `cv/dl_cv.py`) | `datathon-dl-env` | the cross-validation runs above |
| `train_realmlp_eng.py` | `datathon-dl-env` | trains the 10 final networks, exports them to `realmlp_eng_parts/`, checks numpy vs PyTorch |
| `build_v6.py` | `datathon-platform-env` | v4's `model.pkl` + the networks → `model.pkl`, notebook, CSV, `v6_summary.json` |
| `eng_runtime.py` | | the new prediction code; `frozen_v2.py` and `uncertainty_runtime.py` are v4's, unchanged |
