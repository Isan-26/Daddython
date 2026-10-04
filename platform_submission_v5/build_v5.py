"""Build platform submission v5 from frozen artifacts: v4 + the LightGBM-imputed branch averaged over 10 seeds.

The only change from v4: the LightGBM-imputed Ridge/HGB branch (final weight 0.619875) uses the average of the
ten already-fitted seeds 527-536 (from platform_submission/model.pkl) instead of seed 527 alone, with v4's
multiple-imputation averaging applied to each seed. No estimator is fitted (fit methods are blocked).
Nothing was selected with energy labels or leaderboard scores.
"""
from pathlib import Path
import contextlib
import hashlib
import json
import sys
import time
import warnings
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from uncertainty_runtime import *

warnings.filterwarnings('ignore')
SEEDS = list(range(527, 537))
NEIGHBORS, SAMPLES = 64, 64          # as in v4


def log(value):
    print(value, flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def forbid_training():
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    with contextlib.ExitStack() as stack:
        for estimator in [LGBMRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
                          IterativeImputer, Ridge, RidgeCV, MLPRegressor, Pipeline, StandardScaler]:
            stack.enter_context(patch.object(estimator, 'fit',
                side_effect=AssertionError('v5 must never fit an estimator')))
        yield


def build_uncertainty(parts, inputs, neighbors):
    # Complete training-input donors only. No energy label or test input enters
    # this table. Frozen imputers exclude their own numeric target at inference.
    donors = inputs.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)
    truth = donors[NUMERIC_COLUMNS].to_numpy(float)
    assert len(donors) > neighbors
    residuals = {}
    for pattern in range(1, 16):
        log(f'  frozen-imputer inference for donor mask {pattern}/15')
        masked = donors.copy()
        missing = (pattern & (1 << np.arange(4))) > 0
        masked.loc[:, np.array(NUMERIC_COLUMNS)[missing]] = np.nan
        matrix = np.zeros_like(truth)
        features = imputer_inputs(masked)
        for j in np.flatnonzero(missing):
            column = NUMERIC_COLUMNS[j]
            expected = parts['imputers'][column].predict(features.drop(columns=column))
            matrix[:, j] = truth[:, j] - expected
        residuals[pattern] = matrix.astype(np.float32)
    values = inputs[NUMERIC_COLUMNS].to_numpy(float)
    lower, upper = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
    lower[1:] = np.maximum(lower[1:], 0.)
    upper[1] = min(upper[1], 100.)
    state = {
        'reference': {'numeric': truth, 'calendar': calendar_inputs(donors),
                      'building': donors.building_id.to_numpy()},
        'residuals': residuals,
        'numeric_scale': np.maximum(truth.std(axis=0), 1e-6),
        'bounds': np.vstack([lower, upper]),
        'neighbors': neighbors, 'spread': 1.,
        'method': 'nearby same-building centered symmetric residual bootstrap',
        'limitation': 'Residuals use frozen imputers on their training donors; '
                      'they are in-sample auxiliary residuals, not an OOF-calibrated posterior.',
    }
    return state




def write_notebook():
    runtime = (HERE / 'uncertainty_runtime.py').read_text().replace('from frozen_v2 import *\n', '')
    source = (HERE / 'frozen_v2.py').read_text() + '\n' + runtime
    final = """import os
import warnings
import joblib
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore')
VERSION = 'v5_seed_average'  # alternatives: v4_uncertainty (exact v4), v2_fallback (exact v2)
INPUT_PATH = os.environ.get('DATATHON_INPUT_PATH', 'test.csv')
OUTPUT_PATH = os.environ.get('DATATHON_OUTPUT_PATH', 'predictions.csv')
bundle = joblib.load('model.pkl')
rows = pd.read_csv(INPUT_PATH).drop(columns=['id', 'energy_usage'], errors='ignore')
with threadpool_limits(limits=1):
    prediction = np.round(predict_with_uncertainty(bundle, rows, VERSION), 6)
assert len(prediction) == len(rows) and np.isfinite(prediction).all()
pd.DataFrame({'prediction': prediction}).to_csv(OUTPUT_PATH, index=False)
print(f'wrote {len(rows):,} predictions using {VERSION} to {OUTPUT_PATH}')
"""
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Platform submission v5: v4 with the LightGBM-imputed branch averaged over 10 seeds\n',
            'Upload this notebook together with `model.pkl`. No models are retrained.\n',
            'Default `v5_seed_average`: as v4 (frozen v2 + multiple-imputation averaging), but the LightGBM-imputed branch\n',
            'averages the ten already-fitted seeds 527-536 instead of seed 527 alone.\n',
            '`v4_uncertainty` reproduces v4 exactly; `v2_fallback` reproduces v2 exactly.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': final.splitlines(True)},
    ]
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
    (HERE / 'prediction_notebook.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


def main():
    start = time.time()
    paths = {'v4_model': ROOT / 'platform_submission_v4/model.pkl', 'seed_source': ROOT / 'platform_submission/model.pkl',
             'train': ROOT / 'train.csv', 'test': ROOT / 'test.csv'}
    hashes = {key: digest(path) for key, path in paths.items()}
    inputs = pd.read_csv(paths['train']).drop(columns=['id', 'energy_usage'], errors='ignore')
    test = pd.read_csv(paths['test'])
    test_rows = test.drop(columns=['id', 'energy_usage'], errors='ignore')
    bundle = joblib.load(paths['v4_model'])
    seed_source = joblib.load(paths['seed_source'])
    v2 = bundle['v2']
    state_527 = joblib.hash(v2['lightgbm_imputed'][527])

    with threadpool_limits(limits=4), forbid_training():
        log('Check: seed 527 in platform_submission/model.pkl is the same model as in v4')
        a = predict_lightgbm_imputed(v2['lightgbm_imputed'][527], test_rows)
        b = predict_lightgbm_imputed(seed_source['lightgbm_imputed'][527], test_rows)
        same_527 = float(np.abs(a - b).max())
        assert same_527 < 1e-9, same_527
        for seed in SEEDS[1:]:
            v2['lightgbm_imputed'][seed] = seed_source['lightgbm_imputed'][seed]
        v2['versions']['dl13_prevfix_10seeds'] = dict(v2['versions']['dl13_prevfix'], seeds=SEEDS)
        for seed in SEEDS[1:]:
            log(f'Uncertainty table for frozen imputer seed {seed} (inference only)')
            bundle['uncertainty'][seed] = build_uncertainty(v2['lightgbm_imputed'][seed], inputs, NEIGHBORS)
        bundle['versions']['v5_seed_average'] = {'alpha': 1., 'samples': SAMPLES, 'seed': 2026, 'v2_version': 'dl13_prevfix_10seeds'}
        bundle['provenance'] = dict(bundle.get('provenance', {}), v5_sources=hashes)

        log('Predict v5 and re-check v4 / v2 reproduction')
        v5, details = predict_with_uncertainty(bundle, test_rows, 'v5_seed_average', return_details=True)
        v4_again = predict_with_uncertainty(bundle, test_rows, 'v4_uncertainty')
        v2_again = predict_with_uncertainty(bundle, test_rows, 'v2_fallback')
    v4_csv = pd.read_csv(ROOT / 'platform_submission_v4/candidate_v4_uncertainty.csv')
    v2_csv = pd.read_csv(ROOT / 'platform_submission_v2/platform_dl13_prevfix.csv')
    assert v4_csv.id.equals(test.id) and v2_csv.id.equals(test.id)
    v4_diff = float(np.abs(np.round(v4_again, 6) - v4_csv.prediction).max())
    v2_diff = float(np.abs(np.round(v2_again, 6) - v2_csv.prediction).max())
    assert v4_diff == 0 and v2_diff == 0, (v4_diff, v2_diff)
    assert joblib.hash(v2['lightgbm_imputed'][527]) == state_527, 'seed 527 model was modified'

    pd.DataFrame({'id': test.id, 'prediction': np.round(v5, 6)}).to_csv(HERE / 'candidate_v5_seed_average.csv', index=False)
    joblib.dump(bundle, HERE / 'model.pkl', compress=3)
    write_notebook()

    change = np.round(v5, 6) - v4_csv.prediction.to_numpy()
    pattern = details['pattern']
    records = [{'pattern': int(p), 'rows': int((pattern == p).sum()), 'mean_change_vs_v4': float(change[pattern == p].mean()),
                'rms_change_vs_v4': float(np.sqrt(np.mean(change[pattern == p] ** 2))),
                'max_abs_change_vs_v4': float(np.abs(change[pattern == p]).max())} for p in np.unique(pattern)]
    pd.DataFrame(records).to_csv(HERE / 'prediction_changes_vs_v4.csv', index=False)
    complete = pattern == 0
    summary = {'models_fitted': 0, 'seeds': SEEDS, 'seed_527_identical_in_both_sources': same_527,
               'v4_reproduced_max_difference': v4_diff, 'v2_reproduced_max_difference': v2_diff,
               'rms_change_vs_v4_complete_rows': float(np.sqrt(np.mean(change[complete] ** 2))),
               'rms_change_vs_v4_rows_with_blanks': float(np.sqrt(np.mean(change[~complete] ** 2))),
               'max_abs_change_vs_v4': float(np.abs(change).max()),
               'model_pkl_mb': (HERE / 'model.pkl').stat().st_size / 1e6, 'build_seconds': time.time() - start,
               'source_hashes': hashes}
    (HERE / 'v5_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    log(json.dumps({k: v for k, v in summary.items() if k != 'source_hashes'}, indent=1))


if __name__ == '__main__':
    main()
