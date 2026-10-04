"""Build a train-free v4 upload pair from frozen v2 artifacts.

No estimator is fitted. All three CSVs are fixed configurations, not choices
selected using training energy labels or public leaderboard scores.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import time
import warnings
from unittest.mock import patch

import joblib
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from uncertainty_runtime import *

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
warnings.filterwarnings('ignore')


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
                side_effect=AssertionError('v4 must never fit an estimator')))
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


def sampling_diagnostics(bundle, inputs):
    donors = inputs.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)
    rng = np.random.default_rng(2026)
    source = rng.choice(len(donors), min(600, len(donors)), replace=False)
    raw = donors.iloc[source].reset_index(drop=True)
    truth = raw[NUMERIC_COLUMNS].to_numpy(float)
    patterns = rng.integers(1, 16, len(raw))
    missing = (patterns[:, None] & (1 << np.arange(4))) > 0
    raw[NUMERIC_COLUMNS] = raw[NUMERIC_COLUMNS].mask(missing)
    seed = bundle['v2']['versions']['dl13_prevfix']['seeds'][0]
    parts = bundle['v2']['lightgbm_imputed'][seed]
    _, means = expected_and_filled(parts, raw)
    draws = sample_missing_inputs(bundle['uncertainty'][seed], raw, means,
                                  samples=64, exclude_reference=source)
    np.testing.assert_allclose(draws.mean(axis=1), means, rtol=0, atol=1e-10)
    records = []
    for j, column in enumerate(NUMERIC_COLUMNS):
        use = missing[:, j]
        low, high = np.quantile(draws[use, :, j], [.05, .95], axis=1)
        records.append({
            'column': column, 'masked_rows': int(use.sum()),
            'point_reconstruction_rmse': float(np.sqrt(np.mean((truth[use, j] - means[use, j]) ** 2))),
            'empirical_90pct_interval_coverage': float(np.mean((truth[use, j] >= low) & (truth[use, j] <= high))),
            'mean_interval_width': float(np.mean(high - low)),
            'sampling_mean_max_abs_error': float(np.max(np.abs(draws[use, :, j].mean(axis=1) - means[use, j]))),
            'evaluation': 'auxiliary in-sample imputer diagnostic; self donor excluded; no energy RMSE',
        })
    return records


def convergence_diagnostics(bundle, test):
    raw = test.loc[missing_patterns(test) > 0].iloc[:150].reset_index(drop=True)
    seed = bundle['v2']['versions']['dl13_prevfix']['seeds'][0]
    parts = bundle['v2']['lightgbm_imputed'][seed]
    expected, means = expected_and_filled(parts, raw)
    predictions = {}
    for count in [32, 64, 128]:
        draws = sample_missing_inputs(bundle['uncertainty'][seed], raw, means, count)
        predictions[count] = average_energy_predictions(parts, raw, draws, expected)
    coefficient = imputed_branch_weight(bundle['v2']['versions']['dl13_prevfix'])
    return [{'samples': count, 'rows': len(raw),
             'rms_final_prediction_difference_vs_128': float(np.sqrt(np.mean((coefficient * (pred - predictions[128])) ** 2))),
             'max_abs_final_prediction_difference_vs_128': float(np.max(np.abs(coefficient * (pred - predictions[128])))),
             'evaluation': 'Monte Carlo stability only; no labels or configuration selection'}
            for count, pred in predictions.items()]


def write_notebook():
    runtime = (HERE / 'uncertainty_runtime.py').read_text().replace('from frozen_v2 import *\n', '')
    source = (HERE / 'frozen_v2.py').read_text() + '\n' + runtime
    final = '''import os
import warnings
import joblib
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore')
VERSION = 'v4_uncertainty'  # Experimental; alternatives: v4_conservative, v2_fallback.
INPUT_PATH = os.environ.get('DATATHON_INPUT_PATH', 'test.csv')
OUTPUT_PATH = os.environ.get('DATATHON_OUTPUT_PATH', 'predictions.csv')
bundle = joblib.load('model.pkl')
rows = pd.read_csv(INPUT_PATH).drop(columns=['id', 'energy_usage'], errors='ignore')
with threadpool_limits(limits=1):
    prediction = np.round(predict_with_uncertainty(bundle, rows, VERSION), 6)
assert len(prediction) == len(rows) and np.isfinite(prediction).all()
pd.DataFrame({'prediction': prediction}).to_csv(OUTPUT_PATH, index=False)
print(f'wrote {len(rows):,} predictions using {VERSION} to {OUTPUT_PATH}')
'''
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Frozen v2 with experimental multiple-imputation inference\n',
            'Upload this notebook together with `model.pkl`. No models are retrained.\n',
            'Default: average 64 sampled fills through the existing imputed energy branch.\n',
            'Neural predictions, missing-pattern experts, ensemble weights and +0.8 remain v2.\n',
            'No honest new OOF RMSE is available under the no-retraining constraint.\n',
            'Use `VERSION = "v2_fallback"` for exact original v2 inference.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': source.splitlines(True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': final.splitlines(True)},
    ]
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
    (HERE / 'prediction_notebook.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--neighbors', type=int, default=64)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2 or args.neighbors < 2:
        parser.error('samples must be even and >= 2; neighbors must be >= 2')
    start = time.time()
    paths = {'v2_model': ROOT / 'platform_submission_v2/model.pkl',
             'v2_notebook': ROOT / 'platform_submission_v2/prediction_notebook.ipynb',
             'train': ROOT / 'train.csv', 'test': ROOT / 'test.csv'}
    hashes = {key: digest(path) for key, path in paths.items()}
    inputs = pd.read_csv(paths['train']).drop(columns=['id', 'energy_usage'], errors='ignore')
    test = pd.read_csv(paths['test'])
    v2 = joblib.load(paths['v2_model'])
    trained_state_before = joblib.hash(v2)
    settings = v2['versions']['dl13_prevfix']
    config = {'v2_model': hashes['v2_model'], 'train': hashes['train'],
              'neighbors': args.neighbors, 'spread': 1.,
              'builder': digest(Path(__file__)), 'runtime': digest(HERE / 'uncertainty_runtime.py')}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    cache = HERE / 'cache'
    cache.mkdir(exist_ok=True)
    cache_path = cache / f'uncertainty_{key}.pkl'
    with threadpool_limits(limits=4), forbid_training():
        if cache_path.exists():
            log('Reusing matching uncertainty tables; no fitting')
            uncertainty = joblib.load(cache_path)
        else:
            uncertainty = {}
            for seed in settings['seeds']:
                log(f'Build empirical uncertainty table for frozen imputer seed {seed}')
                uncertainty[seed] = build_uncertainty(v2['lightgbm_imputed'][seed], inputs, args.neighbors)
            joblib.dump(uncertainty, cache_path, compress=3)
        bundle = {'v2': v2, 'uncertainty': uncertainty, 'provenance': hashes,
                  'versions': {
                      'v4_uncertainty': {'alpha': 1., 'samples': args.samples, 'seed': 2026},
                      'v4_conservative': {'alpha': .25, 'samples': args.samples, 'seed': 2026},
                      'v2_fallback': {'alpha': 0., 'samples': args.samples, 'seed': 2026},
                  }}
        log('Predict v4 with frozen models and sampled numeric fills')
        prediction, details = predict_with_uncertainty(bundle, test, return_details=True)
        baseline = details['baseline']
        missing = details['pattern'] > 0
        np.testing.assert_array_equal(prediction[~missing], baseline[~missing])
        original_csv = pd.read_csv(ROOT / 'platform_submission_v2/platform_dl13_prevfix.csv')
        np.testing.assert_array_equal(test.id.to_numpy(), original_csv.id.to_numpy())
        np.testing.assert_array_equal(np.round(baseline, 6), original_csv.prediction.to_numpy())
        outputs = {
            'candidate_v4_uncertainty.csv': prediction,
            'candidate_v4_conservative.csv': baseline + .25 * details['full_delta'],
            'candidate_v2_fallback.csv': baseline,
        }
        for filename, values in outputs.items():
            assert len(values) == len(test) == 3000 and np.isfinite(values).all()
            pd.DataFrame({'id': test.id, 'prediction': np.round(values, 6)}).to_csv(HERE / filename, index=False)
        log('Check sampling behavior and convergence; no energy-score tuning')
        diagnostics = sampling_diagnostics(bundle, inputs)
        convergence = convergence_diagnostics(bundle, test)
        pd.DataFrame(diagnostics).to_csv(HERE / 'sampling_diagnostics.csv', index=False)
        pd.DataFrame(convergence).to_csv(HERE / 'sampling_convergence.csv', index=False)
    assert trained_state_before == joblib.hash(v2), 'An existing model was modified'
    assert hashes == {key: digest(path) for key, path in paths.items()}, 'An original file changed'
    joblib.dump(bundle, HERE / 'model.pkl', compress=3)
    write_notebook()
    pattern_records = []
    for pattern in np.unique(details['pattern']):
        delta = details['full_delta'][details['pattern'] == pattern]
        pattern_records.append({'pattern': int(pattern), 'rows': len(delta),
            'mean_prediction_change': float(delta.mean()),
            'rms_prediction_change': float(np.sqrt(np.mean(delta ** 2))),
            'max_abs_prediction_change': float(np.max(np.abs(delta)))})
    pd.DataFrame(pattern_records).to_csv(HERE / 'prediction_changes_by_pattern.csv', index=False)
    summary = {
        'status': 'implemented; experimental inference; no new OOF energy RMSE',
        'models_retrained': 0, 'trained_state_hash_before': trained_state_before,
        'trained_state_hash_after': joblib.hash(v2), 'original_files_unchanged': True,
        'source_hashes': hashes, 'sampling_config': config, 'versions': bundle['versions'],
        'rows': len(test), 'complete_rows': int((~missing).sum()), 'missing_rows': int(missing.sum()),
        'imputed_energy_branch_final_weight': imputed_branch_weight(settings),
        'exact_v2_csv_max_difference': float(np.max(np.abs(np.round(baseline, 6) - original_csv.prediction))),
        'complete_row_max_change': float(np.max(np.abs(prediction[~missing] - baseline[~missing]))),
        'missing_row_rms_change': float(np.sqrt(np.mean(details['full_delta'][missing] ** 2))),
        'missing_row_max_change': float(np.max(np.abs(details['full_delta'][missing]))),
        'sampling_diagnostics': diagnostics, 'sampling_convergence': convergence,
        'energy_labels_used_for_sampling_or_selection': False,
        'limitations': [
            'Frozen energy models were trained on all original training labels; no honest new OOF comparison is possible without saved fold models or refitting.',
            'Uncertainty uses in-sample auxiliary residuals and a symmetric donor approximation; it is not a calibrated posterior and can underestimate or misrepresent uncertainty.',
            'Only the LightGBM-imputed energy branch is marginalized. Neural inputs/predictions, direct experts, MissForest branch, weights and original missing-previous +0.8 remain unchanged.',
            'The 25% configuration is a fixed optional shrinkage, not an empirically selected improvement. The main 100% averaging configuration is experimental.',
            'Prediction consistency and finite outputs do not establish a lower leaderboard RMSE.',
        ],
        'wall_seconds': time.time() - start,
    }
    (HERE / 'v4_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    log(json.dumps({key: summary[key] for key in ['models_retrained', 'rows', 'complete_rows',
        'missing_rows', 'missing_row_rms_change', 'missing_row_max_change', 'wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
