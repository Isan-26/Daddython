"""Isolated weather correction; v2 is frozen and zero correction is the default.

No neural network is fitted. Only legacy non-DL validation models and a tiny
Ridge correction are fitted. Saved neural OOF predictions use seed-0 folds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings
from unittest.mock import patch

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy.linalg import svd as SCIPY_SVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from weather_runtime import *

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CPU_THREADS = 4
RIDGE_ALPHA = 100.
SHRINKAGE_GRID = [0., .1, .2, .5, 1.]
MIN_COMPLETE_GAIN = .005
MIN_OVERALL_GAIN = .004
MAX_FOLD_REGRESSION = .005
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def log(message):
    print(message, flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2)))


def original_fold_ids(rows):
    folds = np.full(len(rows), -1, int)
    for fold, (_, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(rows)):
        folds[b] = fold
    assert np.all(folds >= 0)
    return folds


def legacy_validation_inputs(check, public_inputs, fold):
    # The old NN OOF code used these virtual test inputs. This is retained only
    # to reproduce legacy validation; the correction has no test-based weights.
    rng = np.random.default_rng(fold)
    patterns = public_inputs[NUMERIC_COLUMNS].isna().to_numpy()
    mask = patterns[rng.integers(0, len(public_inputs), len(check))]
    result = check.drop(columns='energy_usage').copy()
    result[NUMERIC_COLUMNS] = result[NUMERIC_COLUMNS].mask(mask)
    return result


def non_dl_oof(train, test, hashes):
    import baseline_training as bt
    config = {'train': hashes['train'], 'test_legacy_only': hashes['test'],
              'v2': hashes['v2'], 'fold_seed': 0, 'folds': 5,
              'source': digest(HERE / 'baseline_training.py'),
              'builder': digest(Path(__file__)), 'sklearn': sklearn.__version__,
              'lightgbm': lightgbm.__version__}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    directory = HERE / 'cache'
    directory.mkdir(exist_ok=True)
    path = directory / f'v2_non_dl_oof_{key}.pkl'
    if path.exists():
        log('Reusing verified v2 non-DL OOF cache')
        data = joblib.load(path)
        np.testing.assert_array_equal(data['ids'], train.id.to_numpy())
        return data['prediction']
    prediction = np.empty(len(train))
    coverage = np.zeros(len(train), int)
    for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train)):
        assert not np.intersect1d(a, b).size
        log(f'Legacy non-DL validation fold {fold + 1}/5; no neural-network training')
        fit = train.iloc[a]
        check = train.iloc[b]
        virtual_test = legacy_validation_inputs(check, test, fold)
        def fit_predict():
            v3 = bt.V3NotebookModel().fit(fit)
            patterns = bt.MissingPatternModels().fit(fit, virtual_test)
            imputed = bt.LightGBMImputedModel(seed=527).fit(fit, virtual_test)
            return (.0625 * v3.predict(check) + .1875 * patterns.predict(check)
                    + .75 * imputed.predict(check))
        try:
            prediction[b] = fit_predict()
        except np.linalg.LinAlgError:
            log('Retrying the same legacy Ridge objective with LAPACK gesvd')
            def stable_svd(*args, **kwargs):
                return SCIPY_SVD(*args, **{**kwargs, 'lapack_driver': 'gesvd'})
            with patch('sklearn.linear_model._ridge.linalg.svd', stable_svd):
                prediction[b] = fit_predict()
        coverage[b] += 1
    assert np.all(coverage == 1) and np.isfinite(prediction).all()
    joblib.dump({'prediction': prediction, 'ids': train.id.to_numpy(), 'config': config}, path, compress=3)
    return prediction


def load_oof_experts(train):
    source = ROOT / 'experiments' / 'neural_net_cv.npz'
    with np.load(source, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['truth'], train.energy_usage.to_numpy())
        neural = saved['neural_net_as_is'].copy()
    seed_predictions, seeds, files = [], [], [source]
    for seed in range(10):
        suffix = '' if seed == 0 else f'_seed{seed}'
        path = ROOT / 'experiments' / f'dl_cv_realmlp{suffix}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                seed_predictions.append(saved['as_is'].copy())
            seeds.append(seed)
            files.append(path)
    assert seeds, 'No reusable RealMLP OOF predictions exist'
    assert all(a.shape == (len(train),) and np.isfinite(a).all()
               for a in [neural, *seed_predictions])
    return neural, np.mean(seed_predictions, axis=0), {
        'fold_seed': 0, 'folds': 5, 'sklearn_mlp_networks': 5,
        'RealMLP_oof_seeds': seeds, 'RealMLP_deployed_seeds': list(range(10)),
        'all_deployed_seeds_available': seeds == list(range(10)),
        'source_files': {str(path.relative_to(ROOT)): digest(path) for path in files},
        'notebook_sources': {name: digest(ROOT / name) for name in
                             ['neural_net_model.ipynb', 'deep_learning_model.ipynb']},
        'limitation': 'Existing base OOF caches have no original runtime/data-hash metadata. '
                      'Truth ordering is checked and notebook fold/recipe provenance is audited. '
                      'Correction CV is conditional on these fixed OOF predictions; it is not '
                      'an end-to-end nested refit of all baseline models.',
    }


def fit_correction(rows, residual):
    complete = complete_mask(rows)
    assert complete.any()
    x = weather_features(rows.loc[complete])
    target = np.asarray(residual)[complete]
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA, solver='lsqr'))
    model.fit(x, target)
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    state = {'center': scaler.mean_.copy(), 'scale': scaler.scale_.copy(),
             'coef': ridge.coef_.copy(), 'intercept': float(ridge.intercept_)}
    np.testing.assert_allclose(weather_delta(state, rows)[complete], model.predict(x), rtol=1e-12, atol=1e-12)
    return state


def correction_oof(train, baseline, folds, positions):
    positions = np.asarray(positions)
    result = np.zeros(len(train))
    coverage = np.zeros(len(train), int)
    for fold in np.unique(folds[positions]):
        a = positions[folds[positions] != fold]
        b = positions[folds[positions] == fold]
        assert not np.intersect1d(a, b).size
        state = fit_correction(train.iloc[a], train.energy_usage.to_numpy()[a] - baseline[a])
        result[b] = weather_delta(state, train.iloc[b])
        coverage[b] += 1
    assert np.all(coverage[positions] == 1)
    return result


def choose_alpha(train, baseline, delta, folds, positions):
    y = train.energy_usage.to_numpy()
    complete = complete_mask(train)
    c = positions[complete[positions]]
    unique_folds = np.unique(folds[positions])
    records = []
    chosen, best = 0., rmse(y[positions], baseline[positions])
    for alpha in SHRINKAGE_GRID:
        candidate = baseline + alpha * delta
        overall_gain = rmse(y[positions], baseline[positions]) - rmse(y[positions], candidate[positions])
        complete_gain = rmse(y[c], baseline[c]) - rmse(y[c], candidate[c])
        gains = [rmse(y[q], baseline[q]) - rmse(y[q], candidate[q]) for fold in unique_folds
                 for q in [positions[folds[positions] == fold]]]
        qualifies = (alpha == 0. or (overall_gain >= MIN_OVERALL_GAIN and
                     complete_gain >= MIN_COMPLETE_GAIN and
                     np.sum(np.array(gains) > 0) >= len(unique_folds) - 1 and
                     min(gains) >= -MAX_FOLD_REGRESSION))
        value = rmse(y[positions], candidate[positions])
        records.append({'alpha': alpha, 'natural_rmse': value, 'complete_rmse': rmse(y[c], candidate[c]),
                        'overall_gain': overall_gain, 'complete_gain': complete_gain,
                        'folds_improved': int(np.sum(np.array(gains) > 0)), 'qualifies': qualifies})
        if qualifies and value < best:
            chosen, best = alpha, value
    return chosen, records


def validate_correction(train, baseline, folds):
    positions = np.arange(len(train))
    raw_oof_delta = correction_oof(train, baseline, folds, positions)
    selected_prediction = baseline.copy()
    fold_records = []
    for fold in range(5):
        a, b = positions[folds != fold], positions[folds == fold]
        inner_delta = correction_oof(train, baseline, folds, a)
        alpha, _ = choose_alpha(train, baseline, inner_delta, folds, a)
        state = fit_correction(train.iloc[a], train.energy_usage.to_numpy()[a] - baseline[a])
        selected_prediction[b] += alpha * weather_delta(state, train.iloc[b])
        complete = complete_mask(train.iloc[b])
        for group, mask in [('natural', np.ones(len(b), bool)), ('complete', complete), ('missing', ~complete)]:
            q = b[mask]
            fold_records.append({'fold': fold, 'group': group, 'row_count': len(q),
                'alpha_selected_on_other_folds': alpha,
                'baseline_rmse': rmse(train.energy_usage.to_numpy()[q], baseline[q]),
                'selected_rmse': rmse(train.energy_usage.to_numpy()[q], selected_prediction[q])})
    final_alpha, trials = choose_alpha(train, baseline, raw_oof_delta, folds, positions)
    return selected_prediction, raw_oof_delta, final_alpha, trials, fold_records


def write_notebook():
    runtime = (HERE / 'weather_runtime.py').read_text().replace('from frozen_v2 import *\n', '')
    source = (HERE / 'frozen_v2.py').read_text() + '\n' + runtime
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Frozen v2 with an optional weather correction\n',
            'Upload this notebook and `model.pkl`. V2 weights and preprocessing are unchanged.\n',
            'The correction is zero unless validation passes, and never modifies incomplete rows.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': '''import os
import warnings
import joblib
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore')
INPUT_PATH = os.environ.get('DATATHON_INPUT_PATH', 'test.csv')
OUTPUT_PATH = os.environ.get('DATATHON_OUTPUT_PATH', 'predictions.csv')
bundle = joblib.load('model.pkl')
rows = pd.read_csv(INPUT_PATH).drop(columns=['id', 'energy_usage'], errors='ignore')
with threadpool_limits(limits=1):
    prediction = np.round(predict_with_weather(bundle, rows), 6)
assert len(prediction) == len(rows) and np.isfinite(prediction).all()
pd.DataFrame({'prediction': prediction}).to_csv(OUTPUT_PATH, index=False)
print(f'wrote {len(rows):,} predictions to {OUTPUT_PATH}')
'''.splitlines(True)}]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'datathon-platform-env',
                'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.15'}},
                'nbformat': 4, 'nbformat_minor': 4}
    (HERE / 'prediction_notebook.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Reproduce v2 without fitting anything')
    args = parser.parse_args()
    started = time.monotonic()
    assert sklearn.__version__ == '1.5.2' and lightgbm.__version__ == '4.5.0'
    train, test = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    hashes = {'train': digest(ROOT / 'train.csv'), 'test': digest(ROOT / 'test.csv'),
              'v2': digest(ROOT / 'platform_submission_v2' / 'model.pkl'),
              'v2_notebook': digest(ROOT / 'platform_submission_v2' / 'prediction_notebook.ipynb')}
    v2 = joblib.load(ROOT / 'platform_submission_v2' / 'model.pkl')
    with threadpool_limits(limits=CPU_THREADS):
        baseline_test = predict_version(v2, test, v2['versions']['dl13_prevfix'])
        original = pd.read_csv(ROOT / 'platform_submission_v2' / 'platform_dl13_prevfix.csv')
        assert original.id.equals(test.id)
        np.testing.assert_array_equal(np.round(baseline_test, 6), original.prediction.to_numpy())
        log('V2 reproduced exactly. All original experts, preprocessing, weights and +0.8 shift are frozen.')
        if args.check_only:
            return
        neural, real, provenance = load_oof_experts(train)
        log(f'Reusing saved neural OOF predictions: 5 sklearn networks, {len(provenance["RealMLP_oof_seeds"])} RealMLP seeds')
        baseline = (.8265 * non_dl_oof(train, test, hashes) + .0435 * neural + .13 * real
                    + .8 * train.previous_usage.isna().to_numpy())
        folds = original_fold_ids(train)
        selected, delta, alpha, trials, records = validate_correction(train, baseline, folds)
        y, complete = train.energy_usage.to_numpy(), complete_mask(train)
        gain = rmse(y, baseline) - rmse(y, selected)
        complete_gain = rmse(y[complete], baseline[complete]) - rmse(y[complete], selected[complete])
        natural_records = [r for r in records if r['group'] == 'natural']
        wins = sum(r['selected_rmse'] < r['baseline_rmse'] for r in natural_records)
        cv_passed = (alpha > 0 and gain >= MIN_OVERALL_GAIN and complete_gain >= MIN_COMPLETE_GAIN
                     and wins >= 4 and all(r['selected_rmse'] <= r['baseline_rmse'] + MAX_FOLD_REGRESSION
                                           for r in natural_records))
        # Do not enable an addition using fewer OOF DL seeds than deployment.
        enabled = bool(cv_passed and provenance['all_deployed_seeds_available'])
        state = fit_correction(train, y - baseline)
        bundle = {'v2': v2, 'correction': {'alpha': alpha if enabled else 0., 'ridge': state,
                  'features': WEATHER_FEATURES, 'enabled': enabled}, 'source_hashes': hashes,
                  'oof_provenance': provenance}
        prediction = np.round(predict_with_weather(bundle, test), 6)
        test_complete = complete_mask(test)
        np.testing.assert_array_equal(prediction[~test_complete], original.prediction.to_numpy()[~test_complete])
        if not enabled:
            np.testing.assert_array_equal(prediction, original.prediction.to_numpy())
        candidate = pd.DataFrame({'id': test.id, 'prediction': prediction})
        assert len(candidate) == len(test) == 3000 and not candidate.id.duplicated().any()
        assert list(candidate.columns) == ['id', 'prediction'] and candidate.id.equals(test.id)
        assert np.isfinite(prediction).all()
        assert digest(ROOT / 'platform_submission_v2' / 'model.pkl') == hashes['v2']
        assert digest(ROOT / 'platform_submission_v2' / 'prediction_notebook.ipynb') == hashes['v2_notebook']
        candidate.to_csv(HERE / 'candidate_v2_weather.csv', index=False)
        joblib.dump(bundle, HERE / 'model.pkl', compress=3)
        write_notebook()
    pd.DataFrame(trials).to_csv(HERE / 'weather_weight_trials.csv', index=False)
    pd.DataFrame(records).to_csv(HERE / 'weather_cv_results.csv', index=False)
    pd.DataFrame({'id': train.id, 'fold': folds, 'baseline_oof': baseline,
                  'weather_delta_oof': delta, 'selected_oof': selected}).to_csv(HERE / 'weather_oof_predictions.csv', index=False)
    summary = {'correction_enabled': enabled, 'exported_alpha': bundle['correction']['alpha'],
        'training_selected_alpha': alpha, 'cv_gate_passed': bool(cv_passed),
        'baseline_natural_oof_rmse': rmse(y, baseline), 'selected_natural_oof_rmse': rmse(y, selected),
        'baseline_complete_oof_rmse': rmse(y[complete], baseline[complete]),
        'selected_complete_oof_rmse': rmse(y[complete], selected[complete]),
        'fixed_weight_1_natural_rmse': rmse(y, baseline + delta),
        'fixed_weight_1_complete_rmse': rmse(y[complete], baseline[complete] + delta[complete]),
        'features': WEATHER_FEATURES, 'ridge_penalty': RIDGE_ALPHA, 'shrinkage_grid': SHRINKAGE_GRID,
        'gate': {'minimum_complete_gain': MIN_COMPLETE_GAIN, 'minimum_natural_gain': MIN_OVERALL_GAIN,
                 'minimum_folds_improved': 4, 'maximum_fold_regression': MAX_FOLD_REGRESSION},
        'oof_provenance': provenance, 'missing_predictions_unchanged': True,
        'candidate_matches_v2': bool(np.array_equal(prediction, original.prediction.to_numpy())),
        'v2_files_unchanged': True, 'deep_learning_models_retrained': 0,
        'runtime_seconds': time.monotonic() - started, 'source_hashes': hashes,
        'decision': ('Enabled after validation' if enabled else
                     'Zero correction: no qualifying gain and/or incomplete deployment-seed OOF coverage')}
    (HERE / 'weather_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    log(json.dumps({k: summary[k] for k in ['correction_enabled', 'exported_alpha',
        'baseline_natural_oof_rmse', 'selected_natural_oof_rmse', 'fixed_weight_1_natural_rmse',
        'candidate_matches_v2', 'runtime_seconds']}, indent=2))
    log('Finished. Upload prediction_notebook.ipynb + model.pkl only if a new candidate is warranted.')


if __name__ == '__main__':
    main()
