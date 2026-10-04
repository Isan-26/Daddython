"""Training-only, nested-CV upgrade of the frozen v2 ensemble.

Run with the platform Python environment. FULL mode is the default; --fast uses
three outer/inner folds and three imputer seeds. No neural network is retrained.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import legacy_training as lt
from upgrade_runtime import *

FAST_MODE = False
RANDOM_STATE = 2026
N_FOLDS = 5
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAMILIES = ['core', 'uncertainty', 'weekday', 'weather', 'deviations', 'physical', 'combined']
MIN_GAIN = 0.004
CPU_THREADS = 4
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def data_digest(rows):
    return hashlib.sha256(pd.util.hash_pandas_object(rows, index=True).values.tobytes()).hexdigest()


def log(message):
    print(message, flush=True)


def cached(name, rows, config, build):
    key = hashlib.sha256((data_digest(rows) + json.dumps(config, sort_keys=True)).encode()).hexdigest()[:20]
    path = HERE / 'cache_upgrade' / f'{name}_{key}.pkl'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        log(f'  cache: {name}')
        return joblib.load(path)
    value = build()
    joblib.dump(value, path, compress=3)
    return value


def fit_hierarchy(rows):
    rows = hierarchy_rows(rows)
    result = {}
    for c in NUMERIC_COLUMNS:
        keys = ([['hour'], ['month'], ['month', 'hour']] if c in ['temperature', 'humidity']
                else [['building_type', 'hour'], ['building_id'], ['building_id', 'hour'],
                      ['building_id', 'hour', 'is_weekend'], ['building_id', 'hour', 'day_of_week']])
        result[c] = {'global': float(rows[c].median()), 'global_std': float(rows[c].std()),
                     'epsilon': 0.2 if c in ['temperature', 'humidity'] else 1.,
                     'smoothing': 20., 'levels': []}
        for key in keys:
            table = rows.groupby(key, observed=True)[c].agg(['median', 'count', 'std'])
            result[c]['levels'].append((key, table))
    return result


def sample_training_masks(rows, n, seed):
    rng = np.random.default_rng(seed)
    natural = rows[NUMERIC_COLUMNS].isna().to_numpy()
    masks = natural[rng.integers(0, len(rows), n)].copy()
    generic = rng.random(n) < 0.5
    pattern = rng.integers(1, 16, generic.sum())
    masks[generic] = (pattern[:, None] & (1 << np.arange(4))) > 0
    return masks


def fit_imputers(rows, config):
    # No energy target, test rows, or IDs enter the imputers.
    rows = rows.drop(columns=['energy_usage', 'id'], errors='ignore').reset_index(drop=True)
    parts = {'lightgbm': {}, 'extra': {}, 'extra_fill': {}, 'hierarchy': fit_hierarchy(rows)}
    for j, c in enumerate(NUMERIC_COLUMNS):
        observed = rows.loc[rows[c].notna()].copy()
        practice = observed.copy()
        mask = sample_training_masks(rows, len(practice), RANDOM_STATE + j)
        mask[:, j] = False
        practice[NUMERIC_COLUMNS] = practice[NUMERIC_COLUMNS].mask(mask)
        data = pd.concat([observed, practice], ignore_index=True)
        x, y = auxiliary_inputs(data, c), data[c].to_numpy()
        assert c not in x and 'energy_usage' not in x and 'id' not in x
        parts['lightgbm'][c] = []
        for seed in config['seeds']:
            m = lightgbm.LGBMRegressor(n_estimators=config['estimators'], learning_rate=0.045,
                max_depth=5, num_leaves=24, min_child_samples=30, reg_lambda=15.,
                subsample=.85, subsample_freq=1, colsample_bytree=.85, cat_smooth=20.,
                random_state=seed, n_jobs=CPU_THREADS, verbose=-1)
            parts['lightgbm'][c].append(m.fit(x, y))
        ex, fill = extratrees_inputs(data, c)
        parts['extra_fill'][c] = fill
        parts['extra'][c] = ExtraTreesRegressor(n_estimators=config['extra_trees'],
            min_samples_leaf=8, max_features=.8, n_jobs=CPU_THREADS,
            random_state=RANDOM_STATE + j).fit(ex, y)
    return parts


def evaluation_views(rows):
    """Natural rows plus all 15 artificial patterns on complete rows only."""
    rows = rows.reset_index(drop=True)
    complete = np.flatnonzero(rows[NUMERIC_COLUMNS].notna().all(axis=1))
    frames = [rows.drop(columns='energy_usage', errors='ignore')]
    origins, scenarios = [np.arange(len(rows))], [np.full(len(rows), -1)]
    for p in range(1, 16):
        q = frames[0].iloc[complete].copy()
        for j, c in enumerate(NUMERIC_COLUMNS):
            if p & (1 << j):
                q[c] = np.nan
        frames.append(q)
        origins.append(complete)
        scenarios.append(np.full(len(complete), p))
    frame = pd.concat(frames, ignore_index=True)
    origin, scenario = np.concatenate(origins), np.concatenate(scenarios)
    y = rows.energy_usage.to_numpy()[origin]
    weights = np.where(scenario == -1, .5 / len(rows), .5 / (15 * max(1, len(complete))))
    return frame, origin, scenario, y, weights


def crossfit_imputers(rows, config):
    views, origin, scenario, y, weights = evaluation_views(rows)
    predictions = {k: np.empty((len(views), 4)) for k in
                   ['lightgbm', 'std', 'extra', 'median', 'second', 'second_std', 'second_extra']}
    folds = np.empty(len(rows), int)
    coverage = np.zeros(len(views), int)
    for fold, (a, b) in enumerate(KFold(config['folds'], shuffle=True, random_state=RANDOM_STATE).split(rows)):
        assert not np.intersect1d(a, b).size
        folds[b] = fold
        positions = np.flatnonzero(np.isin(origin, b))
        log(f'  cross-fit imputers {fold + 1}/{config["folds"]}: {len(a):,} train rows')
        parts = fit_imputers(rows.iloc[a], config)
        out = predict_imputers(parts, views.iloc[positions])
        for k in predictions:
            predictions[k][positions] = out[k]
        coverage[positions] += 1
    assert np.all(coverage == 1), 'An OOF row was skipped or predicted twice'
    assert all(np.isfinite(p).all() for p in predictions.values())
    return {'views': views, 'origin': origin, 'scenario': scenario, 'truth': y,
            'weights': weights, 'predictions': predictions, 'folds': folds}


def choose_imputation_weights(data, rows, fit_origins):
    predictions, origin, scenario = data['predictions'], data['origin'], data['scenario']
    truth = rows[NUMERIC_COLUMNS].to_numpy(float)[origin]
    grid = np.array([[a / 10, b / 10, (10 - a - b) / 10]
                     for a in range(10, -1, -1) for b in range(10 - a, -1, -1)])
    result = []
    for j, c in enumerate(NUMERIC_COLUMNS):
        use = np.isin(origin, fit_origins) & np.isfinite(truth[:, j])
        use &= ((scenario == -1) | ((np.maximum(scenario, 0) & (1 << j)) > 0))
        p = np.column_stack([predictions[k][use, j] for k in ['lightgbm', 'extra', 'median']])
        errors = p @ grid.T - truth[use, j, None]
        score = np.average(errors ** 2, axis=0, weights=data['weights'][use])
        best = grid[np.argmin(score)]
        # A negligible reconstruction gain does not justify another estimator.
        if np.sqrt(score[0]) - np.sqrt(score.min()) < MIN_GAIN:
            best = grid[0]
        result.append(best)
    return np.array(result)


def crossfit_blend(data, rows):
    weights = np.empty((len(data['views']), 4, 3))
    for f in range(int(data['folds'].max()) + 1):
        fit = np.flatnonzero(data['folds'] != f)
        check = np.flatnonzero(data['folds'][data['origin']] == f)
        # No reconstruction target from the held-out origins selects its weights.
        assert not np.intersect1d(fit, np.unique(data['origin'][check])).size
        weights[check] = choose_imputation_weights(data, rows, fit)
    return (blend_imputations(data['predictions'], weights),
            blend_imputations(data['predictions'], weights, second=True))


def training_positions(data, raw_rows, fit_origins):
    origin, scenario = data['origin'], data['scenario']
    positions = np.flatnonzero((scenario == -1) & np.isin(origin, fit_origins)).tolist()
    masks = sample_training_masks(raw_rows.iloc[fit_origins], len(fit_origins), RANDOM_STATE + 913)
    codes = masks @ (1 << np.arange(4))
    lookup = {(int(o), int(p)): i for i, (o, p) in enumerate(zip(origin, scenario)) if p > 0}
    for o, p in zip(fit_origins, codes):
        if (int(o), int(p)) in lookup:
            positions.append(lookup[(int(o), int(p))])
    positions = np.array(positions)
    sw = np.where(scenario[positions] == -1, 1., .25)
    return positions, sw


def slice_aux(aux, positions):
    return {k: v[positions] for k, v in aux.items()}


def fit_ridge(features, target, sample_weight):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    try:
        model.fit(features, target, ridgecv__sample_weight=sample_weight)
    except np.linalg.LinAlgError:
        # LAPACK gesvd is a slower, stable fallback for a failed gesdd SVD.
        # The objective, alpha grid, data, and fitted estimator type are unchanged.
        log('  retrying Ridge SVD with LAPACK gesvd')
        def stable_svd(*args, **kwargs):
            kwargs['lapack_driver'] = 'gesvd'
            return SCIPY_SVD(*args, **kwargs)
        with patch('sklearn.linear_model._ridge.linalg.svd', stable_svd):
            model.fit(features, target, ridgecv__sample_weight=sample_weight)
    return model


def fit_energy(rows, auxiliary, original_train, families, sample_weight=None):
    rows = rows.reset_index(drop=True)
    filled = complete_rows(rows, auxiliary)
    parts = {'typical': lt.typical_values(filled), 'hierarchy': fit_hierarchy(original_train),
             'ridge': {}, 'family': families[0]}
    parts['boosting'] = HistGradientBoostingRegressor(categorical_features='from_dtype',
        max_iter=600, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=8., early_stopping=False, random_state=RANDOM_STATE).fit(
            energy_tree_inputs(rows, auxiliary), rows.energy_usage.to_numpy() - filled.previous_usage,
            sample_weight=sample_weight)
    for family in families:
        features = enhanced_features(rows, auxiliary, parts, family)
        parts['ridge'][family] = fit_ridge(features, rows.energy_usage, sample_weight)
    return parts


def fit_direct(rows):
    # No public-test weights or masks in the new pattern expert.
    return lt.MissingPatternModels(test_mix_power=0.).fit(rows, rows.drop(columns='energy_usage')).models


def scores(y, prediction, weight=None):
    error = prediction - y
    return {'rmse': float(np.sqrt(np.average(error ** 2, weights=weight))),
            'mae': float(mean_absolute_error(y, prediction, sample_weight=weight)),
            'r2': float(r2_score(y, prediction, sample_weight=weight))}


def family_score(data, pred):
    return scores(data['truth'], pred, data['weights'])['rmse']


def fold_score(data, pred):
    return np.array([family_score({**data, 'truth': data['truth'][m], 'weights': data['weights'][m]}, pred[m])
                     for f in range(int(data['folds'].max()) + 1)
                     for m in [data['folds'][data['origin']] == f]])


def consistently_better(data, old, new, threshold=MIN_GAIN):
    before, after = fold_score(data, old), fold_score(data, new)
    natural = data['scenario'] == -1
    if family_score(data, old) - family_score(data, new) < threshold:
        return False
    if np.sum(after < before) < len(before) // 2 + 1:
        return False
    if scores(data['truth'][natural], new[natural])['rmse'] > scores(data['truth'][natural], old[natural])['rmse'] + .005:
        return False
    if after.std(ddof=1) > before.std(ddof=1) * 1.25 + .01:
        return False
    patterns = missing_patterns(data['views'])
    for p in range(16):
        m = patterns == p
        if m.sum() >= 50 and scores(data['truth'][m], new[m])['rmse'] > scores(data['truth'][m], old[m])['rmse'] + .15:
            return False
    return True


def correction_fit(rows, y, pred, weights, mode):
    m = rows.previous_usage.isna().to_numpy()
    global_bias = float(np.average((y - pred)[m], weights=weights[m])) if m.any() else 0.
    result = {'mode': mode, 'global': global_bias if mode != 'none' else 0., 'by_type': {}}
    if mode == 'context':
        for t in BUILDING_TYPES:
            q = m & (rows.building_type.to_numpy() == t)
            if q.sum() >= 100:
                b = float(np.average((y - pred)[q], weights=weights[q]))
                result['by_type'][t] = (q.sum() * b + 100 * global_bias) / (q.sum() + 100)
    return result


def select_upgrade(rows, data, config):
    one, two = crossfit_blend(data, rows)
    pred = {f: np.empty(len(data['views'])) for f in FAMILIES}
    second_pred = {f: np.empty(len(data['views'])) for f in FAMILIES}
    direct = np.empty(len(data['views']))
    origin = data['origin']
    for fold in range(config['folds']):
        a, b = np.flatnonzero(data['folds'] != fold), np.flatnonzero(data['folds'] == fold)
        positions, sw = training_positions(data, rows, a)
        check = np.flatnonzero(np.isin(origin, b))
        assert not np.intersect1d(origin[positions], origin[check]).size
        log(f'  cheap energy ablations {fold + 1}/{config["folds"]}')
        training = data['views'].iloc[positions].copy()
        training['energy_usage'] = data['truth'][positions]
        parts = fit_energy(training, slice_aux(one, positions), rows.iloc[a], FAMILIES, sw)
        check_rows = data['views'].iloc[check].reset_index(drop=True)
        for f in FAMILIES:
            pred[f][check] = predict_energy(parts, check_rows, slice_aux(one, check), f)
            second_pred[f][check] = predict_energy(parts, check_rows, slice_aux(two, check), f)
        direct[check] = predict_patterns(fit_direct(rows.iloc[a]), check_rows)
    family = 'core'
    comparisons = []
    for f in FAMILIES:
        p = .2 * direct + .8 * pred[f]
        base = .2 * direct + .8 * pred['core']
        keep = f == 'core' or consistently_better(data, base, p)
        comparisons.append({'family': f, 'balanced_rmse': family_score(data, p),
                            'natural_rmse': scores(data['truth'][:len(rows)], p[:len(rows)])['rmse'],
                            'fold_std': float(fold_score(data, p).std(ddof=1)), 'eligible': keep})
        if keep and family_score(data, p) < family_score(data, .2 * direct + .8 * pred[family]):
            family = f
    patterns = missing_patterns(data['views'])
    strategy = []
    mixed = np.empty(len(patterns))
    for p in range(16):
        m = patterns == p
        best = {'direct_weight': .2, 'two_pass': False}
        base = .2 * direct[m] + .8 * pred[family][m]
        best_error = np.mean((base - data['truth'][m]) ** 2)
        for second in [False, True] if int(p).bit_count() >= 2 else [False]:
            energy = second_pred[family] if second else pred[family]
            for w in np.arange(0, 1.01, .1):
                candidate = w * direct[m] + (1 - w) * energy[m]
                error = np.mean((candidate - data['truth'][m]) ** 2)
                gains = []
                for f in range(config['folds']):
                    local = data['folds'][origin[m]] == f
                    gains.append(np.mean((base[local] - data['truth'][m][local]) ** 2)
                                 - np.mean((candidate[local] - data['truth'][m][local]) ** 2))
                if error < best_error and np.sqrt(np.mean((base - data['truth'][m]) ** 2)) - np.sqrt(error) >= MIN_GAIN and np.sum(np.array(gains) > 0) >= config['folds'] // 2 + 1:
                    best_error = error
                    best = {'direct_weight': round(float(w), 1), 'two_pass': second}
        strategy.append(best)
        energy = second_pred[family] if best['two_pass'] else pred[family]
        mixed[m] = best['direct_weight'] * direct[m] + (1 - best['direct_weight']) * energy[m]
    mode = 'none'
    corrected = mixed.copy()
    for proposed in ['global', 'context']:
        candidate = mixed.copy()
        for fold in range(config['folds']):
            fit = data['folds'][origin] != fold
            check = ~fit
            correction = correction_fit(data['views'].loc[fit], data['truth'][fit], mixed[fit], data['weights'][fit], proposed)
            candidate[check] += bias_values(data['views'].loc[check], correction)
        if consistently_better(data, corrected, candidate):
            mode, corrected = proposed, candidate
    correction = correction_fit(data['views'], data['truth'], mixed, data['weights'], mode)
    return {'family': family, 'comparisons': comparisons, 'strategy': strategy,
            'correction': correction, 'imputation_weights': choose_imputation_weights(data, rows, np.arange(len(rows))),
            'direct_oof': direct, 'imputed_oof': pred[family], 'two_pass_oof': second_pred[family],
            'selected_oof': corrected}


def train_selected(rows, data, selection, config):
    one, _ = crossfit_blend(data, rows)
    positions, sw = training_positions(data, rows, np.arange(len(rows)))
    training = data['views'].iloc[positions].copy()
    training['energy_usage'] = data['truth'][positions]
    log('  final cheap energy and pattern models')
    energy = fit_energy(training, slice_aux(one, positions), rows, [selection['family']], sw)
    log('  all-training multi-seed imputers')
    return {'energy': energy, 'pattern_models': fit_direct(rows), 'imputers': fit_imputers(rows, config),
            **{k: selection[k] for k in ['strategy', 'correction', 'imputation_weights']}}


def prediction_components(parts, rows):
    p = predict_imputers(parts['imputers'], rows)
    one, two = blend_imputations(p, parts['imputation_weights']), blend_imputations(p, parts['imputation_weights'], second=True)
    direct = predict_patterns(parts['pattern_models'], rows)
    p1, p2 = predict_energy(parts['energy'], rows, one), predict_energy(parts['energy'], rows, two)
    selected = apply_strategy(rows, direct, p1, p2, parts['strategy']) + bias_values(rows, parts['correction'])
    return selected, direct, p1, p2, p


def legacy_baseline(train, validation, test_inputs):
    log('  same-fold legacy baseline (original v2 recipe)')
    v3 = lt.V3NotebookModel().fit(train)
    patterns = lt.MissingPatternModels().fit(train, test_inputs)
    lgb = lt.LightGBMImputedModel(seed=527).fit(train, test_inputs)
    v3p = v3.predict(validation)
    non_dl = .0625 * v3p + .1875 * patterns.predict(validation) + .75 * lgb.predict(validation)
    return non_dl, v3p


def metric_rows(rows, truth, predictions, fold, scenario):
    records = []
    pattern = missing_patterns(rows)
    for name, p in predictions.items():
        groups = [('overall', np.ones(len(rows), bool), None),
                  ('natural', scenario == -1, None)]
        for code in range(16):
            groups.append((str(code), pattern == code, None))
        for tag, m, weight in groups:
            if m.any():
                records.append({'experiment': name, 'fold': fold, 'missing_pattern': tag,
                                'row_count': int(m.sum()), **scores(truth[m], p[m], weight)})
    return records


def reconstruction_rows(raw_train, data, fold, weights):
    truth = raw_train[NUMERIC_COLUMNS].to_numpy(float)[data['origin']]
    output = []
    ensemble = blend_imputations(data['predictions'], weights)['mean']
    for j, c in enumerate(NUMERIC_COLUMNS):
        m = np.isfinite(truth[:, j]) & ((np.maximum(data['scenario'], 0) & (1 << j)) > 0)
        for name, p in [('lightgbm', data['predictions']['lightgbm']), ('extratrees', data['predictions']['extra']),
                        ('hierarchical', data['predictions']['median']), ('ensemble', ensemble),
                        ('two_pass_lightgbm', data['predictions']['second'])]:
            s = scores(truth[m, j], p[m, j])
            output.append({'feature': c, 'imputer': name, 'fold': fold, **s})
    return output


def create_prediction_notebook():
    legacy = (HERE / 'legacy_runtime.py').read_text()
    runtime = (HERE / 'upgrade_runtime.py').read_text().replace('from legacy_runtime import *\n', '')
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Platform submission v3\n',
          '\nUpload this notebook and `feature_upgrade_model.pkl`. It needs no training data or internet.\n',
          'The pretrained neural-network input layouts and weights are frozen.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (legacy + '\n' + runtime).splitlines(True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': '''import os
import joblib
from threadpoolctl import threadpool_limits
TEST_PATH = os.environ.get("DATATHON_INPUT_PATH", "test.csv")
OUTPUT_PATH = os.environ.get("DATATHON_OUTPUT_PATH", "predictions.csv")
bundle = joblib.load("feature_upgrade_model.pkl")
rows = pd.read_csv(TEST_PATH).drop(columns=["id", "energy_usage"], errors="ignore")
with threadpool_limits(limits=1):
    prediction = np.round(predict_upgrade(bundle, rows), 6)
assert len(prediction) == len(rows) and np.isfinite(prediction).all()
pd.DataFrame({"prediction": prediction}).to_csv(OUTPUT_PATH, index=False)
print(f"wrote {len(rows):,} predictions to {OUTPUT_PATH}")
'''.splitlines(True)}]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'datathon-platform-env (Python 3.12)',
                'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.15'}},
                'nbformat': 4, 'nbformat_minor': 4}
    (HERE / 'prediction_notebook.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Check the legacy reproduction without training')
    args = parser.parse_args()
    start = time.monotonic()
    assert sklearn.__version__ == '1.5.2' and lightgbm.__version__ == '4.5.0', 'Use requirements-image.txt versions'
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    legacy_path = ROOT / 'platform_submission_v2' / 'model.pkl'
    legacy = joblib.load(legacy_path)
    config = {'fast': args.fast, 'folds': 3 if args.fast else N_FOLDS,
              'seeds': list(range(527, 530 if args.fast else 532)),
              'estimators': 300 if args.fast else 700, 'extra_trees': 40 if args.fast else 80,
              'random_state': RANDOM_STATE,
              'source_hash': hashlib.sha256(''.join((HERE / p).read_text() for p in
                   ['feature_imputation_upgrade.py', 'upgrade_runtime.py', 'legacy_runtime.py', 'legacy_training.py']).encode()).hexdigest(),
              'sklearn': sklearn.__version__, 'lightgbm': lightgbm.__version__}
    test_inputs = test.drop(columns=['id', 'energy_usage'], errors='ignore')
    hashes = {'train': digest(ROOT / 'train.csv'), 'test': digest(ROOT / 'test.csv'), 'legacy': digest(legacy_path)}
    config['train_hash'] = hashes['train']
    with threadpool_limits(limits=CPU_THREADS):
        reproduced = np.round(predict_version(legacy, test_inputs, legacy['versions']['dl13_prevfix']), 6)
        expected = pd.read_csv(ROOT / 'platform_submission_v2' / 'platform_dl13_prevfix.csv')
        assert expected.id.equals(test.id)
        gap = float(np.max(np.abs(reproduced - expected.prediction.to_numpy())))
        assert gap <= 1e-6, f'Legacy reproduction differs by {gap}'
        log(f'Legacy reproduced: maximum prediction difference {gap:.2e}; NN inputs 40, RealMLP numeric inputs 15')
        if args.check_only:
            return
        fold_results, imputation_results, all_fold_data, selections = [], [], [], []
        for fold, (a, b) in enumerate(KFold(config['folds'], shuffle=True, random_state=RANDOM_STATE).split(train)):
            log(f'OUTER FOLD {fold + 1}/{config["folds"]}')
            fit = train.iloc[a].reset_index(drop=True)
            check = train.iloc[b].reset_index(drop=True)
            views, origin, scenario, truth, weights = evaluation_views(check)
            fold_config = {**config, 'legacy_hash': hashes['legacy'], 'test_hash_legacy_only': hashes['test']}
            baseline, v3p = cached(f'baseline_fold{fold}', check, fold_config,
                                  lambda: legacy_baseline(fit, views, test_inputs))
            data = cached(f'crossfit_fold{fold}', fit, config, lambda: crossfit_imputers(fit, config))
            selection = cached(f'selection_fold{fold}', fit, config, lambda: select_upgrade(fit, data, config))
            selections.append(selection)
            def build_outer_predictions():
                parts = train_selected(fit, data, selection, config)
                return prediction_components(parts, views)
            selected, direct, p1, p2, reconstructed = cached(f'outer_predictions_fold{fold}', fit,
                {**config, 'validation_hash': data_digest(check)}, build_outer_predictions)
            upgraded = .0625 * v3p + .9375 * selected
            compare = {'legacy_non_dl': baseline, 'upgrade_non_dl': upgraded,
                       'direct': direct, 'one_pass': p1, 'two_pass': p2}
            records = metric_rows(views, truth, compare, fold, scenario)
            for name, p in compare.items():
                records.append({'experiment': name, 'fold': fold, 'missing_pattern': 'balanced',
                                'row_count': len(views), **scores(truth, p, weights)})
            fold_results.extend(records)
            cv_data = {'views': views, 'origin': origin, 'scenario': scenario, 'truth': truth,
                       'weights': weights, 'predictions': reconstructed, 'folds': np.zeros(len(check), int)}
            imputation_results.extend(reconstruction_rows(check, cv_data, fold, selection['imputation_weights']))
            all_fold_data.append({'rows': views, 'truth': truth, 'scenario': scenario, 'weights': weights,
                                  'baseline': baseline, 'upgraded': upgraded, 'direct': direct, 'p1': p1, 'p2': p2})
            pd.DataFrame(fold_results).to_csv(HERE / 'feature_upgrade_cv_results.csv', index=False)
            pd.DataFrame(imputation_results).to_csv(HERE / 'imputation_cv_results.csv', index=False)
            natural = scenario == -1
            log(f'  natural RMSE baseline {scores(truth[natural], baseline[natural])["rmse"]:.5f} -> {scores(truth[natural], upgraded[natural])["rmse"]:.5f}; '
                f'balanced {scores(truth, baseline, weights)["rmse"]:.5f} -> {scores(truth, upgraded, weights)["rmse"]:.5f}; selected {selection["family"]}')
        finish(train, test, legacy, all_fold_data, selections, fold_results,
               imputation_results, config, hashes, gap, start)


def finish(train, test, legacy, folds, selections, fold_results, imputation_results, config, hashes, gap, start):
    metrics = pd.DataFrame(fold_results)
    balanced = metrics[metrics.missing_pattern == 'balanced'].pivot(index='fold', columns='experiment', values='rmse')
    natural = metrics[metrics.missing_pattern == 'natural'].pivot(index='fold', columns='experiment', values='rmse')
    gains = balanced.legacy_non_dl - balanced.upgrade_non_dl
    natural_gain = natural.legacy_non_dl.mean() - natural.upgrade_non_dl.mean()
    accepted = (gains.mean() >= MIN_GAIN and natural_gain >= -.005 and
                (gains > 0).sum() >= config['folds'] // 2 + 1 and
                balanced.upgrade_non_dl.std() <= balanced.legacy_non_dl.std() * 1.25 + .01)
    pattern_rows = []
    all_rows = pd.concat([f['rows'] for f in folds], ignore_index=True)
    pattern = missing_patterns(all_rows)
    truth = np.concatenate([f['truth'] for f in folds])
    for p in range(16):
        m = pattern == p
        values = {k: scores(truth[m], np.concatenate([f[k] for f in folds])[m])['rmse']
                  for k in ['baseline', 'upgraded', 'direct', 'p1', 'p2']}
        if values['upgraded'] > values['baseline'] + .15:
            accepted = False
        pattern_rows.append({'pattern': p, 'row_count': int(m.sum()),
            'baseline_rmse': values['baseline'], 'upgraded_rmse': values['upgraded'],
            'direct_rmse': values['direct'], 'imputed_rmse': values['p1'], 'two_pass_rmse': values['p2'],
            'chosen_strategy': 'nested training-only; final choice below', 'chosen_weights': ''})
    log(f'Nested CV acceptance: {accepted}; mean balanced gain {gains.mean():.5f}; natural gain {natural_gain:.5f}')
    full_data = cached('crossfit_all_train', train, config, lambda: crossfit_imputers(train, config))
    full_selection = cached('selection_all_train', train, config, lambda: select_upgrade(train, full_data, config))
    bundle = {'legacy': legacy, 'accepted': bool(accepted), 'config': config, 'data_hashes': hashes}
    if accepted:
        bundle['upgrade'] = train_selected(train, full_data, full_selection, config)
    # Fallback corrections use ONLY naturally missing training rows and OOF residuals.
    # They are screened by leave-fold-out application, never by a leaderboard score.
    raw_rows = pd.concat([f['rows'].loc[f['scenario'] == -1] for f in folds], ignore_index=True)
    raw_truth = np.concatenate([f['truth'][f['scenario'] == -1] for f in folds])
    raw_base = np.concatenate([f['baseline'][f['scenario'] == -1] for f in folds])
    corrected = raw_base.copy()
    offset = 0
    for f in folds:
        n = int((f['scenario'] == -1).sum())
        check = np.arange(offset, offset + n)
        fit = np.setdiff1d(np.arange(len(raw_base)), check)
        c = correction_fit(raw_rows.iloc[fit], raw_truth[fit], raw_base[fit], np.ones(len(fit)), 'global')
        corrected[check] += bias_values(raw_rows.iloc[check], c)
        offset += n
    fallback_gain = scores(raw_truth, raw_base)['rmse'] - scores(raw_truth, corrected)['rmse']
    bundle['fallback_correction'] = correction_fit(raw_rows, raw_truth, raw_base, np.ones(len(raw_base)),
                                                   'global' if fallback_gain >= .003 else 'none')
    # This fallback calibration is a secondary OOF diagnostic, not a nested-CV claim.
    fixed = cached('fixed_dl_predictions', test,
        {**config, 'legacy_hash': hashes['legacy']}, lambda: {
            'sklearn_mlp': predict_neural_net(legacy, test), 'realmlp': predict_realmlp(legacy, test)})
    prediction = np.round(predict_upgrade(bundle, test), 6)
    # Verify fixed experts are unchanged even after fitting the new branch.
    assert np.array_equal(fixed['sklearn_mlp'], predict_neural_net(legacy, test))
    assert np.array_equal(fixed['realmlp'], predict_realmlp(legacy, test))
    candidate = pd.DataFrame({'id': test.id, 'prediction': prediction})
    assert list(candidate.columns) == ['id', 'prediction'] and len(candidate) == len(test) == 3000
    assert candidate.id.equals(test.id) and not candidate.id.duplicated().any()
    assert np.isfinite(prediction).all()
    candidate.to_csv(HERE / 'candidate_feature_upgrade.csv', index=False)
    joblib.dump(bundle, HERE / 'feature_upgrade_model.pkl', compress=3)
    create_prediction_notebook()
    for row in pattern_rows:
        strategy = full_selection['strategy'][row['pattern']]
        row['chosen_strategy'] = ('two_pass' if strategy['two_pass'] else 'one_pass') if accepted else 'legacy_fallback'
        row['chosen_weights'] = json.dumps({'direct': strategy['direct_weight'], 'imputed': round(1 - strategy['direct_weight'], 1)}) if accepted else 'legacy weights preserved'
    pd.DataFrame(pattern_rows).to_csv(HERE / 'missing_pattern_results.csv', index=False)
    comparison = pd.DataFrame([{'outer_fold': f, **r} for f, s in enumerate(selections)
                               for r in s['comparisons']] + [{'outer_fold': 'final_training', **r}
                                                           for r in full_selection['comparisons']])
    comparison.to_csv(HERE / 'feature_family_results.csv', index=False)
    summary = {'accepted_upgrade': bool(accepted), 'mode': 'FAST' if config['fast'] else 'FULL',
        'legacy_reproduction_max_difference': gap, 'random_state': RANDOM_STATE,
        'evaluation': 'nested outer CV; primary balanced = 50% natural rows + 50% uniform artificial patterns 1..15; no test-derived selection',
        'baseline_natural_rmse': scores(raw_truth, raw_base)['rmse'],
        'upgrade_natural_rmse': scores(raw_truth, np.concatenate([f['upgraded'][f['scenario'] == -1] for f in folds]))['rmse'],
        'baseline_mean_balanced_rmse': float(balanced.legacy_non_dl.mean()),
        'upgrade_mean_balanced_rmse': float(balanced.upgrade_non_dl.mean()),
        'baseline_fold_std': float(balanced.legacy_non_dl.std()), 'upgrade_fold_std': float(balanced.upgrade_non_dl.std()),
        'folds': balanced[['legacy_non_dl', 'upgrade_non_dl']].reset_index().to_dict('records'),
        'final_feature_family': full_selection['family'],
        'imputation_weights': {c: w.tolist() for c, w in zip(NUMERIC_COLUMNS, full_selection['imputation_weights'])},
        'imputer_weight_order': ['LightGBM', 'ExtraTrees', 'hierarchical'],
        'two_pass_patterns': [p for p, s in enumerate(full_selection['strategy']) if s['two_pass']] if accepted else [],
        'learned_previous_missing_correction': full_selection['correction'] if accepted else bundle['fallback_correction'],
        'fixed_dl_weights': {'sklearn_mlp': .0435, 'RealMLP': .13, 'non_dl': .8265},
        'dl_oof_limitation': 'Existing DL OOF uses different folds (seed 0), so no full-ensemble OOF score or DL weight tuning is claimed.',
        'fallback_calibration_diagnostic_gain': fallback_gain,
        'fallback_calibration_limitation': 'Leave-fold-out calibration of existing base OOF predictions is secondary, not fully nested.',
        'runtime_seconds': time.monotonic() - start, 'data_hashes': hashes, 'config': config}
    (HERE / 'feature_upgrade_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ('Platform submission v3: training-only upgrade\n\n' + json.dumps(summary, indent=2)
              + '\n\nMissing-pattern comparison\n' + pd.DataFrame(pattern_rows).to_string(index=False)
              + '\n\nOuter validation reconstruction RMSE\n' + pd.DataFrame(imputation_results).groupby(['feature', 'imputer']).rmse.mean().to_string()
              + '\n\nOOF scores cover only the non-DL branch. No leaderboard scores were used.\n')
    (HERE / 'feature_upgrade_summary.txt').write_text(report)
    log(candidate.head(10).to_string(index=False))
    log(f'Prediction min={prediction.min():.6f}, max={prediction.max():.6f}, mean={prediction.mean():.6f}, std={prediction.std():.6f}')
    log(f'Finished in {(time.monotonic() - start) / 60:.1f} minutes; upload prediction_notebook.ipynb + feature_upgrade_model.pkl')


if __name__ == '__main__':
    main()
