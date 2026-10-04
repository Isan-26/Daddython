"""Multiple imputations through the frozen v2 energy branch; no model fitting.

The sampler is a centered, symmetric empirical residual bootstrap, not a
calibrated posterior. Neural and missing-pattern predictions keep raw inputs.
"""
import numpy as np
import pandas as pd
from frozen_v2 import *


def calendar_inputs(rows):
    hour = rows.hour.to_numpy(float)
    month = rows.month.to_numpy(float)
    return np.column_stack([
        .5 * np.sin(2 * np.pi * hour / 24),
        .5 * np.cos(2 * np.pi * hour / 24),
        .35 * np.sin(2 * np.pi * month / 12),
        .35 * np.cos(2 * np.pi * month / 12),
        .75 * is_weekend(rows).to_numpy(float),
    ])


def missing_patterns(rows):
    return rows[NUMERIC_COLUMNS].isna().to_numpy().astype(int) @ (1 << np.arange(4))


def expected_and_filled(parts, rows):
    inputs = imputer_inputs(rows)
    expected = np.column_stack([
        parts['imputers'][column].predict(inputs.drop(columns=column))
        for column in NUMERIC_COLUMNS
    ])
    raw = rows[NUMERIC_COLUMNS].to_numpy(float)
    filled = np.where(np.isnan(raw), expected, raw)
    assert np.isfinite(filled).all() and np.isfinite(expected).all()
    return expected, filled


def donor_pools(state, rows, exclude_reference=None):
    """Find nearby feature donors using only originally observed inputs."""
    reference = state['reference']
    raw = rows[NUMERIC_COLUMNS].to_numpy(float)
    calendar = calendar_inputs(rows)
    pools = []
    for i in range(len(rows)):
        candidates = np.flatnonzero(reference['building'] == rows.building_id.iloc[i])
        if len(candidates) < state['neighbors']:
            candidates = np.arange(len(reference['numeric']))
        if exclude_reference is not None:
            candidates = candidates[candidates != exclude_reference[i]]
        assert len(candidates), 'No uncertainty donors are available'
        observed = np.isfinite(raw[i])
        distance = np.sum((reference['calendar'][candidates] - calendar[i]) ** 2, axis=1)
        if observed.any():
            differences = ((reference['numeric'][candidates][:, observed] - raw[i, observed])
                           / state['numeric_scale'][observed])
            distance += np.sum(differences ** 2, axis=1)
        count = min(state['neighbors'], len(candidates))
        # Stable tie ordering makes pools independent of input order or batches.
        order = np.lexsort((candidates, distance))[:count]
        pools.append(candidates[order])
    return pools


def sample_missing_inputs(state, rows, means, samples=64, seed=2026,
                          exclude_reference=None):
    """Joint residual donors and +/- pairs preserve each imputer's mean.

    Original observed values are copied exactly. Symmetric bounds restrict
    perturbations to training support without moving the frozen point estimate.
    IDs, labels, DataFrame indices and other rows never affect the random seed.
    """
    if samples < 2 or samples % 2:
        raise ValueError('samples must be an even integer >= 2')
    raw = rows[NUMERIC_COLUMNS].to_numpy(float)
    if means.shape != raw.shape or not np.isfinite(means).all():
        raise ValueError('finite means must match the four numeric input columns')
    means = np.where(np.isnan(raw), means, raw)
    patterns = missing_patterns(rows)
    result = np.repeat(means[:, None, :], samples, axis=1)
    if not np.any(patterns):
        return result
    pools = donor_pools(state, rows, exclude_reference)
    keys = rows[['building_id', 'building_type', 'hour', 'day_of_week', 'month']
                + NUMERIC_COLUMNS].copy()
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(np.uint64)
    lower, upper = state['bounds']
    for i, pattern in enumerate(patterns):
        if pattern == 0:
            continue
        pool = state['residuals'][int(pattern)][pools[i]].astype(float)
        pool = pool - pool.mean(axis=0)
        rng = np.random.default_rng(int(hashes[i]) ^ int(seed))
        noise = pool[rng.integers(0, len(pool), samples // 2)] * state['spread']
        missing = np.isnan(raw[i])
        radius = np.maximum(0., np.minimum(means[i] - lower, upper - means[i]))
        noise = np.clip(noise, -radius, radius)
        noise[:, ~missing] = 0.
        result[i, 0::2] += noise
        result[i, 1::2] -= noise
    # Copy observed values explicitly, including negative zero if present.
    observed = np.isfinite(raw)
    for column in range(4):
        result[observed[:, column], :, column] = raw[observed[:, column], column, None]
    assert np.isfinite(result).all()
    return result


def energy_predictions_with_values(parts, raw_rows, values, expected):
    """Reuse exact v2 features and estimators with explicit numeric fills.

    The expected-usage/occupancy features are computed from ORIGINAL inputs;
    a draw is never passed back through imputers or the neural input pipeline.
    """
    filled = raw_rows.copy()
    filled[NUMERIC_COLUMNS] = values
    features = ridge_features(filled, parts['typical']).copy()
    features['expected_previous'] = expected[:, 3]
    features['expected_occupancy'] = expected[:, 2]
    for building_type in BUILDING_TYPES:
        flag = (filled.building_type == building_type).to_numpy(float)
        features[f'expected_previous_{building_type}'] = expected[:, 3] * flag
        features[f'expected_occupancy_{building_type}'] = expected[:, 2] * flag
    ridge = parts['ridge'].predict(features)
    boosting = parts['boosting'].predict(tree_features(filled)) + values[:, 3]
    return .8 * ridge + .2 * boosting


def average_energy_predictions(parts, raw_rows, draws, expected, block_draws=8):
    n, count, _ = draws.shape
    if not n:
        return np.empty(0)
    total = np.zeros(n)
    for start in range(0, count, block_draws):
        block = draws[:, start:start + block_draws]
        k = block.shape[1]
        repeated = pd.concat([raw_rows] * k, ignore_index=True)
        values = block.transpose(1, 0, 2).reshape(k * n, 4)
        prediction = energy_predictions_with_values(
            parts, repeated, values, np.tile(expected, (k, 1)))
        total += prediction.reshape(k, n).sum(axis=0)
    return total / count


def imputed_branch_weight(settings):
    return (settings['weight_on_lightgbm_model']
            * (1 - settings.get('neural_net_weight', 0.))
            * (1 - settings.get('realmlp_weight', 0.)))


def predict_with_uncertainty(bundle, rows, version='v4_uncertainty', return_details=False):
    rows = rows.reset_index(drop=True)
    v2 = bundle['v2']
    settings = v2['versions']['dl13_prevfix']
    options = bundle['versions'][version]
    baseline = predict_version(v2, rows, settings)
    delta = np.zeros(len(rows))
    missing = missing_patterns(rows) > 0
    if options['alpha'] != 0. and missing.any():
        raw = rows.loc[missing].reset_index(drop=True)
        changes = []
        for seed in settings['seeds']:
            parts = v2['lightgbm_imputed'][seed]
            expected, means = expected_and_filled(parts, raw)
            draws = sample_missing_inputs(bundle['uncertainty'][seed], raw, means,
                                          options['samples'], options['seed'])
            point = predict_lightgbm_imputed(parts, raw)
            averaged = average_energy_predictions(parts, raw, draws, expected)
            changes.append(averaged - point)
        delta[missing] = imputed_branch_weight(settings) * np.mean(changes, axis=0)
    result = baseline.copy()
    # No change at all to complete rows or any original missingness decisions.
    result[missing] += options['alpha'] * delta[missing]
    assert np.isfinite(result).all()
    if return_details:
        return result, {'baseline': baseline, 'full_delta': delta,
                        'pattern': missing_patterns(rows)}
    return result
