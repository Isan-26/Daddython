"""Inference shared by training and the self-contained submission notebook.

Artifacts contain dictionaries and library estimators only, never custom classes.
The frozen sklearn/RealMLP inputs live in legacy_runtime and are not altered here.
"""
import numpy as np
import pandas as pd
from legacy_runtime import *


def missing_patterns(rows):
    mask = rows[NUMERIC_COLUMNS].isna().to_numpy()
    return mask @ (1 << np.arange(4))


def hierarchy_predict(state, rows, column, spread=False):
    spec = state[column]
    result = np.full(len(rows), spec['global_std' if spread else 'global'], float)
    # Broad -> specific: each local estimate is shrunk toward its parent.
    for keys, table in spec['levels']:
        lookup = (pd.Index(rows[keys[0]], name=keys[0]) if len(keys) == 1
                  else pd.MultiIndex.from_frame(rows[keys]))
        matched = table.reindex(lookup)
        n = matched['count'].fillna(0).to_numpy(float)
        value = matched['std' if spread else 'median'].to_numpy(float)
        value = np.where(np.isfinite(value), value, result)
        result = (n * value + spec['smoothing'] * result) / (n + spec['smoothing'])
    return np.maximum(result, spec['epsilon']) if spread else result


def hierarchy_rows(rows):
    out = rows.copy()
    out['is_weekend'] = is_weekend(rows)
    return out


def auxiliary_inputs(rows, target):
    x = imputer_inputs(rows).drop(columns=target)
    for c in NUMERIC_COLUMNS:
        if c != target:
            x[c + '_missing'] = rows[c].isna().astype(float)
    return x


def extratrees_inputs(rows, target, fill=None):
    x = auxiliary_inputs(rows, target).copy()
    for c in ['building_id', 'building_type', 'day_of_week']:
        x[c] = x[c].cat.codes.astype(float)
    a = x.to_numpy(float)
    if fill is None:
        fill = np.nanmedian(a, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0)
    return np.where(np.isfinite(a), a, fill), fill


def predict_imputers(parts, rows, two_pass=True):
    rows = rows.reset_index(drop=True)
    contextual = hierarchy_rows(rows)
    mean, std, extra, median = [np.empty((len(rows), 4)) for _ in range(4)]
    for j, c in enumerate(NUMERIC_COLUMNS):
        x = auxiliary_inputs(rows, c)
        seed_predictions = np.array([m.predict(x) for m in parts['lightgbm'][c]])
        mean[:, j], std[:, j] = seed_predictions.mean(0), seed_predictions.std(0)
        ex, _ = extratrees_inputs(rows, c, parts['extra_fill'][c])
        extra[:, j] = parts['extra'][c].predict(ex)
        median[:, j] = hierarchy_predict(parts['hierarchy'], contextual, c)
    second, second_std, second_extra = mean.copy(), std.copy(), extra.copy()
    multi = rows[NUMERIC_COLUMNS].isna().sum(axis=1).to_numpy() >= 2
    if two_pass and multi.any():
        # Simultaneous pass 2: all targets see the same completed pass-1 row.
        completed = rows.copy()
        completed[NUMERIC_COLUMNS] = np.where(rows[NUMERIC_COLUMNS].isna(), mean,
                                             rows[NUMERIC_COLUMNS].to_numpy(float))
        for j, c in enumerate(NUMERIC_COLUMNS):
            positions = np.flatnonzero(multi & rows[c].isna().to_numpy())
            if not len(positions):
                continue
            q = completed.iloc[positions]
            p = np.array([m.predict(auxiliary_inputs(q, c)) for m in parts['lightgbm'][c]])
            second[positions, j], second_std[positions, j] = p.mean(0), p.std(0)
            ex, _ = extratrees_inputs(q, c, parts['extra_fill'][c])
            second_extra[positions, j] = parts['extra'][c].predict(ex)
    return {'lightgbm': mean, 'std': std, 'extra': extra, 'median': median,
            'second': second, 'second_std': second_std, 'second_extra': second_extra}


def blend_imputations(predictions, weights, second=False):
    mean = predictions['second' if second else 'lightgbm']
    extra = predictions['second_extra' if second else 'extra']
    methods = np.stack([mean, extra, predictions['median']], axis=2)
    if np.asarray(weights).ndim == 2:
        weights = np.asarray(weights)[None, :, :]
    return {'mean': (methods * weights).sum(2),
            'std': predictions['second_std' if second else 'std'],
            'disagreement': methods.std(2)}


def complete_rows(rows, auxiliary):
    filled = rows.reset_index(drop=True).copy()
    raw = filled[NUMERIC_COLUMNS].to_numpy(float)
    filled[NUMERIC_COLUMNS] = np.where(np.isnan(raw), auxiliary['mean'], raw)
    return filled


def enhanced_features(rows, auxiliary, state, family):
    rows = rows.reset_index(drop=True)
    filled = complete_rows(rows, auxiliary)
    f = ridge_features(filled, state['typical']).copy()
    raw_missing = rows[NUMERIC_COLUMNS].isna().to_numpy(float)
    day = rows['day_of_week'].map(DAY_NUMBER).to_numpy()
    types = (rows['building_type'].to_numpy()[:, None] == np.array(BUILDING_TYPES)).astype(float)
    columns = {}
    for j, c in enumerate(NUMERIC_COLUMNS):
        name = 'previous' if c == 'previous_usage' else c
        columns['expected_' + name] = auxiliary['mean'][:, j]
        columns[c + '_missing'] = raw_missing[:, j]
        if c in ['occupancy', 'previous_usage']:
            for t, type_name in enumerate(BUILDING_TYPES):
                columns[f'expected_{name}_{type_name}'] = auxiliary['mean'][:, j] * types[:, t]
    pattern = raw_missing @ (1 << np.arange(4))
    columns['missing_count'] = raw_missing.sum(1)
    columns['missing_pattern'] = pattern
    for p in range(1, 16):
        columns[f'missing_pattern_{p}'] = (pattern == p).astype(float)
    # Each family is evaluated separately; "combined" is the only combined trial.
    groups = ['uncertainty', 'weekday', 'weather', 'deviations', 'physical'] if family == 'combined' else [family]
    if 'uncertainty' in groups:
        for j, c in enumerate(NUMERIC_COLUMNS):
            columns[c + '_imputer_std'] = auxiliary['std'][:, j]
            columns[c + '_imputer_disagreement'] = auxiliary['disagreement'][:, j]
            columns[c + '_missing_std'] = raw_missing[:, j] * auxiliary['std'][:, j]
    if 'weekday' in groups:
        for d, name in enumerate(DAY_NAMES):
            flag = (day == d).astype(float)
            columns['weekday_' + name] = flag
            for b in BUILDING_IDS:
                columns[f'{b}_{name}'] = flag * (rows.building_id.to_numpy() == b)
    if 'weather' in groups:
        temperature, humidity = filled.temperature.to_numpy(), filled.humidity.to_numpy()
        columns['temperature_sq'] = temperature ** 2
        columns['humidity_sq'] = humidity ** 2
        for threshold in [24, 27, 30]:
            columns[f'cooling_{threshold}'] = np.maximum(temperature - threshold, 0)
        for threshold in [80, 90]:
            columns[f'humid_{threshold}'] = np.maximum(humidity - threshold, 0)
        columns['temperature_x_humidity'] = temperature * humidity
        columns['cooling27_x_humidity'] = np.maximum(temperature - 27, 0) * humidity
    if 'deviations' in groups:
        contextual = hierarchy_rows(rows)
        for j, c in enumerate(NUMERIC_COLUMNS):
            value = filled[c].to_numpy()
            # Surprises are zero for imputed values: they were never observed.
            columns[c + '_surprise'] = (1 - raw_missing[:, j]) * (value - auxiliary['mean'][:, j])
            if c in ['occupancy', 'previous_usage']:
                typical = hierarchy_predict(state['hierarchy'], contextual, c)
                spread = hierarchy_predict(state['hierarchy'], contextual, c, spread=True)
                columns[c + '_hier_typical'] = typical
                columns[c + '_hier_gap'] = value - typical
                columns[c + '_hier_ratio'] = value / np.maximum(np.abs(typical), 1)
                columns[c + '_zscore'] = np.clip((value - typical) / spread, -8, 8)
    if 'physical' in groups:
        for a, b in [('occupancy', 'temperature'), ('occupancy', 'humidity'),
                     ('occupancy', 'previous_usage'), ('temperature', 'previous_usage'),
                     ('humidity', 'previous_usage'), ('temperature', 'humidity')]:
            columns[a + '_x_' + b] = filled[a].to_numpy() * filled[b].to_numpy()
        for c in ['occupancy', 'previous_usage', 'temperature']:
            columns[c + '_sq'] = filled[c].to_numpy() ** 2
        for c in ['hour_sin', 'hour_cos']:
            columns['occupancy_x_' + c] = filled.occupancy.to_numpy() * f[c].to_numpy()
    extra = pd.DataFrame(columns)
    # Avoid duplicate legacy columns (occupancy_x_weekend etc. are already present).
    f = pd.concat([f, extra.loc[:, ~extra.columns.isin(f.columns)]], axis=1)
    if not np.isfinite(f.to_numpy(float)).all():
        raise ValueError('Non-finite engineered features')
    return f


def energy_tree_inputs(rows, auxiliary):
    filled = complete_rows(rows, auxiliary)
    x = tree_features(filled)
    for c in NUMERIC_COLUMNS:
        x[c + '_missing'] = rows[c].isna().to_numpy(float)
    x['missing_count'] = rows[NUMERIC_COLUMNS].isna().sum(axis=1).to_numpy()
    return x


def predict_energy(parts, rows, auxiliary, family=None):
    family = family or parts['family']
    filled = complete_rows(rows, auxiliary)
    return (0.8 * parts['ridge'][family].predict(enhanced_features(rows, auxiliary, parts, family))
            + 0.2 * (parts['boosting'].predict(energy_tree_inputs(rows, auxiliary))
                     + filled.previous_usage.to_numpy()))


def apply_strategy(rows, direct, one_pass, two_pass, strategy):
    patterns = missing_patterns(rows)
    weight = np.array([s['direct_weight'] for s in strategy])[patterns]
    second = np.array([s['two_pass'] for s in strategy])[patterns]
    imputed = np.where(second, two_pass, one_pass)
    return weight * direct + (1 - weight) * imputed


def bias_values(rows, correction):
    missing = rows.previous_usage.isna().to_numpy()
    values = rows.building_type.map(correction.get('by_type', {})).fillna(correction['global']).to_numpy(float)
    return missing * values


def predict_upgrade(bundle, rows):
    rows = rows.reset_index(drop=True)
    legacy = bundle['legacy']
    if bundle['accepted']:
        parts = bundle['upgrade']
        raw = predict_imputers(parts['imputers'], rows)
        one = blend_imputations(raw, parts['imputation_weights'])
        two = blend_imputations(raw, parts['imputation_weights'], second=True)
        direct = predict_patterns(parts['pattern_models'], rows)
        p1, p2 = predict_energy(parts['energy'], rows, one), predict_energy(parts['energy'], rows, two)
        mix = apply_strategy(rows, direct, p1, p2, parts['strategy'])
        non_dl = (0.0625 * predict_v3(legacy['v3'], rows)
                  + 0.9375 * (mix + bias_values(rows, parts['correction'])))
    else:
        non_dl = predict_final(legacy, rows, [527], 0.75)
        non_dl += bias_values(rows, bundle['fallback_correction'])
    # The pretrained experts keep their exact input layouts and effective weights.
    return (0.8265 * non_dl + 0.0435 * predict_neural_net(legacy, rows)
            + 0.13 * predict_realmlp(legacy, rows))
