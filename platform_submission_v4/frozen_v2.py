"""Exact inference functions copied from the v2 prediction notebook."""
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ['temperature', 'humidity', 'occupancy', 'previous_usage']
BUILDING_IDS = ['ADM_A', 'BUS_A', 'BUS_B', 'ENG_A', 'ENG_B', 'LEC_A', 'LIB_A', 'RES_A', 'RES_B', 'SCI_A', 'SCI_B', 'SPT_A']
BUILDING_TYPES = ['Administration', 'Business', 'Engineering', 'LectureHall', 'Library', 'Residential', 'Science', 'Sports']
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_NUMBER = {day: number for number, day in enumerate(DAY_NAMES)}
BUILDING_NUMBER = {building: number for number, building in enumerate(BUILDING_IDS)}
IMPUTER_CATEGORIES = {'building_id': BUILDING_IDS, 'building_type': BUILDING_TYPES, 'day_of_week': DAY_NAMES}


def is_weekend(rows):
    return rows['day_of_week'].isin(['Saturday', 'Sunday'])

# ---- features shared by the v3 model and the LightGBM-imputed model ----
def missforest_inputs(rows):
    inputs = rows[NUMERIC_COLUMNS].copy()
    inputs['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
    inputs['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
    inputs['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
    inputs['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
    inputs['is_weekend'] = is_weekend(rows).astype(float)
    building_onehot = pd.get_dummies(pd.Categorical(rows['building_id'], categories=BUILDING_IDS), prefix='bid').astype(float)
    building_onehot.index = rows.index
    return pd.concat([inputs, building_onehot], axis=1)


def ridge_features(rows, typical):
    features = rows[NUMERIC_COLUMNS].copy()
    features['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
    features['is_weekend'] = is_weekend(rows).astype(float)
    features['occupancy_x_weekend'] = features['occupancy'] * features['is_weekend']
    building_onehot = pd.get_dummies(pd.Categorical(rows['building_id'], categories=BUILDING_IDS), prefix='bid').astype(float)
    type_onehot = pd.get_dummies(pd.Categorical(rows['building_type'], categories=BUILDING_TYPES), prefix='bt').astype(float)
    building_onehot.index = type_onehot.index = rows.index
    interactions = {}
    for column in ['occupancy', 'previous_usage', 'temperature', 'hour_sin', 'hour_cos']:
        for building in building_onehot.columns:
            interactions[f'{column}_x_{building}'] = features[column] * building_onehot[building]
    for column in ['is_weekend', 'occupancy_x_weekend']:
        for building_type in type_onehot.columns:
            interactions[f'{column}_x_{building_type}'] = features[column] * type_onehot[building_type]
    lookup_keys = pd.DataFrame({'building_id': rows['building_id'], 'hour': rows['hour'], 'is_weekend': features['is_weekend'] > 0})
    typical_for_row = lookup_keys.join(typical, on=['building_id', 'hour', 'is_weekend']).fillna(typical.median())
    events = pd.DataFrame({
        'occupancy_gap': features['occupancy'] - typical_for_row['occupancy'],
        'previous_ratio': features['previous_usage'] / typical_for_row['previous_usage'],
        'previous_gap': features['previous_usage'] - typical_for_row['previous_usage'],
        'heat': (features['temperature'] >= 33.5).astype(float),
        'rain': (features['humidity'] >= 93).astype(float),
    }, index=rows.index)
    for column in events.columns:
        for building_type in type_onehot.columns:
            interactions[f'{column}_x_{building_type}'] = events[column] * type_onehot[building_type]
    part_of_day = pd.cut(rows['hour'], [-1, 5, 8, 11, 14, 17, 20, 23], labels=False)
    part_onehot = pd.get_dummies(pd.Categorical(part_of_day, categories=range(7)), prefix='dp').astype(float)
    part_onehot.index = rows.index
    for part in part_onehot.columns:
        for building_type in type_onehot.columns:
            interactions[f'occupancy_{part}_x_{building_type}'] = features['occupancy'] * part_onehot[part] * type_onehot[building_type]
    return pd.concat([features, type_onehot, building_onehot, events, pd.DataFrame(interactions, index=rows.index)], axis=1)


def tree_features(rows):
    features = rows[['hour', 'month'] + NUMERIC_COLUMNS].copy()
    features['day'] = rows['day_of_week'].map(DAY_NUMBER)
    features['is_weekend'] = (features['day'] >= 5).astype(int)
    features['building_type'] = pd.Categorical(rows['building_type'], categories=BUILDING_TYPES)
    features['building_id'] = pd.Categorical(rows['building_id'], categories=BUILDING_IDS)
    return features

# ---- v4 part 1: the v3 notebook model (MissForest filling, Ridge 80% + boosting 20%) ----
def predict_v3(parts, rows):
    filled = rows.copy()
    filled[NUMERIC_COLUMNS] = parts['imputer'].transform(missforest_inputs(rows))[:, :len(NUMERIC_COLUMNS)]
    ridge_prediction = parts['ridge'].predict(ridge_features(filled, parts['typical']))
    boosting_prediction = parts['boosting'].predict(tree_features(filled)) + filled['previous_usage'].to_numpy()
    return 0.8 * ridge_prediction + 0.2 * boosting_prediction

# ---- v4 part 2: one model per pattern of missing inputs ----
def pattern_model_inputs(rows, available_columns, n_daily_curves=3):
    hour = rows['hour'].to_numpy(float)
    columns = {name: rows[name].to_numpy(float) for name in available_columns}
    columns['weekend'] = is_weekend(rows).to_numpy(float)
    for k in range(1, n_daily_curves + 1):
        columns[f'hour_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24)
        columns[f'hour_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24)
    columns['month_sin'] = np.sin(2 * np.pi * rows['month'].to_numpy() / 12)
    columns['month_cos'] = np.cos(2 * np.pi * rows['month'].to_numpy() / 12)
    return np.column_stack(list(columns.values()))


def pattern_ridge_inputs(rows, available_columns, scaler):
    scaled = scaler.transform(pattern_model_inputs(rows, available_columns))
    building_onehot = (rows['building_id'].to_numpy()[:, None] == np.array(BUILDING_IDS)[None, :]).astype(float)
    per_building = (scaled[:, :, None] * building_onehot[:, None, :]).reshape(len(scaled), -1)
    return np.column_stack([scaled, building_onehot, per_building])


def pattern_correction_inputs(rows, available_columns):
    numeric = rows[['hour', 'month'] + available_columns].to_numpy(float)
    return np.column_stack([numeric, rows['day_of_week'].map(DAY_NUMBER), rows['building_id'].map(BUILDING_NUMBER)])


def predict_patterns(pattern_models, rows, correction_weight=0.65):
    blanks = rows[NUMERIC_COLUMNS].isna().to_numpy().astype(int)
    patterns = (blanks * 2 ** np.arange(len(NUMERIC_COLUMNS))).sum(axis=1)
    prediction = np.empty(len(rows))
    for pattern in np.unique(patterns):
        positions = np.flatnonzero(patterns == pattern)
        pattern_rows = rows.iloc[positions]
        available, scaler, ridge, correction = pattern_models[pattern]
        prediction[positions] = (ridge.predict(pattern_ridge_inputs(pattern_rows, available, scaler))
                                 + correction_weight * correction.predict(pattern_correction_inputs(pattern_rows, available)))
    return prediction


def predict_v4(model, rows):
    return 0.25 * predict_v3(model['v3'], rows) + 0.75 * predict_patterns(model['pattern_models'], rows)

# ---- the LightGBM-imputed model ----
def imputer_inputs(rows):
    inputs = rows[['building_id', 'building_type', 'hour', 'day_of_week', 'month'] + NUMERIC_COLUMNS].copy()
    inputs['weekend'] = is_weekend(rows).astype(int)
    inputs['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
    inputs['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
    inputs['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
    inputs['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
    for column, categories in IMPUTER_CATEGORIES.items():
        inputs[column] = pd.Categorical(inputs[column], categories=categories)
    return inputs


def predict_lightgbm_imputed(parts, rows):
    imputers = parts['imputers']
    expected = lambda frame, column: imputers[column].predict(imputer_inputs(frame).drop(columns=column))
    filled = rows.copy()
    for column in NUMERIC_COLUMNS:                       # fill each blank with the imputer's expected value
        blank = rows[column].isna()
        if blank.any():
            filled.loc[blank, column] = expected(rows.loc[blank], column)
    features = ridge_features(filled, parts['typical']).copy()
    expected_previous, expected_occupancy = expected(rows, 'previous_usage'), expected(rows, 'occupancy')
    features['expected_previous'] = expected_previous
    features['expected_occupancy'] = expected_occupancy
    for building_type in BUILDING_TYPES:
        this_type = (filled['building_type'] == building_type).to_numpy(float)
        features[f'expected_previous_{building_type}'] = expected_previous * this_type
        features[f'expected_occupancy_{building_type}'] = expected_occupancy * this_type
    ridge_prediction = parts['ridge'].predict(features)
    boosting_prediction = parts['boosting'].predict(tree_features(filled)) + filled['previous_usage'].to_numpy()
    return 0.8 * ridge_prediction + 0.2 * boosting_prediction


def predict_final(model, rows, seeds, weight_on_lightgbm_model):
    v4 = predict_v4(model, rows)
    lightgbm_model = np.mean([predict_lightgbm_imputed(model['lightgbm_imputed'][seed], rows) for seed in seeds], axis=0)
    return (1 - weight_on_lightgbm_model) * v4 + weight_on_lightgbm_model * lightgbm_model

class StoredLightGBMModel:
    """The LightGBM-imputed model's imputers and typical values, with the methods the network inputs need."""

    def __init__(self, parts):
        self.imputers, self.typical = parts['imputers'], parts['typical']

    def expected_value(self, rows, column):
        return self.imputers[column].predict(imputer_inputs(rows).drop(columns=column))

    def fill_blanks(self, rows):
        filled = rows.copy()
        for column in NUMERIC_COLUMNS:
            blank = rows[column].isna()
            if blank.any():
                filled.loc[blank, column] = self.expected_value(rows.loc[blank], column)
        return filled


def network_inputs(lightgbm_model, filled_rows, original_rows):
    hour = filled_rows['hour'].to_numpy(float)
    weekend = is_weekend(filled_rows).to_numpy()
    columns = {name: filled_rows[name].to_numpy(float) for name in NUMERIC_COLUMNS}
    columns['expected_previous'] = lightgbm_model.expected_value(original_rows, 'previous_usage')
    columns['expected_occupancy'] = lightgbm_model.expected_value(original_rows, 'occupancy')

    # how unusual the row is compared with its building's typical values (same lookup as ridge_features)
    typical = lightgbm_model.typical
    lookup_keys = pd.DataFrame({'building_id': filled_rows['building_id'], 'hour': filled_rows['hour'], 'is_weekend': weekend})
    typical_for_row = lookup_keys.join(typical, on=['building_id', 'hour', 'is_weekend']).fillna(typical.median())
    columns['occupancy_gap'] = columns['occupancy'] - typical_for_row['occupancy'].to_numpy()
    columns['previous_gap'] = columns['previous_usage'] - typical_for_row['previous_usage'].to_numpy()
    columns['previous_ratio'] = columns['previous_usage'] / typical_for_row['previous_usage'].to_numpy()
    columns['heat'] = (columns['temperature'] >= 33.5).astype(float)
    columns['rain'] = (columns['humidity'] >= 93).astype(float)

    columns['weekend'] = weekend.astype(float)
    for k in range(1, 4):                                       # once, twice, three times a day
        columns[f'hour_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24)
        columns[f'hour_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24)
    columns['month_sin'] = np.sin(2 * np.pi * filled_rows['month'].to_numpy() / 12)
    columns['month_cos'] = np.cos(2 * np.pi * filled_rows['month'].to_numpy() / 12)

    building_onehot = (filled_rows['building_id'].to_numpy()[:, None] == np.array(BUILDING_IDS)[None, :]).astype(float)
    type_onehot = (filled_rows['building_type'].to_numpy()[:, None] == np.array(BUILDING_TYPES)[None, :]).astype(float)
    return np.column_stack(list(columns.values()) + [building_onehot, type_onehot])


def predict_neural_net(model, rows):
    parts = model['neural_net']
    lightgbm_model = StoredLightGBMModel(model['lightgbm_imputed'][parts['lightgbm_seed']])
    filled = lightgbm_model.fill_blanks(rows)
    inputs = parts['scaler'].transform(network_inputs(lightgbm_model, filled, rows))
    change = np.mean([network.predict(inputs) for network in parts['networks']], axis=0)
    return change * parts['change_std'] + parts['change_mean'] + filled['previous_usage'].to_numpy()

DAYS_ALPHABETICAL = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']   # pytabkit's category order


def realmlp_inputs(rows, medians):
    """The 15 numeric inputs (same order and blank filling as training) and the 3 category codes (1-based, 0 = unknown)."""
    building = rows['building_id'].map({b: i for i, b in enumerate(BUILDING_IDS)}).to_numpy()
    hour = rows['hour'].to_numpy()
    numeric = {
        'hour': rows['hour'].to_numpy(float), 'month': rows['month'].to_numpy(float),
        'weekend': rows['day_of_week'].isin(['Saturday', 'Sunday']).to_numpy(float),
        'hour_sin': np.sin(2 * np.pi * rows['hour'].to_numpy(float) / 24), 'hour_cos': np.cos(2 * np.pi * rows['hour'].to_numpy(float) / 24),
        'month_sin': np.sin(2 * np.pi * rows['month'].to_numpy(float) / 12), 'month_cos': np.cos(2 * np.pi * rows['month'].to_numpy(float) / 12),
    }
    for column in NUMERIC_COLUMNS:
        values = rows[column].to_numpy(float)
        fill = medians[f'{column}_by_building_hour'][building, hour]                  # building x hour median
        fill = np.where(np.isnan(fill), medians[f'{column}_by_building'][building], fill)   # fallback: building median
        numeric[column] = np.where(np.isnan(values), fill, values)
        numeric[f'{column}_blank'] = np.isnan(values).astype(float)
    order = ['hour', 'month', 'weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
             'temperature', 'temperature_blank', 'humidity', 'humidity_blank', 'occupancy', 'occupancy_blank',
             'previous_usage', 'previous_usage_blank']
    x = np.column_stack([numeric[name] for name in order]).astype(np.float32).astype(np.float64)   # pytabkit feeds float32
    codes = np.column_stack([
        rows['building_id'].map({b: i + 1 for i, b in enumerate(BUILDING_IDS)}).fillna(0).to_numpy(int),
        rows['building_type'].map({t: i + 1 for i, t in enumerate(BUILDING_TYPES)}).fillna(0).to_numpy(int),
        rows['day_of_week'].map({d: i + 1 for i, d in enumerate(DAYS_ALPHABETICAL)}).fillna(0).to_numpy(int)])
    return x, codes


def _smooth_clip(x, max_abs_value=3.0):
    return x / np.sqrt(1 + x ** 2 / max_abs_value ** 2)


def _mish(x):
    return x * np.tanh(np.logaddexp(0, x))


def realmlp_predict(p, x, codes):
    """Forward pass of one fitted RealMLP; p = dict of numpy weights for one seed."""
    # numeric branch: robust shift/scale, smooth clip, periodic (PLR) embeddings + the clipped values
    z = _smooth_clip((x + p['num_bias']) * p['num_scale'])
    h = np.cos(2 * np.pi * z[:, :, None] * p['plr_w1'][None, :, :] + p['plr_b1'][None, :, :])     # (n, 15, 16)
    h = np.einsum('njk,jkl->njl', h, p['plr_w2']) + p['plr_b2'][None, :, :]                       # (n, 15, 3)
    numeric_out = np.concatenate([h.reshape(len(x), -1), z], axis=1)                              # (n, 60)
    # categorical branch: one-hot type and day (missing slot dropped), shift/scale, smooth clip; building embedding
    onehot = np.concatenate([(codes[:, 1:2] == np.arange(1, p['type_cats'] + 1)).astype(float),
                             (codes[:, 2:3] == np.arange(1, p['day_cats'] + 1)).astype(float)], axis=1)
    categorical_out = np.concatenate([_smooth_clip((onehot + p['cat_bias']) * p['cat_scale']), p['emb'][codes[:, 0]]], axis=1)
    h = np.concatenate([numeric_out, categorical_out], axis=1) * p['scale0']                       # (n, 83)
    for k in (1, 2, 3):
        h = h @ p[f'w{k}'] * p[f'f{k}'] + p[f'b{k}']
        h = h + (_mish(h) - h) * p[f'a{k}']
    out = (h @ p['w4'] * p['f4'] + p['b4'])[:, 0]
    return np.clip(out * p['out_std'] + p['out_mean'], p['out_low'], p['out_high'])


def predict_realmlp(model, rows):
    parts = model['realmlp']
    x, codes = realmlp_inputs(rows, parts['medians'])
    return np.mean([realmlp_predict(weights, x, codes) for weights in parts['seeds']], axis=0)


def predict_version(model, rows, settings):
    prediction = predict_final(model, rows, settings['seeds'], settings['weight_on_lightgbm_model'])
    if 'neural_net_weight' in settings:
        prediction = (1 - settings['neural_net_weight']) * prediction + settings['neural_net_weight'] * predict_neural_net(model, rows)
    if 'realmlp_weight' in settings:
        prediction = (1 - settings['realmlp_weight']) * prediction + settings['realmlp_weight'] * predict_realmlp(model, rows)
    if 'previous_usage_blank_shift' in settings:
        prediction = prediction + settings['previous_usage_blank_shift'] * rows['previous_usage'].isna().to_numpy()
    return prediction
