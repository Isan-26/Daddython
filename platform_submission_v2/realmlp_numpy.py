"""RealMLP (pytabkit, 'TD' defaults) prediction in plain numpy, so it runs without PyTorch.

Exactly mirrors the fitted pytabkit model used in deep_learning_model.ipynb:
  inputs  -> 15 numeric columns (blanks filled with building x hour medians, plus 'was blank' flags)
           + building_id (learned embedding), building_type and day_of_week (one-hot)
  numeric -> shift/scale, smooth clip, periodic embeddings (PLR, 3 per column) + the clipped values
  network -> 3 hidden layers of 256 units (parametric mish), 1 output, then undo target scaling and clip.
"""
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ['temperature', 'humidity', 'occupancy', 'previous_usage']
BUILDING_IDS = ['ADM_A', 'BUS_A', 'BUS_B', 'ENG_A', 'ENG_B', 'LEC_A', 'LIB_A', 'RES_A', 'RES_B', 'SCI_A', 'SCI_B', 'SPT_A']
BUILDING_TYPES = ['Administration', 'Business', 'Engineering', 'LectureHall', 'Library', 'Residential', 'Science', 'Sports']
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
