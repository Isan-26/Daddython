"""v6: v4 + an engineered-input RealMLP ensemble (10 networks, numpy forward pass; no PyTorch needed).

Inputs are exactly those of neural_net_model.ipynb's network (seed-527 LightGBM imputers: filled values, expected
values, deviations from typical, heat/rain, time curves) plus the four 'was blank' flags, with building, building type
and weekday as categories. 'eng' networks predict energy_usage; 'eng_delta' networks predict the change from the
filled previous_usage. Final v6 prediction = (1 - w) x v4 + w x average of the 10 networks.
"""
import numpy as np
import pandas as pd
from uncertainty_runtime import *


def eng_inputs(v2, rows):
    lightgbm_model = StoredLightGBMModel(v2['lightgbm_imputed'][527])
    filled = lightgbm_model.fill_blanks(rows)
    numeric = network_inputs(lightgbm_model, filled, rows)[:, :20]
    flags = rows[NUMERIC_COLUMNS].isna().to_numpy(float)
    x = np.column_stack([numeric, flags]).astype(np.float32).astype(np.float64)    # pytabkit feeds float32
    codes = np.column_stack([
        rows['building_id'].map({b: i + 1 for i, b in enumerate(BUILDING_IDS)}).fillna(0).to_numpy(int),
        rows['building_type'].map({t: i + 1 for i, t in enumerate(BUILDING_TYPES)}).fillna(0).to_numpy(int),
        rows['day_of_week'].map({d: i + 1 for i, d in enumerate(DAYS_ALPHABETICAL)}).fillna(0).to_numpy(int)])
    return x, codes


def predict_realmlp_eng(bundle, rows):
    x, codes = eng_inputs(bundle['v2'], rows)
    predictions = [realmlp_predict(net, x, codes) + (x[:, 3] if float(net['add_previous_usage']) else 0.)
                   for net in bundle['realmlp_eng']['nets']]
    return np.mean(predictions, axis=0)


def predict_v6(bundle, rows, version='v6_eng'):
    rows = rows.reset_index(drop=True)
    options = bundle['versions'][version]
    if 'eng_weight' not in options:                     # v4 / v2 versions: exactly as in v4
        return predict_with_uncertainty(bundle, rows, version)
    base = predict_with_uncertainty(bundle, rows, options['base'])
    w = options['eng_weight']
    return (1 - w) * base + w * predict_realmlp_eng(bundle, rows)
