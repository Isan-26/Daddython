"""An optional three-feature correction beside the unmodified v2 ensemble."""
import numpy as np
from frozen_v2 import *

WEATHER_FEATURES = [
    'occupancy_x_temperature',
    'occupancy_x_humidity',
    'occupancy_x_cooling27',
]


def complete_mask(rows):
    return np.isfinite(rows[NUMERIC_COLUMNS].to_numpy(float)).all(axis=1)


def weather_features(rows):
    occupancy = rows.occupancy.to_numpy(float)
    temperature = rows.temperature.to_numpy(float)
    humidity = rows.humidity.to_numpy(float)
    return np.column_stack([
        occupancy * temperature,
        occupancy * humidity,
        occupancy * np.maximum(temperature - 27., 0.),
    ])


def weather_delta(state, rows):
    result = np.zeros(len(rows))
    complete = complete_mask(rows)
    if state is not None and complete.any():
        x = weather_features(rows.loc[complete])
        result[complete] = ((x - state['center']) / state['scale']) @ state['coef'] + state['intercept']
    assert np.isfinite(result).all()
    return result


def predict_with_weather(bundle, rows):
    rows = rows.reset_index(drop=True)
    v2 = bundle['v2']
    baseline = predict_version(v2, rows, v2['versions']['dl13_prevfix'])
    alpha = bundle['correction']['alpha']
    if alpha == 0.:
        return baseline
    delta = weather_delta(bundle['correction']['ridge'], rows)
    # Originally incomplete rows are never assigned to this update.
    result = baseline.copy()
    complete = complete_mask(rows)
    result[complete] += alpha * delta[complete]
    assert np.isfinite(result).all()
    return result
