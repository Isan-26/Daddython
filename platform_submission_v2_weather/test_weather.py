import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_weather_correction as training
import weather_runtime as runtime


def rows(n=16):
    return pd.DataFrame({'id': np.arange(n), 'temperature': np.full(n, 30.),
        'humidity': np.full(n, 70.), 'occupancy': np.full(n, 2.),
        'previous_usage': np.full(n, 40.), 'energy_usage': np.full(n, 1.)})


def state():
    return {'center': np.zeros(3), 'scale': np.ones(3),
            'coef': np.array([.01, .02, .03]), 'intercept': .5}


class WeatherChecks(unittest.TestCase):
    def test_three_features_and_no_target_input(self):
        q = rows(1)
        np.testing.assert_array_equal(runtime.weather_features(q), [[60., 140., 6.]])
        q.energy_usage = 1e9
        np.testing.assert_array_equal(runtime.weather_features(q), [[60., 140., 6.]])

    def test_all_patterns_skip_incomplete_rows_and_preserve_inputs(self):
        q = rows()
        for pattern in range(16):
            for bit, c in enumerate(runtime.NUMERIC_COLUMNS):
                if pattern & (1 << bit):
                    q.loc[pattern, c] = np.nan
        original = q.copy(deep=True)
        delta = runtime.weather_delta(state(), q)
        self.assertNotEqual(delta[0], 0.)
        np.testing.assert_array_equal(delta[1:], np.zeros(15))
        pd.testing.assert_frame_equal(q, original)

    def test_zero_alpha_does_not_call_corrector(self):
        baseline = np.arange(16, dtype=float)
        v2 = {'versions': {'dl13_prevfix': {'previous_usage_blank_shift': .8}}}
        bundle = {'v2': v2, 'correction': {'alpha': 0., 'ridge': state()}}
        with patch.object(runtime, 'predict_version', return_value=baseline) as frozen:
            with patch.object(runtime, 'weather_delta', side_effect=AssertionError('must be skipped')):
                prediction = runtime.predict_with_weather(bundle, rows())
        self.assertIs(prediction, baseline)
        self.assertIs(frozen.call_args.args[0], v2)
        self.assertIs(frozen.call_args.args[2], v2['versions']['dl13_prevfix'])

    def test_nonzero_alpha_preserves_every_missing_row(self):
        q = rows()
        for i, c in enumerate(runtime.NUMERIC_COLUMNS):
            q.loc[i + 1, c] = np.nan
        baseline = np.arange(16, dtype=float)
        bundle = {'v2': {'versions': {'dl13_prevfix': {}}},
                  'correction': {'alpha': .1, 'ridge': state()}}
        with patch.object(runtime, 'predict_version', return_value=baseline):
            prediction = runtime.predict_with_weather(bundle, q)
        np.testing.assert_array_equal(prediction[1:5], baseline[1:5])
        self.assertNotEqual(prediction[0], baseline[0])
        np.testing.assert_array_equal(baseline, np.arange(16, dtype=float))

    def test_corrector_validation_excludes_held_out_targets(self):
        q = rows(12)
        folds = np.repeat(np.arange(3), 4)
        def fit(fit_rows, residual):
            return set(fit_rows.id)
        def predict(seen, check_rows):
            self.assertFalse(seen.intersection(check_rows.id))
            return np.zeros(len(check_rows))
        with patch.object(training, 'fit_correction', side_effect=fit):
            with patch.object(training, 'weather_delta', side_effect=predict):
                delta = training.correction_oof(q, np.zeros(len(q)), folds, np.arange(len(q)))
        np.testing.assert_array_equal(delta, np.zeros(len(q)))

    def test_negligible_gain_keeps_zero_correction(self):
        q = rows(100)
        baseline = np.full(100, .9995)
        delta = np.full(100, .0005)
        folds = np.repeat(np.arange(5), 20)
        alpha, diagnostics = training.choose_alpha(q, baseline, delta, folds, np.arange(100))
        self.assertEqual(alpha, 0.)
        self.assertTrue(all(not r['qualifies'] for r in diagnostics if r['alpha'] != 0.))


if __name__ == '__main__':
    unittest.main()
