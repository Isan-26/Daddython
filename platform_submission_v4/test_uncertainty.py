"""Meaningful inference checks; no models are trained in these tests."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import uncertainty_runtime as runtime


def frame(count=4):
    return pd.DataFrame({'building_id': ['ADM_A'] * count,
        'building_type': ['Administration'] * count, 'hour': np.arange(count) % 24,
        'day_of_week': ['Monday'] * count, 'month': [6] * count,
        'temperature': [30.] * count, 'humidity': [60.] * count,
        'occupancy': [100.] * count, 'previous_usage': [40.] * count})


def sampler_state():
    donors = frame(8)
    errors = np.arange(8, dtype=float) - 3.5
    residuals = {}
    for pattern in range(1, 16):
        values = np.column_stack([errors, errors * 2, errors * 3, errors * 4])
        values[:, (pattern & (1 << np.arange(4))) == 0] = 0.
        residuals[pattern] = values
    return {'reference': {'numeric': donors[runtime.NUMERIC_COLUMNS].to_numpy(float),
                         'calendar': runtime.calendar_inputs(donors),
                         'building': donors.building_id.to_numpy()},
            'numeric_scale': np.ones(4), 'residuals': residuals,
            'neighbors': 8, 'spread': 1.,
            'bounds': np.array([[0., 0., 0., 0.], [100., 100., 1000., 1000.]])}


class UncertaintyTests(unittest.TestCase):
    def test_all_patterns_preserve_observed_values_and_point_means(self):
        rows = pd.concat([frame(1)] * 16, ignore_index=True)
        truth = rows[runtime.NUMERIC_COLUMNS].to_numpy(float)
        for p in range(16):
            for j, column in enumerate(runtime.NUMERIC_COLUMNS):
                if p & (1 << j):
                    rows.loc[p, column] = np.nan
        before = rows.copy(deep=True)
        draws = runtime.sample_missing_inputs(sampler_state(), rows, truth, samples=64)
        np.testing.assert_allclose(draws.mean(axis=1), truth, atol=1e-12, rtol=0)
        for j, column in enumerate(runtime.NUMERIC_COLUMNS):
            use = rows[column].notna().to_numpy()
            np.testing.assert_array_equal(draws[use, :, j], np.repeat(truth[use, j, None], 64, axis=1))
        pd.testing.assert_frame_equal(rows, before)
        self.assertTrue(np.isfinite(draws).all())
        self.assertGreater(np.std(draws[15, :, 0]), 0.)

    def test_shuffle_batch_id_and_label_independence(self):
        rows = frame(4)
        rows.loc[:, ['temperature', 'humidity']] = np.nan
        means = frame(4)[runtime.NUMERIC_COLUMNS].to_numpy(float)
        state = sampler_state()
        original = runtime.sample_missing_inputs(state, rows, means)
        order = np.array([3, 0, 2, 1])
        shuffled = rows.iloc[order].copy()
        shuffled['id'] = [99, 11, 12, 999]
        shuffled['energy_usage'] = [1e9, -1e9, 3., 4.]
        result = runtime.sample_missing_inputs(state, shuffled, means[order])
        np.testing.assert_array_equal(result, original[order])
        for i in range(len(rows)):
            np.testing.assert_array_equal(runtime.sample_missing_inputs(state, rows.iloc[[i]], means[[i]])[0], original[i])

    def test_joint_donor_errors_and_symmetric_bounds(self):
        rows = frame(1)
        rows.loc[:, ['temperature', 'humidity']] = np.nan
        means = np.array([[30., 60., 100., 40.]])
        draws = runtime.sample_missing_inputs(sampler_state(), rows, means)
        np.testing.assert_allclose(draws[0, :, 1] - 60., 2 * (draws[0, :, 0] - 30.))
        state = sampler_state()
        state['bounds'][0, 0] = 29.9
        state['bounds'][1, 0] = 30.1
        draws = runtime.sample_missing_inputs(state, rows, means)
        self.assertGreaterEqual(draws[:, :, 0].min(), 29.9)
        self.assertLessEqual(draws[:, :, 0].max(), 30.1)
        np.testing.assert_allclose(draws.mean(axis=1), means, atol=1e-12)

    def test_average_predictions_differs_from_prediction_of_average(self):
        rows = frame(1)
        rows.temperature = np.nan
        draws = np.array([[[-2., 60., 100., 40.], [2., 60., 100., 40.]]])
        with patch.object(runtime, 'energy_predictions_with_values',
                          side_effect=lambda parts, rows, values, expected: values[:, 0] ** 2):
            result = runtime.average_energy_predictions({}, rows, draws, np.zeros((1, 4)), block_draws=1)
        np.testing.assert_array_equal(result, [4.])
        self.assertEqual(float(draws[:, :, 0].mean()) ** 2, 0.)

    def test_raw_expected_features_do_not_change_with_draws(self):
        rows = frame(1)
        rows.previous_usage = np.nan
        class RidgeSpy:
            def predict(self, features):
                np.testing.assert_array_equal(features.expected_previous, [77.])
                np.testing.assert_array_equal(features.expected_occupancy, [99.])
                return np.zeros(len(features))
        class ZeroTree:
            def predict(self, features):
                return np.zeros(len(features))
        parts = {'ridge': RidgeSpy(), 'boosting': ZeroTree(), 'typical': None}
        with patch.object(runtime, 'ridge_features', side_effect=lambda rows, typical: rows[runtime.NUMERIC_COLUMNS].copy()):
            result = runtime.energy_predictions_with_values(parts, rows,
                np.array([[30., 60., 100., 999.]]), np.array([[30., 60., 99., 77.]]))
        np.testing.assert_allclose(result, [.2 * 999.])
        self.assertTrue(rows.previous_usage.isna().all())

    def test_only_imputed_branch_is_updated_and_zero_version_skips_sampler(self):
        rows = frame(2)
        rows.loc[1, 'previous_usage'] = np.nan
        settings = {'seeds': [527], 'weight_on_lightgbm_model': .75,
                    'neural_net_weight': .05, 'realmlp_weight': .13,
                    'previous_usage_blank_shift': .8}
        v2 = {'versions': {'dl13_prevfix': settings}, 'lightgbm_imputed': {527: {}}}
        bundle = {'v2': v2, 'uncertainty': {527: sampler_state()},
                  'versions': {'v4_uncertainty': {'alpha': 1., 'samples': 64, 'seed': 2026},
                               'v2_fallback': {'alpha': 0., 'samples': 64, 'seed': 2026}}}
        with patch.object(runtime, 'predict_version', return_value=np.array([3., 3.8])) as base, \
             patch.object(runtime, 'expected_and_filled', return_value=(np.zeros((1, 4)), np.zeros((1, 4)))), \
             patch.object(runtime, 'sample_missing_inputs', return_value=np.zeros((1, 64, 4))) as sampler, \
             patch.object(runtime, 'predict_lightgbm_imputed', return_value=np.array([1.])), \
             patch.object(runtime, 'average_energy_predictions', return_value=np.array([4.])):
            result = runtime.predict_with_uncertainty(bundle, rows)
            np.testing.assert_allclose(result, [3., 3.8 + 3 * .75 * .95 * .87])
            self.assertTrue(base.call_args.args[1].previous_usage.isna().iloc[1])
            sampler.reset_mock()
            np.testing.assert_array_equal(runtime.predict_with_uncertainty(bundle, rows, 'v2_fallback'), [3., 3.8])
            sampler.assert_not_called()

    def test_invalid_sample_count(self):
        with self.assertRaises(ValueError):
            runtime.sample_missing_inputs(sampler_state(), frame(1), np.ones((1, 4)), samples=3)

    def test_leave_self_out_donor_pool(self):
        rows = frame(1)
        rows.temperature = np.nan
        pools = runtime.donor_pools(sampler_state(), rows, exclude_reference=np.array([0]))
        self.assertNotIn(0, pools[0])


if __name__ == '__main__':
    unittest.main()
