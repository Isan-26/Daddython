"""Small integration checks for the new imputation and feature contracts."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import feature_imputation_upgrade as upgrade
from upgrade_runtime import *


class UpgradeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(upgrade.ROOT / 'train.csv').iloc[:240].copy()
        cls.rows = pd.read_csv(upgrade.ROOT / 'train.csv').iloc[240:260].drop(columns='energy_usage').reset_index(drop=True)
        cls.config = {'seeds': [527], 'estimators': 12, 'extra_trees': 8, 'folds': 3}
        cls.parts = upgrade.fit_imputers(cls.train, cls.config)

    def test_target_value_is_excluded_from_own_auxiliary_prediction(self):
        changed = self.rows.copy()
        changed['previous_usage'] = 1e6
        before = predict_imputers(self.parts, self.rows, two_pass=False)
        after = predict_imputers(self.parts, changed, two_pass=False)
        np.testing.assert_array_equal(before['lightgbm'][:, 3], after['lightgbm'][:, 3])
        np.testing.assert_array_equal(before['extra'][:, 3], after['extra'][:, 3])
        self.assertNotIn('previous_usage', auxiliary_inputs(self.rows, 'previous_usage'))
        self.assertNotIn('energy_usage', auxiliary_inputs(self.rows, 'previous_usage'))

    def test_original_flags_surprises_and_observed_values(self):
        rows = self.rows.copy()
        rows.loc[0, ['occupancy', 'previous_usage']] = np.nan
        original = rows.copy(deep=True)
        p = predict_imputers(self.parts, rows)
        a = blend_imputations(p, np.tile([1., 0., 0.], (4, 1)))
        filled = complete_rows(rows, a)
        observed = rows[NUMERIC_COLUMNS].notna().to_numpy()
        np.testing.assert_array_equal(filled[NUMERIC_COLUMNS].to_numpy()[observed], rows[NUMERIC_COLUMNS].to_numpy()[observed])
        state = {'typical': upgrade.lt.typical_values(filled), 'hierarchy': self.parts['hierarchy']}
        features = enhanced_features(rows, a, state, 'combined')
        self.assertEqual(features.loc[0, 'missing_pattern'], 12)
        self.assertEqual(features.loc[0, 'missing_count'], 2)
        self.assertEqual(features.loc[0, 'occupancy_surprise'], 0)
        self.assertEqual(features.loc[0, 'previous_usage_surprise'], 0)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        pd.testing.assert_frame_equal(rows, original)

    def test_unseen_hierarchy_groups_have_finite_fallbacks(self):
        rows = hierarchy_rows(self.rows.copy())
        rows['building_id'] = 'UNKNOWN'
        for c in NUMERIC_COLUMNS:
            self.assertTrue(np.isfinite(hierarchy_predict(self.parts['hierarchy'], rows, c)).all())

    def test_mask_views_preserve_source_groups_and_hidden_truth(self):
        data = upgrade.crossfit_imputers(self.train.iloc[:60], self.config)
        self.assertTrue(all(np.isfinite(a).all() for a in data['predictions'].values()))
        self.assertEqual(set(data['scenario']), {-1, *range(1, 16)})
        for fold in range(3):
            fit = np.flatnonzero(data['folds'] != fold)
            check = np.flatnonzero(data['folds'] == fold)
            self.assertFalse(np.intersect1d(fit, check).size)
            positions, _ = upgrade.training_positions(data, self.train.iloc[:60], fit)
            self.assertFalse(np.intersect1d(data['origin'][positions], check).size)

    def test_svd_retry_preserves_ridge_predictions(self):
        rng = np.random.default_rng(2026)
        x = pd.DataFrame(rng.normal(size=(60, 6)))
        x[6] = x[0]  # A deliberately rank-deficient design.
        y = rng.normal(size=60)
        expected = upgrade.fit_ridge(x, y, np.ones(60)).predict(x)
        with patch('sklearn.linear_model._ridge.linalg.svd', side_effect=np.linalg.LinAlgError('test failure')):
            actual = upgrade.fit_ridge(x, y, np.ones(60)).predict(x)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
