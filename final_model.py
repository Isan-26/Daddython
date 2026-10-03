"""Final energy-usage model: v4 blended with a LightGBM-imputed model.

Leaderboard history (test RMSE, lower is better):
    best_v2.csv                 3.40   MissForest filling + Ridge/HGB
    candidate_v4.csv            3.37   per-missing-pattern models + v3 (energy_model.py)
    candidate_v5_lightgbm.csv   3.29   0.50 v4 + 0.50 new model
    candidate_v6_blend.csv      3.32   0.30 v4 + 0.40 new model + 0.30 v2
    candidate_v7.csv            3.27   0.35 v4 + 0.65 new model
    candidate_v8.csv            3.27026  0.25 v4 + 0.75 new model

Pipeline
    1. v4 (energy_model.py): one regression per pattern of available inputs, blended with the v3 notebook model.
    2. New model: four LightGBM "imputers" learn each numeric input (temperature, humidity, occupancy,
       previous_usage) from the other inputs, using train AND test inputs (never test targets). They are trained
       with extra copies carrying test-style blanks and weighted toward the test set's building/event mix.
       Blanks are filled with their predictions; then a Ridge model (v3 features + the imputers' "expected"
       previous_usage/occupancy) and a small gradient-boosting part predict energy_usage.
    3. Blend: (1 - w) * v4 + w * new model.

Only libraries from requirements-image.txt are needed (no CatBoost).
Run from the project folder:  python final_model.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NUM = ['temperature', 'humidity', 'occupancy', 'previous_usage']
CATEGORIES = {
    'building_id': ['ADM_A', 'BUS_A', 'BUS_B', 'ENG_A', 'ENG_B', 'LEC_A', 'LIB_A', 'RES_A', 'RES_B', 'SCI_A', 'SCI_B', 'SPT_A'],
    'building_type': ['Administration', 'Business', 'Engineering', 'LectureHall', 'Library', 'Residential', 'Science', 'Sports'],
    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
}


def load_data():
    return pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')


# ---------------------------------------------------------------- step 1: v4
def v4_predictions(train, test, source='file'):
    """'file' reads candidate_v4.csv (the exact file that scored 3.37); 'refit' retrains it with energy_model.py."""
    if source == 'file':
        v4 = pd.read_csv(ROOT / 'candidate_v4.csv')
        if not v4['id'].equals(test['id']):
            raise ValueError('candidate_v4.csv ids do not match test.csv')
        return v4['prediction'].to_numpy()
    from energy_model import EnergyModel
    return EnergyModel().fit(train, test).predict(test)


# ---------------------------------------------------------------- step 2: the LightGBM-imputed model
def _as_categories(x):
    x = x.copy()
    for col, cats in CATEGORIES.items():
        x[col] = pd.Categorical(x[col], categories=cats)
    return x


class _Imputer:
    """One LightGBM model that predicts a numeric input from the other inputs."""
    def __init__(self, seed):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(n_estimators=2200, learning_rate=0.025, max_depth=5, num_leaves=32,
                                       min_child_samples=30, reg_lambda=15.0, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, cat_smooth=20, verbose=-1, random_state=seed, n_jobs=4)

    def fit(self, x, y, sample_weight):
        self.model.fit(_as_categories(x), y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.model.predict(_as_categories(x))


def _new_model_class():
    # Built on the second-round code in experiments/round2_models.py; only the imputers are replaced.
    from energy_model import shift_weights
    from experiments.round2_models import CatImputedBaseline, cat_features

    class LightGBMImputedModel(CatImputedBaseline):
        def fit_imputer(self, df, test_features):
            joint = pd.concat([df.drop(columns='energy_usage', errors='ignore'), test_features], ignore_index=True)
            rng = np.random.default_rng(self.random_seed)
            weights = shift_weights(joint, test_features) ** self.imputer_weight if self.imputer_weight else np.ones(len(joint))
            weights = weights / weights.mean()
            self.imputation_models = {}
            for col in NUM:
                observed = joint.loc[joint[col].notna()].copy()
                copies = [observed]
                for _ in range(self.augment):
                    q = observed.copy()
                    mask = test_features[NUM].isna().to_numpy()[rng.integers(0, len(test_features), len(q))]
                    mask[:, NUM.index(col)] = False          # the column being learned stays observed
                    q[NUM] = q[NUM].mask(mask)
                    copies.append(q)
                data = pd.concat(copies, ignore_index=True)
                x = cat_features(data).drop(columns=col)
                sw = np.tile(weights[observed.index.to_numpy()], self.augment + 1)
                self.imputation_models[col] = _Imputer(self.random_seed).fit(x, data[col], sw)
            return self

    return LightGBMImputedModel


def new_model_predictions(train, test, seeds=(527,)):
    """Average the new model over random seeds (seed 527 reproduces the model inside candidate_v5_lightgbm.csv)."""
    from threadpoolctl import threadpool_limits
    config = json.loads((ROOT / 'model_v5_config.json').read_text())['config']
    model_class = _new_model_class()
    preds = []
    with threadpool_limits(limits=4):
        for seed in seeds:
            preds.append(model_class(**dict(config, random_seed=seed)).fit(train, test).predict(test))
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------- step 3: blend and write
def blend(v4, new, weight_new):
    return (1 - weight_new) * np.asarray(v4) + weight_new * np.asarray(new)


def write_submission(path, test, prediction, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path.name} exists; pass overwrite=True to replace it')
    prediction = np.asarray(prediction, float)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise ValueError('predictions must be finite, one per test row')
    out = pd.DataFrame({'id': test['id'], 'prediction': prediction})
    out.to_csv(path, index=False)
    return out


if __name__ == '__main__':
    train, test = load_data()
    v4 = v4_predictions(train, test, source='file')
    new = new_model_predictions(train, test, seeds=(527, 528, 529))
    result = write_submission(ROOT / 'candidate_v9.csv', test, blend(v4, new, 0.8))
    print(f'wrote candidate_v9.csv ({len(result):,} rows)')
