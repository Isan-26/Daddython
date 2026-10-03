"""Clean model: learns ONLY from train.csv. No test inputs, no test-based weights, no leaderboard tuning.

Same two ingredients as the best submissions, with every use of test.csv removed:
    A. "v4" = 0.25 x v3 notebook model + 0.75 x per-missing-pattern models
       - imputer for the v3 part fitted on train rows only (as before)
       - pattern models WITHOUT weighting toward the test building/event mix
    B. LightGBM-imputed model
       - the four imputers learn from train inputs only
       - extra training copies get GENERIC random blanks (each value blanked with 10% chance),
         not blank patterns copied from test
       - no weighting toward the test mix
    Blend: (1 - w) * A + w * B, with w chosen by 5-fold cross-validation on train.csv
           (scored on validation rows as they are and with the same generic random blanks).

test.csv is only read at the very end, to make predictions.
Run from the project folder:  python clean_model.py
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
NUM = ['temperature', 'humidity', 'occupancy', 'previous_usage']
MASK_RATE = 0.10          # generic stress level for blanks; not taken from test.csv
SEED = 0


def random_blanks(df, rate, seed, keep=None):
    """Blank each numeric value independently with probability `rate` (optionally never blank column `keep`)."""
    out = df.copy()
    mask = np.random.default_rng(seed).random((len(out), len(NUM))) < rate
    if keep is not None:
        mask[:, NUM.index(keep)] = False
    out[NUM] = out[NUM].mask(mask)
    return out


# ------------------------------------------------------------------ ingredient A: v4 without test weighting
def fit_predict_A(train, frames):
    from energy_model import NotebookBaseline, PatternModel
    cfg = json.loads((ROOT / 'model_config.json').read_text())['pattern_config']
    cfg = dict(cfg, weight_power=0)                                   # no weighting toward the test mix
    base = NotebookBaseline().fit(train)                              # its imputer sees train rows only
    experts = PatternModel(**cfg).fit(train, test_features=None)
    return [0.25 * base.predict(f) + 0.75 * experts.predict(f) for f in frames]


# ------------------------------------------------------------------ ingredient B: LightGBM-imputed model, train only
def _clean_model_class():
    from final_model import _Imputer
    from experiments.round2_models import CatImputedBaseline, cat_features

    class CleanImputedModel(CatImputedBaseline):
        def fit_imputer(self, df, test_features=None):
            data_all = df.drop(columns='energy_usage', errors='ignore').reset_index(drop=True)
            self.imputation_models = {}
            for k, col in enumerate(NUM):
                observed = data_all.loc[data_all[col].notna()]
                copies = [observed] + [random_blanks(observed, MASK_RATE, self.random_seed + 100 * k + j, keep=col)
                                       for j in range(self.augment)]
                data = pd.concat(copies, ignore_index=True)
                x = cat_features(data).drop(columns=col)
                self.imputation_models[col] = _Imputer(self.random_seed).fit(x, data[col], np.ones(len(data)))
            return self

    return CleanImputedModel


def fit_predict_B(train, frames, seed=527):
    cfg = json.loads((ROOT / 'model_v5_config.json').read_text())['config']
    cfg = dict(cfg, imputer_weight=0, random_seed=seed)
    model = _clean_model_class()(**cfg).fit(train, None)
    return [model.predict(f) for f in frames]


# ------------------------------------------------------------------ choose the blend weight by cross-validation
def _fold(train, tr_idx, va_idx, k):
    from threadpoolctl import threadpool_limits
    t, v = train.iloc[tr_idx], train.iloc[va_idx]
    vm = random_blanks(v, MASK_RATE, 1000 + k)
    with threadpool_limits(limits=3):
        a = fit_predict_A(t, [v, vm])
        b = fit_predict_B(t, [v, vm])
    return va_idx, a, b


def cross_validate(train, n_jobs=5):
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(train))
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(train, tr, va, k) for k, (tr, va) in enumerate(folds))
    oof = {key: np.zeros(len(train)) for key in ['A', 'A_blank', 'B', 'B_blank']}
    for va, (a, a_blank), (b, b_blank) in res:
        oof['A'][va], oof['A_blank'][va], oof['B'][va], oof['B_blank'][va] = a, a_blank, b, b_blank
    return oof


def choose_weight(train, oof):
    y = train['energy_usage'].to_numpy()
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        r1 = rmse((1 - w) * oof['A'] + w * oof['B'])
        r2 = rmse((1 - w) * oof['A_blank'] + w * oof['B_blank'])
        rows.append({'weight on B': w, 'CV RMSE': r1, 'CV RMSE with random blanks': r2, 'average': (r1 + r2) / 2})
    table = pd.DataFrame(rows)
    best = table.loc[table['average'].idxmin()]
    return float(best['weight on B']), table


if __name__ == '__main__':
    import time
    t0 = time.time()
    train = pd.read_csv(ROOT / 'train.csv')                 # everything below the next line uses train only
    oof = cross_validate(train)
    w, table = choose_weight(train, oof)
    print(table.round(4).to_string(index=False))
    print(f'\nchosen weight on B (by cross-validation): {w:.2f}   [{time.time() - t0:.0f}s]')

    test = pd.read_csv(ROOT / 'test.csv')                   # used only to predict
    a, = fit_predict_A(train, [test])
    b, = fit_predict_B(train, [test])
    pred = (1 - w) * a + w * b
    out = ROOT / 'candidate_clean.csv'
    if out.exists():
        raise SystemExit('candidate_clean.csv exists; not overwriting')
    pd.DataFrame({'id': test['id'], 'prediction': pred}).to_csv(out, index=False)
    np.save(ROOT / 'experiments' / 'clean_components.npy', np.vstack([a, b]))
    print(f'wrote {out.name} ({len(pred):,} rows)   [{time.time() - t0:.0f}s total]')
