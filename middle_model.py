"""Middle model: uses test INPUTS (never answers), but nothing is tuned on leaderboard scores.

Ingredients (both use unlabelled test inputs, as in the best submissions):
    A. v4 (energy_model.py): per-missing-pattern models + v3, training rows weighted toward the test mix.
    B. LightGBM-imputed model (final_model.py): imputers learn from train + test inputs, with test-style blank
       patterns and test-mix weighting; averaged over 3 random seeds to reduce noise.

Blend weight: chosen by 5-fold cross-validation on train.csv only. Validation rows are scored
    (1) with blank patterns copied from real test rows (applied to complete validation rows, 3 draws), and
    (2) the same, weighted toward the test set's building/event mix.
The weight with the best average of the two is used. No leaderboard score is used anywhere.

Run from the project folder:  python middle_model.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
SEEDS = (527, 528, 529)
MASK_DRAWS = 3
CV_FILE = ROOT / 'experiments' / 'middle_cv.npz'


def _fold(train, test, tr_idx, va_idx, k):
    from threadpoolctl import threadpool_limits
    from energy_model import EnergyModel, masked_view
    import final_model as F
    t, v = train.iloc[tr_idx], train.iloc[va_idx]
    views = [masked_view(v, test, 5000 + 10 * k + j) for j in range(MASK_DRAWS)]   # complete rows, test blank patterns
    frames = views
    with threadpool_limits(limits=3):
        a_model = EnergyModel().fit(t, test)
        a = [a_model.predict(f) for f in frames]
        model_class = F._new_model_class()
        config = __import__('json').loads((ROOT / 'model_v5_config.json').read_text())['config']
        b_runs = []
        for s in SEEDS:
            m = model_class(**dict(config, random_seed=s)).fit(t, test)
            b_runs.append([m.predict(f) for f in frames])
        b = [np.mean([run[i] for run in b_runs], axis=0) for i in range(len(frames))]
    idx = views[0].index.to_numpy()
    return idx, a, b


def cross_validate(train, test, n_jobs=5):
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    folds = list(KFold(5, shuffle=True, random_state=0).split(train))
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(train, test, tr, va, k) for k, (tr, va) in enumerate(folds))
    idx = np.concatenate([r[0] for r in res])
    A = np.concatenate([np.vstack(r[1]) for r in res], axis=1)      # shape (draws, rows)
    B = np.concatenate([np.vstack(r[2]) for r in res], axis=1)
    np.savez(CV_FILE, idx=idx, A=A, B=B)
    return idx, A, B


def choose_weight(train, test, idx, A, B):
    from energy_model import shift_weights
    rows_ = train.loc[idx]
    y = rows_['energy_usage'].to_numpy()
    w_mix = shift_weights(rows_, test)                               # test-mix weights from test inputs only
    table = []
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        p = (1 - w) * A + w * B
        e2 = (p - y) ** 2
        blanks = float(np.sqrt(e2.mean()))
        weighted = float(np.sqrt(np.average(e2, weights=np.tile(w_mix, (len(A), 1)))))
        table.append({'weight on B': w, 'test-like blanks': blanks, 'test-like blanks, test-mix weighted': weighted,
                      'average': (blanks + weighted) / 2})
    table = pd.DataFrame(table)
    return float(table.loc[table['average'].idxmin(), 'weight on B']), table


def main():
    import time
    import final_model as F
    t0 = time.time()
    train, test = F.load_data()
    if CV_FILE.exists():
        d = np.load(CV_FILE)
        idx, A, B = d['idx'], d['A'], d['B']
    else:
        idx, A, B = cross_validate(train, test)
    w, table = choose_weight(train, test, idx, A, B)
    print(table.round(4).to_string(index=False))
    print(f'\nchosen weight on B (by cross-validation, no leaderboard): {w:.2f}   [{time.time() - t0:.0f}s]')

    v4 = F.v4_predictions(train, test, source='file')               # the exact v4 file
    new = F.new_model_predictions(train, test, seeds=SEEDS)
    pred = F.blend(v4, new, w)
    out = ROOT / 'candidate_middle.csv'
    F.write_submission(out, test, pred)
    print(f'wrote {out.name}   [{time.time() - t0:.0f}s total]')
    return w, table


if __name__ == '__main__':
    main()
