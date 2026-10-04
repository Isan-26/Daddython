"""RealMLP variants on the standard 5 folds. usage: python realmlp_variants.py <variant> <seed>
variants: delta (current inputs, target = change from previous_usage), eng (engineered inputs), eng_delta (both)."""
import sys, time, warnings, logging
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for n in ['lightning', 'lightning.pytorch', 'lightning.fabric', 'pytorch_lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
import dl_cv
from pytabkit import RealMLP_TD_Regressor
from sklearn.model_selection import KFold

ENG_COLUMNS = ['temperature', 'humidity', 'occupancy', 'previous_usage', 'expected_previous', 'expected_occupancy',
               'occupancy_gap', 'previous_gap', 'previous_ratio', 'heat', 'rain', 'weekend',
               'hour_sin_1', 'hour_cos_1', 'hour_sin_2', 'hour_cos_2', 'hour_sin_3', 'hour_cos_3', 'month_sin', 'month_cos',
               'temperature_blank', 'humidity_blank', 'occupancy_blank', 'previous_usage_blank']
CATS = dl_cv.CATEGORICAL
variant, seed = sys.argv[1], int(sys.argv[2])
train, y = dl_cv.train, dl_cv.truth
feats = np.load(dl_cv.FOLDER + 'experiments/nn_features_folds.npz')


def eng_frame(matrix, rows):
    df = pd.DataFrame(matrix.astype(float), columns=ENG_COLUMNS, index=rows.index)
    for c in CATS:
        df[c] = rows[c].astype(str).to_numpy()
    return df


oof = {'as_is': np.zeros(len(y)), 'test_blanks': np.zeros(len(y))}
for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train)):
    t = time.time()
    fit_rows, check = train.iloc[a], train.iloc[b].drop(columns='energy_usage')
    check_tb = dl_cv.with_test_like_blanks(check, seed=f)
    if variant == 'delta':
        filler = dl_cv.Filler().fit(fit_rows)
        Xa, Xb, Xt = filler.features(fit_rows), filler.features(check), filler.features(check_tb)
        prev = lambda X: X['previous_usage'].to_numpy()
    else:
        Xa = eng_frame(feats[f'fold{f}_fit'], fit_rows)
        Xb, Xt = eng_frame(feats[f'fold{f}_asis'], check), eng_frame(feats[f'fold{f}_tb'], check_tb)
        prev = lambda X: X['previous_usage'].to_numpy()
    use_delta = variant.endswith('delta')
    target = y[a] - prev(Xa) if use_delta else y[a]
    model = RealMLP_TD_Regressor(device='cpu', random_state=seed, n_threads=3, verbosity=0)
    model.fit(Xa, target, cat_col_names=CATS)
    oof['as_is'][b] = model.predict(Xb) + (prev(Xb) if use_delta else 0)
    oof['test_blanks'][b] = model.predict(Xt) + (prev(Xt) if use_delta else 0)
    print(f'{variant} seed {seed} fold {f} ({time.time() - t:.0f}s)', flush=True)
np.savez(dl_cv.FOLDER + f'experiments/dl_cv_rmlp_{variant}_seed{seed}.npz', **oof)
print(f'{variant} seed {seed} saved', flush=True)
