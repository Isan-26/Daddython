"""Engineered network inputs per CV fold (and for the final fit), from the seed-527 LightGBM-imputed model.
Same folds / masks / models as neural_net_model.ipynb Part E. Runs in datathon-platform-env (LightGBM)."""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
FOLDER = '/Users/ethanis/Downloads/Track 1 Dataset/'
nb = json.load(open(FOLDER + 'neural_net_model.ipynb')); g = {}
for i in (2, 3, 5, 7, 10, 12, 14):
    s = ''.join(nb['cells'][i]['source']).replace("DATA_FOLDER = Path('.')", f"DATA_FOLDER = Path('{FOLDER}')")
    if i == 14:
        s = s.split('MODEL_NAMES')[0]
    exec(s, g)
train, test_inputs, NUM = g['train'], g['test_inputs'], g['NUMERIC_COLUMNS']
Model, network_inputs, mask = g['LightGBMImputedModel'], g['NeuralNetModel'].network_inputs, g['with_test_like_blanks']
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

def block(model, rows):
    filled = model.fill_blanks(rows)
    X = network_inputs(model, filled, rows)[:, :20]                     # numeric part (one-hots dropped; categories passed separately)
    flags = rows[NUM].isna().to_numpy(float)
    return np.column_stack([X, flags]).astype(np.float32)

out = {}
folds = list(KFold(5, shuffle=True, random_state=0).split(train))
with threadpool_limits(4):
    for f, (a, b) in enumerate(folds):
        t = time.time()
        fit_rows = train.iloc[a]
        check = train.iloc[b].drop(columns='energy_usage'); check_tb = mask(check, seed=f)
        m = Model(seed=527).fit(fit_rows, check_tb)
        out[f'fold{f}_fit'] = block(m, fit_rows); out[f'fold{f}_asis'] = block(m, check); out[f'fold{f}_tb'] = block(m, check_tb)
        print(f'fold {f} ({time.time() - t:.0f}s)', flush=True)
    m = Model(seed=527).fit(train, test_inputs)
    out['final_train'] = block(m, train); out['final_test'] = block(m, test_inputs)
np.savez(FOLDER + 'experiments/nn_features_folds.npz', **out)
print('saved experiments/nn_features_folds.npz; numeric columns:', out['final_train'].shape[1])
