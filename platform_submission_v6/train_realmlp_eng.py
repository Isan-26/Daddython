"""Train the final 'engineered-input' RealMLP networks on all of train.csv (datathon-dl-env), export their weights
to numpy and check the numpy forward pass against PyTorch. usage: python train_realmlp_eng.py <variant> <seed> ...
variant 'eng': target = energy_usage; 'eng_delta': target = energy_usage - filled previous_usage.
Inputs: the 20 numeric network inputs of neural_net_model.ipynb (seed-527 LightGBM imputers) + 4 blank flags,
from ../experiments/nn_features_folds.npz (made in datathon-platform-env), plus building / type / weekday categories."""
import logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for name in ['lightning', 'lightning.pytorch', 'lightning.fabric', 'pytorch_lightning']:
    logging.getLogger(name).setLevel(logging.ERROR)
from pytabkit import RealMLP_TD_Regressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'platform_submission_v2'))
from export_realmlp import extract_weights                 # the verified exporter from platform_submission_v2
from realmlp_numpy import realmlp_predict, BUILDING_IDS, BUILDING_TYPES, DAYS_ALPHABETICAL

ENG_COLUMNS = ['temperature', 'humidity', 'occupancy', 'previous_usage', 'expected_previous', 'expected_occupancy',
               'occupancy_gap', 'previous_gap', 'previous_ratio', 'heat', 'rain', 'weekend',
               'hour_sin_1', 'hour_cos_1', 'hour_sin_2', 'hour_cos_2', 'hour_sin_3', 'hour_cos_3', 'month_sin', 'month_cos',
               'temperature_blank', 'humidity_blank', 'occupancy_blank', 'previous_usage_blank']
CATS = ['building_id', 'building_type', 'day_of_week']
train, test = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
feats = np.load(ROOT / 'experiments' / 'nn_features_folds.npz')


def frame(matrix, rows):
    df = pd.DataFrame(matrix.astype(float), columns=ENG_COLUMNS, index=rows.index)
    for c in CATS:
        df[c] = rows[c].astype(str).to_numpy()
    return df


def codes(rows):
    return np.column_stack([rows.building_id.map({b: i + 1 for i, b in enumerate(BUILDING_IDS)}).fillna(0).to_numpy(int),
                            rows.building_type.map({t: i + 1 for i, t in enumerate(BUILDING_TYPES)}).fillna(0).to_numpy(int),
                            rows.day_of_week.map({d: i + 1 for i, d in enumerate(DAYS_ALPHABETICAL)}).fillna(0).to_numpy(int)])


if __name__ == '__main__':
    variant = sys.argv[1]
    Xtr, Xte = frame(feats['final_train'], train), frame(feats['final_test'], test)
    delta = variant == 'eng_delta'
    target = train.energy_usage.to_numpy() - (Xtr.previous_usage.to_numpy() if delta else 0)
    x_test = feats['final_test'].astype(np.float32).astype(np.float64)
    for seed in map(int, sys.argv[2:]):
        model = RealMLP_TD_Regressor(device='cpu', random_state=seed, n_threads=2, verbosity=0)
        model.fit(Xtr, target, cat_col_names=CATS)
        torch_pred = model.predict(Xte)                              # restores the best-epoch weights
        weights = extract_weights(model)
        numpy_pred = realmlp_predict(weights, x_test, codes(test))
        weights['add_previous_usage'] = np.array(1.0 if delta else 0.0)
        np.savez(HERE / 'realmlp_eng_parts' / f'{variant}_seed{seed}.npz', **weights)
        np.save(HERE / 'realmlp_eng_parts' / f'{variant}_seed{seed}_torch_test.npy', torch_pred + (Xte.previous_usage.to_numpy() if delta else 0))
        print(f'{variant} seed {seed}: numpy vs PyTorch largest difference {np.abs(numpy_pred - torch_pred).max():.2e}', flush=True)
