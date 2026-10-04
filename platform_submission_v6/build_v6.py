"""Build platform submission v6 = v4 + 15% engineered-input RealMLP ensemble (datathon-platform-env).
v4's models are reused unchanged; the 10 new networks were trained by train_realmlp_eng.py (datathon-dl-env)."""
import json, sys, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from eng_runtime import *
warnings.filterwarnings('ignore')
ENG_WEIGHT = 0.15          # nested 5-fold CV chose 0.14-0.18 per fold; not tuned on the leaderboard
VARIANTS, SEEDS = ['eng', 'eng_delta'], [0, 1, 2, 3, 4]

start = time.time()
test = pd.read_csv(ROOT / 'test.csv')
rows = test.drop(columns=['id', 'energy_usage'], errors='ignore')
bundle = joblib.load(ROOT / 'platform_submission_v4' / 'model.pkl')
nets = []
for variant in VARIANTS:
    for seed in SEEDS:
        net = dict(np.load(HERE / 'realmlp_eng_parts' / f'{variant}_seed{seed}.npz'))
        net['variant'], net['seed'] = variant, seed
        nets.append(net)
bundle['realmlp_eng'] = {'nets': nets, 'inputs': '20 neural_net_model.ipynb network inputs (seed-527 imputers) + 4 blank flags'}
bundle['versions']['v6_eng'] = {'alpha': 1., 'samples': 64, 'seed': 2026, 'base': 'v4_uncertainty', 'eng_weight': ENG_WEIGHT}

with threadpool_limits(limits=1):
    x, _ = eng_inputs(bundle['v2'], rows)
    saved = np.load(ROOT / 'experiments' / 'nn_features_folds.npz')['final_test'].astype(np.float64)
    feature_diff = float(np.abs(x - saved).max())
    eng = predict_realmlp_eng(bundle, rows)
    torch_avg = np.mean([np.load(HERE / 'realmlp_eng_parts' / f'{v}_seed{s}_torch_test.npy') for v in VARIANTS for s in SEEDS], axis=0)
    eng_diff = float(np.abs(eng - torch_avg).max())
    v6 = predict_v6(bundle, rows, 'v6_eng')
    v4_again = predict_v6(bundle, rows, 'v4_uncertainty')
    v2_again = predict_v6(bundle, rows, 'v2_fallback')
v4_csv = pd.read_csv(ROOT / 'platform_submission_v4' / 'candidate_v4_uncertainty.csv')
v2_csv = pd.read_csv(ROOT / 'platform_submission_v2' / 'platform_dl13_prevfix.csv')
assert v4_csv.id.equals(test.id) and v2_csv.id.equals(test.id)
v4_diff = float(np.abs(np.round(v4_again, 6) - v4_csv.prediction).max())
v2_diff = float(np.abs(np.round(v2_again, 6) - v2_csv.prediction).max())
assert feature_diff < 1e-4 and eng_diff < 1e-3 and v4_diff == 0 and v2_diff == 0, (feature_diff, eng_diff, v4_diff, v2_diff)

pd.DataFrame({'id': test.id, 'prediction': np.round(v6, 6)}).to_csv(HERE / 'candidate_v6_eng.csv', index=False)
joblib.dump(bundle, HERE / 'model.pkl', compress=3)

runtime = '\n'.join(p.read_text().replace('from frozen_v2 import *\n', '').replace('from uncertainty_runtime import *\n', '')
                    for p in [HERE / 'frozen_v2.py', HERE / 'uncertainty_runtime.py', HERE / 'eng_runtime.py'])
final = """import os
import warnings
import joblib
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore')
VERSION = 'v6_eng'  # alternatives: v4_uncertainty (exact v4), v2_fallback (exact v2)
INPUT_PATH = os.environ.get('DATATHON_INPUT_PATH', 'test.csv')
OUTPUT_PATH = os.environ.get('DATATHON_OUTPUT_PATH', 'predictions.csv')
bundle = joblib.load('model.pkl')
rows = pd.read_csv(INPUT_PATH).drop(columns=['id', 'energy_usage'], errors='ignore')
with threadpool_limits(limits=1):
    prediction = np.round(predict_v6(bundle, rows, VERSION), 6)
assert len(prediction) == len(rows) and np.isfinite(prediction).all()
pd.DataFrame({'prediction': prediction}).to_csv(OUTPUT_PATH, index=False)
print(f'wrote {len(rows):,} predictions using {VERSION} to {OUTPUT_PATH}')
"""
cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Platform submission v6: v4 + engineered-input RealMLP ensemble (15%)\n',
            'Upload this notebook together with `model.pkl`. Only pre-installed libraries are used (numpy forward pass for the networks).\n',
            'Default `v6_eng` = 0.85 x v4 + 0.15 x average of 10 RealMLP networks fed the pipeline\'s engineered inputs and blank flags.\n',
            '`v4_uncertainty` reproduces v4 exactly; `v2_fallback` reproduces v2 exactly.\n']},
         {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': runtime.splitlines(True)},
         {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': final.splitlines(True)}]
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
(HERE / 'prediction_notebook.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')

change = np.round(v6, 6) - v4_csv.prediction.to_numpy()
blank = rows[NUMERIC_COLUMNS].isna().any(axis=1).to_numpy()
summary = {'eng_weight': ENG_WEIGHT, 'networks': len(nets), 'feature_diff_vs_training': feature_diff,
           'numpy_vs_pytorch_ensemble_max_diff': eng_diff, 'v4_reproduced_max_diff': v4_diff, 'v2_reproduced_max_diff': v2_diff,
           'rms_change_vs_v4_complete': float(np.sqrt(np.mean(change[~blank] ** 2))), 'rms_change_vs_v4_blank': float(np.sqrt(np.mean(change[blank] ** 2))),
           'mean_change_vs_v4_complete': float(change[~blank].mean()), 'mean_change_vs_v4_blank': float(change[blank].mean()),
           'max_abs_change_vs_v4': float(np.abs(change).max()), 'model_pkl_mb': (HERE / 'model.pkl').stat().st_size / 1e6,
           'build_seconds': time.time() - start}
(HERE / 'v6_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary, indent=1))
