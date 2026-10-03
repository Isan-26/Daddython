"""Train RealMLP seeds on all of train.csv (datathon-dl-env), export their weights to numpy, and check the numpy
forward pass (realmlp_numpy.py) against PyTorch. usage: python export_realmlp.py 0 1 2 ...
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
for name in ['lightning', 'lightning.pytorch', 'lightning.fabric', 'pytorch_lightning']:
    logging.getLogger(name).setLevel(logging.ERROR)
from pytabkit import RealMLP_TD_Regressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from realmlp_numpy import BUILDING_IDS, NUMERIC_COLUMNS, realmlp_inputs, realmlp_predict

CATEGORICAL_COLUMNS = ['building_id', 'building_type', 'day_of_week']
train = pd.read_csv(ROOT / 'train.csv')
test_inputs = pd.read_csv(ROOT / 'test.csv').drop(columns=['energy_usage'], errors='ignore')


class RealMLPFeatures:
    """Identical to deep_learning_model.ipynb: blanks -> building x hour median (fallback building median) + 'was blank' flags."""

    def fit(self, rows):
        self.by_hour = {c: rows.groupby(['building_id', 'hour'])[c].median() for c in NUMERIC_COLUMNS}
        self.by_building = {c: rows.groupby('building_id')[c].median() for c in NUMERIC_COLUMNS}
        return self

    def transform(self, rows):
        features = pd.DataFrame(index=rows.index)
        for column in CATEGORICAL_COLUMNS:
            features[column] = rows[column].astype(str)
        features['hour'] = rows['hour'].astype(float)
        features['month'] = rows['month'].astype(float)
        features['weekend'] = rows['day_of_week'].isin(['Saturday', 'Sunday']).astype(float)
        features['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
        lookup = pd.MultiIndex.from_arrays([rows['building_id'], rows['hour']])
        for column in NUMERIC_COLUMNS:
            fill = pd.Series(self.by_hour[column].reindex(lookup).to_numpy(), index=rows.index)
            fill = fill.fillna(rows['building_id'].map(self.by_building[column]))
            features[column] = rows[column].fillna(fill).astype(float)
            features[f'{column}_blank'] = rows[column].isna().astype(float)
        return features

    def as_numpy(self):
        """The medians as plain arrays (12 buildings x 24 hours, and 12 buildings) for realmlp_numpy.py."""
        out = {}
        for column in NUMERIC_COLUMNS:
            table = np.full((len(BUILDING_IDS), 24), np.nan)
            for (building, hour), value in self.by_hour[column].items():
                table[BUILDING_IDS.index(building), int(hour)] = value
            out[f'{column}_by_building_hour'] = table
            out[f'{column}_by_building'] = self.by_building[column].reindex(BUILDING_IDS).to_numpy(float)
        return out


def _t(x):
    return x.detach().cpu().numpy().astype(np.float64)


def extract_weights(model):
    """Read every weight of a fitted RealMLP_TD_Regressor, checking each layer is what realmlp_numpy.py assumes."""
    tm = model.alg_interface_.model
    names = lambda layers: [type(l).__name__ for l in layers]
    concat, mlp, normalize, clamp = list(tm.model.tfms)
    assert names([concat, mlp, normalize, clamp]) == ['ConcatParallelLayer', 'SequentialLayer', 'NormalizeOutputLayer', 'ClampLayer']
    numeric_branch, categorical_branch = [list(b.tfms) for b in concat.layers]
    numeric_pre, plr = list(numeric_branch[1].tfms), numeric_branch[2]
    assert names(numeric_pre) == ['RenameTensorLayer', 'BiasLayer', 'ScaleLayer', 'FunctionLayer']
    assert type(plr).__name__ == 'PLREmbeddingsLayerCosBias' and plr.plr_act_name == 'linear' and plr.plr_use_densenet
    categorical_pre = list(categorical_branch[1].tfms)
    assert names(categorical_pre) == ['EncodingLayer', 'RenameTensorLayer', 'BiasLayer', 'ScaleLayer', 'FunctionLayer', 'EncodingLayer']
    for f in (numeric_pre[3].f, categorical_pre[4].f):
        assert f.func.__name__ == 'smooth_clip_func' and f.keywords == {'max_abs_value': 3.0}
    static_layers = list(list(tm.creator.static_model.tfms)[0].tfms)[0].emb_layers
    assert names(static_layers) == ['IdentityLayer', 'SingleOneHotLayer', 'SingleOneHotLayer']
    assert all(l.use_missing_zero and l.onoff == (1.0, 0.0) for l in static_layers[1:])
    bias_layers = [numeric_pre[1], categorical_pre[2]]
    p = {
        'num_bias': _t(numeric_pre[1].bias).reshape(-1), 'num_scale': _t(numeric_pre[2].scale).reshape(-1),
        'plr_w1': _t(plr.weight_1)[0, :, 0, :], 'plr_b1': _t(plr.bias_1)[0, :, 0, :],
        'plr_w2': _t(plr.weight_2)[0], 'plr_b2': _t(plr.bias_2)[0, :, 0, :],
        'cat_bias': _t(categorical_pre[2].bias).reshape(-1), 'cat_scale': _t(categorical_pre[3].scale).reshape(-1),
        'emb': _t(categorical_pre[5].emb_layers[0].emb)[0],
        'type_cats': np.array(static_layers[1].cat_size - 1), 'day_cats': np.array(static_layers[2].cat_size - 1),
    }
    blocks = [list(b.tfms) for b in mlp.tfms]
    assert names(blocks[0]) == ['ScaleLayer', 'WeightLayer', 'BiasLayer', 'ParametricActivationLayer', 'DropoutLayer']
    assert names(blocks[1]) == names(blocks[2]) == ['WeightLayer', 'BiasLayer', 'ParametricActivationLayer', 'DropoutLayer']
    assert names(blocks[3]) == ['WeightLayer', 'BiasLayer']
    p['scale0'] = _t(blocks[0][0].scale).reshape(-1)
    for k, layers in zip((1, 2, 3, 4), [blocks[0][1:], blocks[1], blocks[2], blocks[3]]):
        weight, bias = layers[0], layers[1]
        p[f'w{k}'], p[f'f{k}'], p[f'b{k}'] = _t(weight.weight)[0], np.array(float(weight.factor)), _t(bias.bias).reshape(-1)
        bias_layers.append(bias)
        if k < 4:
            assert layers[2].f.__name__ == 'mish'
            p[f'a{k}'] = _t(layers[2].weight).reshape(-1)
    assert all(float(b.factor) == 1.0 for b in bias_layers)
    p['out_mean'], p['out_std'] = _t(normalize.mean).reshape(()), _t(normalize.std).reshape(())
    p['out_low'], p['out_high'] = _t(clamp.low).reshape(()), _t(clamp.high).reshape(())
    return p


if __name__ == '__main__':
    out_dir = HERE / 'realmlp_parts'
    out_dir.mkdir(exist_ok=True)
    features = RealMLPFeatures().fit(train)
    medians = features.as_numpy()
    np.savez(out_dir / 'medians.npz', **medians)
    all_rows = pd.concat([train.drop(columns='energy_usage'), test_inputs], ignore_index=True)
    for seed in map(int, sys.argv[1:]):
        model = RealMLP_TD_Regressor(device='cpu', random_state=seed, n_threads=2, verbosity=0)
        model.fit(features.transform(train), train['energy_usage'].to_numpy(), cat_col_names=CATEGORICAL_COLUMNS)
        torch_prediction = model.predict(features.transform(all_rows))       # also restores the best-epoch weights
        weights = extract_weights(model)
        x, codes = realmlp_inputs(all_rows, medians)
        numpy_prediction = realmlp_predict(weights, x, codes)
        saved = ROOT / 'experiments' / f'realmlp_test_seed{seed}.npy'
        vs_saved = np.abs(torch_prediction[len(train):] - np.load(saved)).max() if saved.exists() else float('nan')
        np.savez(out_dir / f'seed{seed}.npz', **weights)
        print(f'seed {seed}: numpy vs PyTorch largest difference {np.abs(numpy_prediction - torch_prediction).max():.2e} '
              f'(11,000 rows) | PyTorch vs earlier saved test predictions {vs_saved:.2e}', flush=True)
