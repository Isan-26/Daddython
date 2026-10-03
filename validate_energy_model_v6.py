"""Validate fixed third-round settings against V5 on the reserved partition."""
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from energy_model import ROOT, NUM, masked_view, metrics, shift_weights
from energy_model_v6 import EnergyModelV6


def validate():
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    dev, holdout = train_test_split(train, test_size=.2, random_state=2028,
                                   stratify=train.building_id)
    views = [masked_view(holdout, test, 51000 + k) for k in range(12)]
    full = pd.concat([holdout] + views)
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter('ignore')
        print('Fitting fixed V6 and reproduced V5 on the development partition.', flush=True)
        model = EnergyModelV6().fit(dev, test)
        old = model.baseline.predict(full)
        new = model.predict(full, baseline_prediction=old)
    n, m = len(holdout), len(views[0])
    y = full.energy_usage.to_numpy()
    w = shift_weights(holdout.loc[views[0].index], test)
    counts = full[NUM].isna().sum(axis=1).to_numpy()
    assert np.array_equal(old[counts == 0], new[counts == 0])

    def score(prediction):
        error = (prediction[n:] - y[n:]) ** 2
        return {
            'ordinary': metrics(holdout, prediction[:n]),
            'test_masks_rmse': float(np.sqrt(error.mean())),
            'weighted_rmse': float(np.sqrt(np.average(error, weights=np.tile(w, len(views))))),
            'masked_by_missing_count': {
                str(k): float(np.sqrt(error[counts[n:] == k].mean()))
                for k in np.unique(counts[n:])
            },
        }

    result = {
        'baseline_file': 'candidate_v5.csv',
        'baseline_competition_rmse': 3.31,
        'split_seed': 2028,
        'holdout_rows': n,
        'complete_holdout_rows': m,
        'mask_trials': len(views),
        'note': 'Reserved from third-round selection. These labelled rows may have been used in earlier rounds. Weighted scores approximate test building/event frequencies using input values only.',
        'config': model.config,
        'baseline': score(old),
        'candidate': score(new),
    }
    rng = np.random.default_rng(91)
    eo = ((old[n:] - y[n:]) ** 2).reshape(len(views), m).mean(axis=0)
    en = ((new[n:] - y[n:]) ** 2).reshape(len(views), m).mean(axis=0)
    deltas, weighted = [], []
    for _ in range(1000):
        ix = rng.integers(0, m, m)
        deltas.append(np.sqrt(eo[ix].mean()) - np.sqrt(en[ix].mean()))
        weighted.append(np.sqrt(np.average(eo[ix], weights=w[ix])) -
                        np.sqrt(np.average(en[ix], weights=w[ix])))
    result['bootstrap_gain_intervals'] = {
        'positive_favors_v6': True,
        'masked_95pct': np.quantile(deltas, [.025, .975]).tolist(),
        'weighted_95pct': np.quantile(weighted, [.025, .975]).tolist(),
    }
    result['data_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in ['train.csv', 'test.csv']}
    np.savez(ROOT / 'experiments/round3/confirmation_predictions.npz',
             baseline=old, candidate=new, y=y, n_raw=n, weights=w, indices=full.index.to_numpy())
    (ROOT / 'validation_v6_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    validate()
