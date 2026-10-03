"""V6: blend conditional imputers into the confirmed V5 for incomplete rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from energy_model import ROOT, NUM
from energy_model_v5 import EnergyModelV5
from experiments.round3_models import ConditionalImputedEnergy


class EnergyModelV6:
    def __init__(self, config=None):
        self.config = config or json.loads((ROOT / 'model_v6_config.json').read_text())

    def fit(self, train, test_features, fit_baseline=True):
        inputs = test_features.drop(columns='energy_usage', errors='ignore')
        self.new_model = ConditionalImputedEnergy(**self.config['config']).fit(train, inputs)
        self.baseline = EnergyModelV5().fit(train, inputs) if fit_baseline else None
        return self

    def predict(self, df, baseline_prediction=None):
        if baseline_prediction is None:
            if self.baseline is None:
                raise ValueError('Supply baseline predictions or fit the baseline.')
            baseline_prediction = self.baseline.predict(df)
        baseline_prediction = np.asarray(baseline_prediction, dtype=float)
        if baseline_prediction.shape != (len(df),):
            raise ValueError('Baseline predictions must have one value per input row.')
        if not np.isfinite(baseline_prediction).all():
            raise ValueError('Baseline contains non-finite predictions.')
        counts = df[NUM].isna().sum(axis=1).to_numpy()
        weights = np.asarray(self.config['weights_by_missing_count'])[np.minimum(counts, 2)]
        prediction = baseline_prediction.copy()
        ix = np.flatnonzero(weights)
        if len(ix):
            new = self.new_model.predict(df.iloc[ix])
            prediction[ix] += weights[ix] * (new - baseline_prediction[ix])
        return prediction


def write_submission(output_path=ROOT / 'candidate_v6.csv'):
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    config = json.loads((ROOT / 'model_v6_config.json').read_text())
    baseline_path = ROOT / config['baseline_file']
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    if baseline_hash != config['baseline_sha256']:
        raise ValueError('The confirmed V5 file changed; verify it before blending.')
    baseline = pd.read_csv(baseline_path)
    if not baseline.id.equals(test.id):
        raise ValueError('Baseline IDs differ from test IDs.')
    with threadpool_limits(limits=4):
        model = EnergyModelV6(config).fit(train, test, fit_baseline=False)
        prediction = model.predict(test, baseline_prediction=baseline.prediction)
        # Meaningful checks: preserve observations and handle repeated validation indexes.
        filled = model.new_model.fill(test)
        for col in NUM:
            observed = test[col].notna()
            if not np.array_equal(filled.loc[observed, col], test.loc[observed, col]):
                raise AssertionError(f'Imputation changed observed {col}.')
        if not np.isfinite(filled[NUM].to_numpy()).all():
            raise AssertionError('Imputation left missing or non-finite numeric inputs.')
        incomplete = test.loc[test[NUM].isna().any(axis=1)].iloc[:5]
        repeated = pd.concat([incomplete, incomplete])
        repeated_prediction = model.new_model.predict(repeated)
        np.testing.assert_allclose(repeated_prediction[:5], repeated_prediction[5:], rtol=0, atol=1e-10)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction.')
    complete = test[NUM].notna().all(axis=1).to_numpy()
    if not np.array_equal(prediction[complete], baseline.prediction.to_numpy()[complete]):
        raise AssertionError('Complete-row predictions must equal the confirmed V5.')
    output_path = Path(output_path)
    result = pd.DataFrame({'id': test.id, 'prediction': prediction})
    result.to_csv(output_path, index=False)
    audit = {
        'output_file': str(output_path),
        'rows': len(result),
        'complete_rows_unchanged': int(complete.sum()),
        'incomplete_rows': int((~complete).sum()),
        'prediction_change_rms': float(np.sqrt(np.mean((prediction - baseline.prediction) ** 2))),
        'prediction_change_max_abs': float(np.max(np.abs(prediction - baseline.prediction))),
        'observed_inputs_preserved': True,
        'repeated_index_check_passed': True,
        'baseline_sha256': baseline_hash,
        'output_sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
        'data_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ['train.csv', 'test.csv']},
    }
    (ROOT / 'submission_v6_audit.json').write_text(json.dumps(audit, indent=2))
    return result


if __name__ == '__main__':
    result = write_submission()
    print(f'Saved {len(result):,} rows to {ROOT / "candidate_v6.csv"}', flush=True)
