"""Check the new upload pair independently of helper files and training data."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_v2 import NUMERIC_COLUMNS, predict_version

HERE = Path(__file__).resolve().parent
warnings.filterwarnings('ignore')


def main():
    test = pd.read_csv(HERE.parent / 'test.csv')
    expected = pd.read_csv(HERE / 'candidate_v2_weather.csv').prediction.to_numpy()
    notebook = json.loads((HERE / 'prediction_notebook.ipynb').read_text())
    script = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    assert 'from frozen_v2 import' not in script and 'from weather_runtime import' not in script
    with tempfile.TemporaryDirectory(prefix='v2-weather-check-') as directory:
        root = Path(directory)
        (root / 'model.pkl').symlink_to(HERE / 'model.pkl')
        (root / 'run_notebook.py').write_text(script)
        def run(rows, tag):
            input_path, output_path = root / f'{tag}_input.csv', root / f'{tag}_output.csv'
            rows.to_csv(input_path, index=False)
            env = {**os.environ, 'DATATHON_INPUT_PATH': str(input_path),
                   'DATATHON_OUTPUT_PATH': str(output_path), 'MPLCONFIGDIR': str(root / 'mpl_cache')}
            subprocess.run([sys.executable, str(root / 'run_notebook.py')], cwd=root, env=env, check=True)
            output = pd.read_csv(output_path)
            assert list(output.columns) == ['prediction'] and len(output) == len(rows)
            assert np.isfinite(output.prediction).all()
            return output.prediction.to_numpy()
        plain = test.drop(columns='id')
        prediction = run(plain, 'ordered')
        np.testing.assert_array_equal(prediction, expected)
        np.testing.assert_array_equal(run(plain, 'repeat'), prediction)
        order = np.random.default_rng(2026).permutation(len(plain))
        shuffled = run(plain.iloc[order].reset_index(drop=True), 'shuffled')
        restored = np.empty(len(order))
        restored[order] = shuffled
        np.testing.assert_array_equal(restored, prediction)
        complete = plain.dropna(subset=NUMERIC_COLUMNS).iloc[:20]
        cases = []
        for pattern in range(16):
            q = complete.copy()
            for bit, c in enumerate(NUMERIC_COLUMNS):
                if pattern & (1 << bit):
                    q[c] = np.nan
            cases.append(q)
        masked = pd.concat(cases, ignore_index=True)
        output = run(masked, 'all_patterns')
        v2 = joblib.load(HERE.parent / 'platform_submission_v2' / 'model.pkl')
        with threadpool_limits(limits=1):
            original = np.round(predict_version(v2, masked, v2['versions']['dl13_prevfix']), 6)
        np.testing.assert_array_equal(output[20:], original[20:])
    print('PASS: isolated notebook; exact candidate; repeated/shuffled/id-free input; all 16 patterns; missing predictions equal v2.')


if __name__ == '__main__':
    main()
