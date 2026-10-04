"""Execute the upload notebook in an isolated directory with no training files."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    test = pd.read_csv(HERE.parent / 'test.csv')
    expected = pd.read_csv(HERE / 'candidate_feature_upgrade.csv').prediction.to_numpy()
    notebook = json.loads((HERE / 'prediction_notebook.ipynb').read_text())
    script = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    assert 'from legacy_runtime import' not in script
    assert 'from upgrade_runtime import' not in script
    with tempfile.TemporaryDirectory(prefix='datathon-v3-check-') as directory:
        root = Path(directory)
        (root / 'feature_upgrade_model.pkl').symlink_to(HERE / 'feature_upgrade_model.pkl')
        (root / 'run_notebook.py').write_text(script)

        def run(rows, tag):
            input_path, output_path = root / (tag + '_input.csv'), root / (tag + '_output.csv')
            rows.to_csv(input_path, index=False)
            env = {**os.environ, 'DATATHON_INPUT_PATH': str(input_path),
                   'DATATHON_OUTPUT_PATH': str(output_path)}
            subprocess.run([sys.executable, str(root / 'run_notebook.py')], cwd=root, env=env, check=True)
            frame = pd.read_csv(output_path)
            assert list(frame.columns) == ['prediction'] and len(frame) == len(rows)
            assert np.isfinite(frame.prediction.to_numpy()).all()
            return frame.prediction.to_numpy()

        plain = test.drop(columns='id')
        prediction = run(plain, 'ordered')
        np.testing.assert_array_equal(prediction, expected)
        np.testing.assert_array_equal(run(plain, 'repeat'), prediction)
        order = np.random.default_rng(2026).permutation(len(plain))
        shuffled = run(plain.iloc[order].reset_index(drop=True), 'shuffled')
        restored = np.empty(len(order))
        restored[order] = shuffled
        np.testing.assert_array_equal(restored, prediction)
        columns = ['temperature', 'humidity', 'occupancy', 'previous_usage']
        complete = plain.dropna(subset=columns).iloc[:20]
        cases = []
        for pattern in range(16):
            q = complete.copy()
            for bit, c in enumerate(columns):
                if pattern & (1 << bit):
                    q[c] = np.nan
            cases.append(q)
        run(pd.concat(cases, ignore_index=True), 'all_patterns')
    print('PASS: isolated notebook, exact candidate match, repeatability, shuffled/id-free rows, all 16 patterns.')


if __name__ == '__main__':
    main()
