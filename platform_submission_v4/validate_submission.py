"""Run the notebook in isolation with only its artifact and input files."""
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
from uncertainty_runtime import predict_with_uncertainty

HERE = Path(__file__).resolve().parent
warnings.filterwarnings('ignore')


def main():
    test = pd.read_csv(HERE.parent / 'test.csv')
    expected = pd.read_csv(HERE / 'candidate_v4_uncertainty.csv').prediction.to_numpy()
    notebook = json.loads((HERE / 'prediction_notebook.ipynb').read_text())
    script = '\n\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert 'from frozen_v2 import' not in script and 'from uncertainty_runtime import' not in script
    with tempfile.TemporaryDirectory(prefix='v4-uncertainty-check-') as directory:
        root = Path(directory)
        (root / 'model.pkl').symlink_to(HERE / 'model.pkl')
        def run(rows, tag, version='v4_uncertainty'):
            source = script.replace("VERSION = 'v4_uncertainty'", f"VERSION = '{version}'")
            (root / 'run_notebook.py').write_text(source)
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
        np.testing.assert_array_equal(run(test.iloc[:100], 'with_ids'), prediction[:100])
        # Separate batch calls must give the same results as the complete file.
        np.testing.assert_array_equal(run(plain.iloc[100:200].reset_index(drop=True), 'separate_batch'), prediction[100:200])
        complete = plain.dropna(subset=NUMERIC_COLUMNS).iloc[:20]
        cases = []
        for pattern in range(16):
            masked = complete.copy()
            for bit, column in enumerate(NUMERIC_COLUMNS):
                if pattern & (1 << bit):
                    masked[column] = np.nan
            cases.append(masked)
        masked = pd.concat(cases, ignore_index=True)
        all_patterns = run(masked, 'all_patterns')
        bundle = joblib.load(HERE / 'model.pkl')
        with threadpool_limits(limits=1):
            original = np.round(predict_version(bundle['v2'], masked,
                bundle['v2']['versions']['dl13_prevfix']), 6)
            conservative = np.round(predict_with_uncertainty(bundle, plain, 'v4_conservative'), 6)
        np.testing.assert_array_equal(all_patterns[:20], original[:20])
        np.testing.assert_array_equal(run(masked, 'all_patterns_fallback', 'v2_fallback'), original)
        baseline = run(plain, 'v2_fallback', 'v2_fallback')
        np.testing.assert_array_equal(baseline,
            pd.read_csv(HERE / 'candidate_v2_fallback.csv').prediction.to_numpy())
        np.testing.assert_array_equal(baseline,
            pd.read_csv(HERE.parent / 'platform_submission_v2/platform_dl13_prevfix.csv').prediction.to_numpy())
        np.testing.assert_array_equal(conservative,
            pd.read_csv(HERE / 'candidate_v4_conservative.csv').prediction.to_numpy())
        complete_mask = plain[NUMERIC_COLUMNS].notna().all(axis=1).to_numpy()
        np.testing.assert_array_equal(prediction[complete_mask], baseline[complete_mask])
    result = {'isolated_notebook': True, 'candidate_exact': True,
        'repeat_exact': True, 'shuffle_exact': True, 'ids_ignored': True,
        'separate_batches_exact': True, 'all_16_patterns_finite': True,
        'complete_rows_equal_v2': True, 'fallback_all_patterns_equal_v2': True,
        'fallback_csv_equal_original_v2': True, 'conservative_csv_exact': True,
        'new_energy_rmse_claimed': False}
    (HERE / 'validation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: isolated notebook; repeat/shuffle/batch/id-free consistency; all 16 patterns; '
          'complete rows unchanged; exact original v2 fallback; conservative CSV matches.')


if __name__ == '__main__':
    main()
