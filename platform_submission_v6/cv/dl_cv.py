"""Cross-validate modern tabular deep-learning models on the same folds / test-like blanks as neural_net_model.ipynb.

usage: python dl_cv.py <model> [folds]     model in: realmlp, realmlp_aug, tabm, tabm_aug
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
FOLDER = "/Users/ethanis/Downloads/Track 1 Dataset/"
NUMERIC_COLUMNS = ["temperature", "humidity", "occupancy", "previous_usage"]
CATEGORICAL = ["building_id", "building_type", "day_of_week"]
train = pd.read_csv(FOLDER + "train.csv")
test_inputs = pd.read_csv(FOLDER + "test.csv").drop(columns=["energy_usage"], errors="ignore")
truth = train["energy_usage"].to_numpy()
test_blank_patterns = test_inputs[NUMERIC_COLUMNS].isna().to_numpy()


def with_test_like_blanks(rows, seed):
    """Identical to neural_net_model.ipynb (same random stream -> same masks)."""
    random_generator = np.random.default_rng(seed)
    blanks = test_blank_patterns[random_generator.integers(0, len(test_inputs), len(rows))]
    masked = rows.copy()
    masked[NUMERIC_COLUMNS] = masked[NUMERIC_COLUMNS].mask(blanks)
    return masked


class Filler:
    """Building x hour median (fallback: building median) learned from the fit rows."""

    def fit(self, rows):
        self.by_hour = {c: rows.groupby(["building_id", "hour"])[c].median() for c in NUMERIC_COLUMNS}
        self.by_building = {c: rows.groupby("building_id")[c].median() for c in NUMERIC_COLUMNS}
        return self

    def features(self, rows):
        out = pd.DataFrame(index=rows.index)
        for c in CATEGORICAL:
            out[c] = rows[c].astype(str)
        out["hour"] = rows["hour"].astype(float)
        out["month"] = rows["month"].astype(float)
        out["weekend"] = rows["day_of_week"].isin(["Saturday", "Sunday"]).astype(float)
        out["hour_sin"] = np.sin(2 * np.pi * rows["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * rows["hour"] / 24)
        out["month_sin"] = np.sin(2 * np.pi * rows["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * rows["month"] / 12)
        key = pd.MultiIndex.from_arrays([rows["building_id"], rows["hour"]])
        for c in NUMERIC_COLUMNS:
            fill = pd.Series(self.by_hour[c].reindex(key).to_numpy(), index=rows.index)
            fill = fill.fillna(rows["building_id"].map(self.by_building[c]))
            out[c] = rows[c].fillna(fill).astype(float)
            out[f"{c}_blank"] = rows[c].isna().astype(float)
        return out


def make_model(name, seed):
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
    if name.startswith("realmlp"):
        return RealMLP_TD_Regressor(device="cpu", random_state=seed, n_threads=8, verbosity=0)
    return TabM_D_Regressor(device="cpu", random_state=seed, n_threads=8, verbosity=0)


def fit_predict(name, fit_rows, predict_sets, seed=0):
    filler = Filler().fit(fit_rows)
    rows = fit_rows
    if name.endswith("_aug"):           # practise on test-like blanks: 2 extra masked copies of every row
        copies = [with_test_like_blanks(fit_rows, seed=1000 + seed * 10 + k) for k in range(2)]
        rows = pd.concat([fit_rows] + copies)
    X = filler.features(rows)
    model = make_model(name, seed)
    model.fit(X, rows["energy_usage"].to_numpy(), cat_col_names=CATEGORICAL)
    return [model.predict(filler.features(p)) for p in predict_sets]


if __name__ == "__main__":
    name = sys.argv[1]
    n_folds_to_run = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    folds = list(KFold(5, shuffle=True, random_state=0).split(train))
    oof = {"as_is": np.full(len(train), np.nan), "test_blanks": np.full(len(train), np.nan)}
    for fold, (fit_positions, check_positions) in enumerate(folds[:n_folds_to_run]):
        started = time.time()
        check = train.iloc[check_positions].drop(columns="energy_usage")
        as_is, masked = fit_predict(name, train.iloc[fit_positions], [check, with_test_like_blanks(check, seed=fold)])
        oof["as_is"][check_positions], oof["test_blanks"][check_positions] = as_is, masked
        r = lambda p: np.sqrt(np.mean((p - truth[check_positions]) ** 2))
        print(f"{name} fold {fold}: as is {r(as_is):.3f} | test-like blanks {r(masked):.3f}  ({time.time() - started:.0f}s)", flush=True)
    if n_folds_to_run == 5:
        np.savez(FOLDER + f"experiments/dl_cv_{name}.npz", **oof)
        print("saved", f"experiments/dl_cv_{name}.npz")
