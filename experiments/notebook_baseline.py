"""The current v3 model with preprocessing restricted to training rows."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.experimental import enable_iterative_imputer  # must come before importing IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor



num_cols = ['temperature', 'humidity', 'occupancy', 'previous_usage']
building_ids = ['ADM_A','BUS_A','BUS_B','ENG_A','ENG_B','LEC_A','LIB_A','RES_A','RES_B','SCI_A','SCI_B','SPT_A']
building_types = ['Administration','Business','Engineering','LectureHall','Library','Residential','Science','Sports']
day_to_int = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
              'Friday': 4, 'Saturday': 5, 'Sunday': 6}


def imputer_inputs(df):
    """The 4 columns that have blanks, plus never-blank columns the imputer can learn from."""
    X = df[num_cols].copy()
    X['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    X['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    X['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    X['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(float)
    bid = pd.get_dummies(pd.Categorical(df['building_id'], categories=building_ids), prefix='bid').astype(float)
    bid.index = df.index
    return pd.concat([X, bid], axis=1)


def fit_imputer(train_df):
    """MissForest-style: each blank column is guessed by a tree model trained on the other columns."""
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.6,
                                      n_jobs=4, random_state=0),
        max_iter=8, random_state=0)
    imputer.fit(imputer_inputs(train_df))
    return imputer


def fill_blanks(imputer, df):
    df = df.copy()
    df[num_cols] = imputer.transform(imputer_inputs(df))[:, :len(num_cols)]
    return df


def linear_features(df, typical):
    X = df[num_cols].copy()
    X['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    X['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    X['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    X['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(float)
    X['occ_x_weekend'] = X['occupancy'] * X['is_weekend']

    bid = pd.get_dummies(pd.Categorical(df['building_id'], categories=building_ids), prefix='bid').astype(float)
    btype = pd.get_dummies(pd.Categorical(df['building_type'], categories=building_types), prefix='bt').astype(float)
    bid.index = btype.index = df.index

    # each building gets its own slope for the main drivers
    inter = {}
    for col in ['occupancy', 'previous_usage', 'temperature', 'hour_sin', 'hour_cos']:
        for b in bid.columns:
            inter[f'{col}_x_{b}'] = X[col] * bid[b]
    for col in ['is_weekend', 'occ_x_weekend']:
        for b in btype.columns:
            inter[f'{col}_x_{b}'] = X[col] * btype[b]
        # NEW: how unusual the row is for its building, hour and weekday/weekend (event rows)
    key = pd.DataFrame({'building_id': df['building_id'], 'hour': df['hour'], 'is_weekend': X['is_weekend'] > 0})
    t = key.join(typical, on=['building_id', 'hour', 'is_weekend']).fillna(typical.median())
    events = pd.DataFrame({
        'occ_gap': X['occupancy'] - t['occupancy'],
        'prev_ratio': X['previous_usage'] / t['previous_usage'],
        'prev_gap': X['previous_usage'] - t['previous_usage'],
        'heat': (X['temperature'] >= 33.5).astype(float),
        'rain': (X['humidity'] >= 93).astype(float),
    }, index=df.index)
    for col in events.columns:
        for b in btype.columns:
            inter[f'{col}_x_{b}'] = events[col] * btype[b]

    # NEW: occupancy effect by part of the day, per building type
    part = pd.cut(df['hour'], [-1, 5, 8, 11, 14, 17, 20, 23], labels=False)
    daypart = pd.get_dummies(pd.Categorical(part, categories=range(7)), prefix='dp').astype(float)
    daypart.index = df.index
    for p in daypart.columns:
        for b in btype.columns:
            inter[f'occ_{p}_x_{b}'] = X['occupancy'] * daypart[p] * btype[b]

    return pd.concat([X, btype, bid, events, pd.DataFrame(inter, index=df.index)], axis=1)


def tree_features(df):
    X = df[['hour', 'month'] + num_cols].copy()
    X['day'] = df['day_of_week'].map(day_to_int)
    X['is_weekend'] = (X['day'] >= 5).astype(int)
    X['building_type'] = pd.Categorical(df['building_type'], categories=building_types)
    X['building_id'] = pd.Categorical(df['building_id'], categories=building_ids)
    return X


class NotebookBaseline:
    def fit(self, train_df):
        self.imputer = fit_imputer(train_df)
        train_df = fill_blanks(self.imputer, train_df)
        y = train_df['energy_usage']
        weekend = train_df.day_of_week.isin(['Saturday', 'Sunday'])
        self.typical = train_df.assign(is_weekend=weekend).groupby(
            ['building_id', 'hour', 'is_weekend'])[['occupancy', 'previous_usage']].median()
        self.ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.ridge.fit(linear_features(train_df, self.typical), y)
        self.hgb = HistGradientBoostingRegressor(
            categorical_features='from_dtype', max_iter=1500, learning_rate=.03,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.,
            early_stopping=True, n_iter_no_change=50, random_state=42)
        self.hgb.fit(tree_features(train_df), y-train_df.previous_usage)
        return self
    def predict(self, pred_df):
        df = fill_blanks(self.imputer, pred_df)
        ridge = self.ridge.predict(linear_features(df, self.typical))
        hgb = self.hgb.predict(tree_features(df)) + df.previous_usage.to_numpy()
        return .8*ridge + .2*hgb
