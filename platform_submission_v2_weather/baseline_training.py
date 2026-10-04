"""Frozen v2 cheap-model training recipe, used for the same-fold baseline."""
import numpy as np
import pandas as pd
import lightgbm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from frozen_v2 import *
CPU_THREADS = 4
MISSFOREST_TREES, MISSFOREST_MIN_LEAF, MISSFOREST_ROUNDS = 50, 10, 4

def is_weekend(rows):
    return rows['day_of_week'].isin(['Saturday', 'Sunday'])


def event_type(rows, reference_rows):
    """0 = normal row, 1-6 = the first event (see table above) that applies to the row."""
    group_keys = ['building_id', 'hour', 'weekend']
    reference = reference_rows.assign(weekend=is_weekend(reference_rows))
    typical = reference.groupby(group_keys)[['occupancy', 'previous_usage']].median().add_suffix('_typical')
    joined = rows.assign(weekend=is_weekend(rows)).join(typical, on=group_keys)

    occupancy_gap = joined['occupancy'] - joined['occupancy_typical']
    previous_ratio = joined['previous_usage'] / joined['previous_usage_typical']
    event_flags = np.column_stack([
        joined['temperature'] >= 33.5,     # 1 heat
        joined['humidity'] >= 93,          # 2 rain
        occupancy_gap >= 100,              # 3 occupancy surge
        occupancy_gap <= -60,              # 4 occupancy drop
        previous_ratio > 1.6,              # 5 previous-usage spike
        previous_ratio < 0.5,              # 6 previous-usage drop
    ])                                     # blank values count as "no event"
    first_event = np.argmax(event_flags, axis=1) + 1
    return pd.Series(np.where(event_flags.any(axis=1), first_event, 0), index=rows.index)


def test_mix_weights(source_rows, target_rows, smoothing=15):
    """Weight for each source row so the source's building x event mix matches the target's."""
    source_events = event_type(source_rows, source_rows)
    target_events = event_type(target_rows, source_rows)       # "typical" values always come from the source rows
    all_events = range(7)

    source_counts = pd.crosstab(source_rows['building_id'], source_events).reindex(index=BUILDING_IDS, columns=all_events, fill_value=0)
    target_counts = pd.crosstab(target_rows['building_id'], target_events).reindex(index=BUILDING_IDS, columns=all_events, fill_value=0)

    # share of each event within each building, smoothed toward the overall event mix
    source_overall_mix = (np.bincount(source_events, minlength=7) + 1) / (len(source_events) + 7)
    target_overall_mix = (np.bincount(target_events, minlength=7) + 1) / (len(target_events) + 7)
    source_event_share = (source_counts + smoothing * source_overall_mix).div(source_counts.sum(axis=1) + smoothing, axis=0)
    target_event_share = (target_counts + smoothing * target_overall_mix).div(target_counts.sum(axis=1) + smoothing, axis=0)
    event_ratio = target_event_share / source_event_share

    # how much more (or less) common each building is in the target
    building_ratio = (target_counts.sum(axis=1) / len(target_rows)) / (source_counts.sum(axis=1) / len(source_rows))

    weights = [event_ratio.loc[building, event] * building_ratio[building]
               for building, event in zip(source_rows['building_id'], source_events)]
    return np.clip(np.array(weights), 0.2, 10.0)


def missforest_inputs(rows):
    """The four numeric columns (may contain blanks) plus never-blank time and building columns."""
    inputs = rows[NUMERIC_COLUMNS].copy()
    inputs['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
    inputs['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
    inputs['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
    inputs['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
    inputs['is_weekend'] = is_weekend(rows).astype(float)
    building_onehot = pd.get_dummies(pd.Categorical(rows['building_id'], categories=BUILDING_IDS), prefix='bid').astype(float)
    building_onehot.index = rows.index
    return pd.concat([inputs, building_onehot], axis=1)


def fit_missforest(training_rows):
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=MISSFOREST_TREES, min_samples_leaf=MISSFOREST_MIN_LEAF, max_features=0.6,
                                      n_jobs=CPU_THREADS, random_state=0),
        max_iter=MISSFOREST_ROUNDS, random_state=0)
    return imputer.fit(missforest_inputs(training_rows))


def fill_blanks_with_missforest(imputer, rows):
    filled = rows.copy()
    filled[NUMERIC_COLUMNS] = imputer.transform(missforest_inputs(rows))[:, :len(NUMERIC_COLUMNS)]
    return filled


def typical_values(filled_rows):
    """Median occupancy and previous usage for each building, hour and weekday/weekend."""
    return (filled_rows.assign(is_weekend=is_weekend(filled_rows))
            .groupby(['building_id', 'hour', 'is_weekend'])[['occupancy', 'previous_usage']].median())


def ridge_features(rows, typical):
    """Features for the Ridge model (rows must have no blanks)."""
    features = rows[NUMERIC_COLUMNS].copy()
    features['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
    features['is_weekend'] = is_weekend(rows).astype(float)
    features['occupancy_x_weekend'] = features['occupancy'] * features['is_weekend']

    building_onehot = pd.get_dummies(pd.Categorical(rows['building_id'], categories=BUILDING_IDS), prefix='bid').astype(float)
    type_onehot = pd.get_dummies(pd.Categorical(rows['building_type'], categories=BUILDING_TYPES), prefix='bt').astype(float)
    building_onehot.index = type_onehot.index = rows.index

    interactions = {}
    # each building gets its own slope for the main drivers
    for column in ['occupancy', 'previous_usage', 'temperature', 'hour_sin', 'hour_cos']:
        for building in building_onehot.columns:
            interactions[f'{column}_x_{building}'] = features[column] * building_onehot[building]
    # weekend effect per building type
    for column in ['is_weekend', 'occupancy_x_weekend']:
        for building_type in type_onehot.columns:
            interactions[f'{column}_x_{building_type}'] = features[column] * type_onehot[building_type]

    # how unusual the row is compared with its building's typical values
    lookup_keys = pd.DataFrame({'building_id': rows['building_id'], 'hour': rows['hour'], 'is_weekend': features['is_weekend'] > 0})
    typical_for_row = lookup_keys.join(typical, on=['building_id', 'hour', 'is_weekend']).fillna(typical.median())
    events = pd.DataFrame({
        'occupancy_gap': features['occupancy'] - typical_for_row['occupancy'],
        'previous_ratio': features['previous_usage'] / typical_for_row['previous_usage'],
        'previous_gap': features['previous_usage'] - typical_for_row['previous_usage'],
        'heat': (features['temperature'] >= 33.5).astype(float),
        'rain': (features['humidity'] >= 93).astype(float),
    }, index=rows.index)
    for column in events.columns:
        for building_type in type_onehot.columns:
            interactions[f'{column}_x_{building_type}'] = events[column] * type_onehot[building_type]

    # occupancy effect by part of the day, per building type
    part_of_day = pd.cut(rows['hour'], [-1, 5, 8, 11, 14, 17, 20, 23], labels=False)
    part_onehot = pd.get_dummies(pd.Categorical(part_of_day, categories=range(7)), prefix='dp').astype(float)
    part_onehot.index = rows.index
    for part in part_onehot.columns:
        for building_type in type_onehot.columns:
            interactions[f'occupancy_{part}_x_{building_type}'] = features['occupancy'] * part_onehot[part] * type_onehot[building_type]

    return pd.concat([features, type_onehot, building_onehot, events, pd.DataFrame(interactions, index=rows.index)], axis=1)


def tree_features(rows):
    """Features for the gradient-boosting model (raw values; building columns treated as categories)."""
    features = rows[['hour', 'month'] + NUMERIC_COLUMNS].copy()
    features['day'] = rows['day_of_week'].map(DAY_NUMBER)
    features['is_weekend'] = (features['day'] >= 5).astype(int)
    features['building_type'] = pd.Categorical(rows['building_type'], categories=BUILDING_TYPES)
    features['building_id'] = pd.Categorical(rows['building_id'], categories=BUILDING_IDS)
    return features


def new_gradient_boosting():
    return HistGradientBoostingRegressor(categorical_features='from_dtype', max_iter=1500, learning_rate=0.03,
                                         max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
                                         early_stopping=True, n_iter_no_change=50, random_state=42)


class V3NotebookModel:
    def fit(self, training_rows):
        self.imputer = fit_missforest(training_rows)                    # learns from training rows only
        filled = fill_blanks_with_missforest(self.imputer, training_rows)
        target = filled['energy_usage']
        self.typical = typical_values(filled)
        self.ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.ridge.fit(ridge_features(filled, self.typical), target)
        self.boosting = new_gradient_boosting()
        self.boosting.fit(tree_features(filled), target - filled['previous_usage'])     # predicts the change
        return self

    def predict(self, rows):
        filled = fill_blanks_with_missforest(self.imputer, rows)
        ridge_prediction = self.ridge.predict(ridge_features(filled, self.typical))
        boosting_prediction = self.boosting.predict(tree_features(filled)) + filled['previous_usage'].to_numpy()
        return 0.8 * ridge_prediction + 0.2 * boosting_prediction


def pattern_model_inputs(rows, available_columns, n_daily_curves=3):
    """Available numeric columns, a weekend flag, daily sine/cosine curves and a yearly sine/cosine curve."""
    hour = rows['hour'].to_numpy(float)
    columns = {name: rows[name].to_numpy(float) for name in available_columns}
    columns['weekend'] = is_weekend(rows).to_numpy(float)
    for k in range(1, n_daily_curves + 1):                     # k = 1, 2, 3: once, twice, three times a day
        columns[f'hour_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24)
        columns[f'hour_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24)
    columns['month_sin'] = np.sin(2 * np.pi * rows['month'].to_numpy() / 12)
    columns['month_cos'] = np.cos(2 * np.pi * rows['month'].to_numpy() / 12)
    return np.column_stack(list(columns.values()))


def blank_pattern_number(rows):
    """0-15: bit k is set when NUMERIC_COLUMNS[k] is blank (0 = nothing blank)."""
    blanks = rows[NUMERIC_COLUMNS].isna().to_numpy().astype(int)
    return (blanks * 2 ** np.arange(len(NUMERIC_COLUMNS))).sum(axis=1)


class MissingPatternModels:
    def __init__(self, n_daily_curves=3, ridge_alpha=1.0, correction_weight=0.65,
                 correction_trees=250, correction_min_leaf=25, correction_max_leaves=15, test_mix_power=0.5):
        self.n_daily_curves = n_daily_curves
        self.ridge_alpha = ridge_alpha
        self.correction_weight = correction_weight
        self.correction_trees = correction_trees
        self.correction_min_leaf = correction_min_leaf
        self.correction_max_leaves = correction_max_leaves
        self.test_mix_power = test_mix_power

    def _ridge_inputs(self, rows, available_columns, scaler):
        scaled = scaler.transform(pattern_model_inputs(rows, available_columns, self.n_daily_curves))
        building_onehot = (rows['building_id'].to_numpy()[:, None] == np.array(BUILDING_IDS)[None, :]).astype(float)
        per_building = (scaled[:, :, None] * building_onehot[:, None, :]).reshape(len(scaled), -1)   # every input x every building
        return np.column_stack([scaled, building_onehot, per_building])

    @staticmethod
    def _correction_inputs(rows, available_columns):
        numeric = rows[['hour', 'month'] + available_columns].to_numpy(float)
        return np.column_stack([numeric, rows['day_of_week'].map(DAY_NUMBER), rows['building_id'].map(BUILDING_NUMBER)])

    def fit(self, training_rows, test_rows):
        weights = test_mix_weights(training_rows, test_rows) ** self.test_mix_power
        weights = pd.Series(weights / weights.mean(), index=training_rows.index)
        self.models = {}
        for pattern in range(2 ** len(NUMERIC_COLUMNS)):
            available = [column for bit, column in enumerate(NUMERIC_COLUMNS) if not pattern & (1 << bit)]
            usable_rows = training_rows.dropna(subset=available)          # rows that have every available column
            scaler = StandardScaler().fit(pattern_model_inputs(usable_rows, available, self.n_daily_curves))
            ridge_inputs = self._ridge_inputs(usable_rows, available, scaler)
            row_weights = weights.loc[usable_rows.index].to_numpy()

            ridge = Ridge(alpha=self.ridge_alpha).fit(ridge_inputs, usable_rows['energy_usage'], sample_weight=row_weights)
            correction_inputs = self._correction_inputs(usable_rows, available)
            correction = HistGradientBoostingRegressor(
                max_iter=self.correction_trees, max_leaf_nodes=self.correction_max_leaves, learning_rate=0.05,
                min_samples_leaf=self.correction_min_leaf, l2_regularization=8.0, early_stopping=False, random_state=42,
                categorical_features=[False] * (correction_inputs.shape[1] - 1) + [True])   # last column = building
            correction.fit(correction_inputs, usable_rows['energy_usage'].to_numpy() - ridge.predict(ridge_inputs),
                           sample_weight=row_weights)
            self.models[pattern] = (available, scaler, ridge, correction)
        return self

    def predict(self, rows):
        patterns = blank_pattern_number(rows)
        prediction = np.empty(len(rows))
        for pattern in np.unique(patterns):
            positions = np.flatnonzero(patterns == pattern)
            pattern_rows = rows.iloc[positions]
            available, scaler, ridge, correction = self.models[pattern]
            prediction[positions] = (ridge.predict(self._ridge_inputs(pattern_rows, available, scaler))
                                     + self.correction_weight * correction.predict(self._correction_inputs(pattern_rows, available)))
        return prediction


IMPUTER_CATEGORIES = {'building_id': BUILDING_IDS, 'building_type': BUILDING_TYPES, 'day_of_week': DAY_NAMES}


def imputer_inputs(rows):
    """Everything the imputers may look at (the column being predicted is dropped later)."""
    inputs = rows[['building_id', 'building_type', 'hour', 'day_of_week', 'month'] + NUMERIC_COLUMNS].copy()
    inputs['weekend'] = is_weekend(rows).astype(int)
    inputs['hour_sin'] = np.sin(2 * np.pi * rows['hour'] / 24)
    inputs['hour_cos'] = np.cos(2 * np.pi * rows['hour'] / 24)
    inputs['month_sin'] = np.sin(2 * np.pi * rows['month'] / 12)
    inputs['month_cos'] = np.cos(2 * np.pi * rows['month'] / 12)
    for column, categories in IMPUTER_CATEGORIES.items():          # text columns -> LightGBM categories
        inputs[column] = pd.Categorical(inputs[column], categories=categories)
    return inputs


def new_lightgbm_imputer(seed):
    return lightgbm.LGBMRegressor(n_estimators=2200, learning_rate=0.025, max_depth=5, num_leaves=32,
                                  min_child_samples=30, reg_lambda=15.0, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, cat_smooth=20, verbose=-1, random_state=seed, n_jobs=CPU_THREADS)


class LightGBMImputedModel:
    def __init__(self, seed, practice_copies=3, test_mix_power=1.5, boosting_share=0.2):
        self.seed = seed
        self.practice_copies = practice_copies
        self.test_mix_power = test_mix_power
        self.boosting_share = boosting_share

    # ---- C1: the imputers ----
    def fit_imputers(self, training_rows, test_rows):
        all_inputs = pd.concat([training_rows.drop(columns='energy_usage'), test_rows], ignore_index=True)
        random_generator = np.random.default_rng(self.seed)
        weights = test_mix_weights(all_inputs, test_rows) ** self.test_mix_power
        weights = weights / weights.mean()
        test_blank_patterns = test_rows[NUMERIC_COLUMNS].isna().to_numpy()

        self.imputers = {}
        for column in NUMERIC_COLUMNS:
            observed = all_inputs.loc[all_inputs[column].notna()].copy()   # rows where this column is known
            practice_sets = [observed]
            for _ in range(self.practice_copies):
                copy = observed.copy()
                blanks = test_blank_patterns[random_generator.integers(0, len(test_rows), len(copy))]
                blanks[:, NUMERIC_COLUMNS.index(column)] = False            # keep the column being learned
                copy[NUMERIC_COLUMNS] = copy[NUMERIC_COLUMNS].mask(blanks)
                practice_sets.append(copy)
            practice_data = pd.concat(practice_sets, ignore_index=True)
            practice_weights = np.tile(weights[observed.index.to_numpy()], self.practice_copies + 1)

            imputer = new_lightgbm_imputer(self.seed)
            imputer.fit(imputer_inputs(practice_data).drop(columns=column), practice_data[column], sample_weight=practice_weights)
            self.imputers[column] = imputer
        return self

    def expected_value(self, rows, column):
        """What the imputer thinks `column` should be, given the rest of the row."""
        return self.imputers[column].predict(imputer_inputs(rows).drop(columns=column))

    def fill_blanks(self, rows):
        filled = rows.copy()
        for column in NUMERIC_COLUMNS:
            blank = rows[column].isna()
            if blank.any():
                filled.loc[blank, column] = self.expected_value(rows.loc[blank], column)
        return filled

    # ---- C2: the energy model ----
    def model_features(self, filled_rows, original_rows):
        features = ridge_features(filled_rows, self.typical).copy()
        expected_previous = self.expected_value(original_rows, 'previous_usage')
        expected_occupancy = self.expected_value(original_rows, 'occupancy')
        features['expected_previous'] = expected_previous
        features['expected_occupancy'] = expected_occupancy
        for building_type in BUILDING_TYPES:
            this_type = (filled_rows['building_type'] == building_type).to_numpy(float)
            features[f'expected_previous_{building_type}'] = expected_previous * this_type
            features[f'expected_occupancy_{building_type}'] = expected_occupancy * this_type
        return features

    def fit(self, training_rows, test_rows):
        self.fit_imputers(training_rows, test_rows)
        filled = self.fill_blanks(training_rows)
        self.typical = typical_values(filled)
        self.ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.ridge.fit(self.model_features(filled, training_rows), filled['energy_usage'])
        self.boosting = new_gradient_boosting()
        self.boosting.fit(tree_features(filled), filled['energy_usage'] - filled['previous_usage'])
        return self

    def predict(self, rows):
        filled = self.fill_blanks(rows)
        ridge_prediction = self.ridge.predict(self.model_features(filled, rows))
        boosting_prediction = self.boosting.predict(tree_features(filled)) + filled['previous_usage'].to_numpy()
        return (1 - self.boosting_share) * ridge_prediction + self.boosting_share * boosting_prediction
