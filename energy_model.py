"""Energy prediction with models specialized for the numeric inputs available.

The selected model blends the notebook's v3 architecture with supervised experts
for each missing-value pattern. Only labelled training rows are used to fit
regressions or imputation. Unlabelled test features estimate the event/building
mix for modest importance weighting. Settings were selected on development CV;
confirmation metrics are stored in validation_results.json.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
NUM = ['temperature', 'humidity', 'occupancy', 'previous_usage']
BUILDINGS = ['ADM_A','BUS_A','BUS_B','ENG_A','ENG_B','LEC_A','LIB_A','RES_A','RES_B','SCI_A','SCI_B','SPT_A']
DAYS = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

def features(df, observed, harmonics=1, extra='none'):
    z = {c:df[c].to_numpy(float) for c in observed}
    w = df.day_of_week.isin(['Saturday','Sunday']).to_numpy(float)
    h = df.hour.to_numpy(float)
    z['weekend'] = w
    for k in range(1, harmonics+1):
        z[f'sin{k}'] = np.sin(2*np.pi*k*h/24)
        z[f'cos{k}'] = np.cos(2*np.pi*k*h/24)
    z['month_sin'] = np.sin(2*np.pi*df.month.to_numpy()/12)
    z['month_cos'] = np.cos(2*np.pi*df.month.to_numpy()/12)
    if extra in ['weekend','interactions','hinges']:
        z['weekend_sin'] = w*z['sin1']
        z['weekend_cos'] = w*z['cos1']
    if extra in ['interactions','hinges']:
        if 'occupancy' in observed:
            z['occ_weekend'] = z['occupancy']*w
            z['occ_sin'] = z['occupancy']*z['sin1']
            z['occ_cos'] = z['occupancy']*z['cos1']
    if extra == 'hinges':
        if 'temperature' in observed:
            z['cooling'] = np.maximum(0,z['temperature']-28)
        if 'humidity' in observed:
            z['rain'] = np.maximum(0,z['humidity']-90)
    return np.column_stack(list(z.values()))

class PatternModel:
    """Train one supervised model for each subset of available numeric inputs."""
    def __init__(self, harmonics=1, alpha=10., extra='none', tree_weight=0., trees=180, min_leaf=35, weight_power=0., tree_leaves=12, categorical=False):
        self.harmonics=harmonics;self.alpha=alpha;self.extra=extra
        self.tree_weight=tree_weight;self.trees=trees;self.min_leaf=min_leaf;self.weight_power=weight_power;self.tree_leaves=tree_leaves;self.categorical=categorical
    def fit(self, df, patterns=None, test_features=None):
        self.models={}
        weights = shift_weights(df, test_features) ** self.weight_power if self.weight_power else np.ones(len(df))
        weights = pd.Series(weights / np.mean(weights), index=df.index)
        if patterns is None: patterns=range(16)
        for pattern in patterns:
            obs=[c for k,c in enumerate(NUM) if not (pattern & (1<<k))]
            sub=df.dropna(subset=obs).copy()
            a=features(sub,obs,self.harmonics,self.extra)
            sc=StandardScaler().fit(a);a=sc.transform(a)
            b=(sub.building_id.to_numpy()[:,None]==np.array(BUILDINGS)[None,:]).astype(float)
            x=np.column_stack([a,b,(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)])
            sw = weights.loc[sub.index].to_numpy()
            m=Ridge(alpha=self.alpha).fit(x,sub.energy_usage,sample_weight=sw)
            t=None
            if self.tree_weight:
                tx=self.tree_features(sub,obs)
                t=HistGradientBoostingRegressor(max_iter=self.trees,max_leaf_nodes=self.tree_leaves,learning_rate=.05,
                      categorical_features=([False]*(tx.shape[1]-1)+[True]) if self.categorical else None,
                      min_samples_leaf=self.min_leaf,l2_regularization=8.,early_stopping=False,random_state=42)
                t.fit(tx,sub.energy_usage.to_numpy()-m.predict(x),sample_weight=sw)
            self.models[pattern]=(obs,sc,m,t)
        return self
    @staticmethod
    def tree_features(df,obs):
        a=df[['hour','month']+obs].to_numpy(float)
        return np.column_stack([a,df.day_of_week.map(DAYS),df.building_id.map(dict(zip(BUILDINGS,range(12))))])
    def predict(self,df):
        patterns=(df[NUM].isna().to_numpy().astype(int)*(2**np.arange(4))).sum(axis=1)
        pred=np.empty(len(df))
        for pattern in np.unique(patterns):
            ix=np.flatnonzero(patterns==pattern);sub=df.iloc[ix]
            obs,sc,m,t=self.models[pattern]
            a=sc.transform(features(sub,obs,self.harmonics,self.extra))
            b=(sub.building_id.to_numpy()[:,None]==np.array(BUILDINGS)[None,:]).astype(float)
            x=np.column_stack([a,b,(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)])
            pred[ix]=m.predict(x)
            if t is not None:pred[ix]+=self.tree_weight*t.predict(self.tree_features(sub,obs))
        return pred

def event_groups(df, reference):
    """Feature-only event strata; typical values are fitted on the reference fold."""
    key=['building_id','hour','weekend']
    ref=reference.assign(weekend=reference.day_of_week.isin(['Saturday','Sunday']))
    typical=ref.groupby(key)[['occupancy','previous_usage']].median()
    q=df.assign(weekend=df.day_of_week.isin(['Saturday','Sunday'])).join(typical.add_suffix('_typical'),on=key)
    flags=np.column_stack([
        q.temperature>=33.5, q.humidity>=93,
        q.occupancy-q.occupancy_typical>=100, q.occupancy-q.occupancy_typical<=-60,
        q.previous_usage/q.previous_usage_typical>1.6, q.previous_usage/q.previous_usage_typical<.5])
    # A primary event class avoids unreliable weights for nearly empty intersections.
    event=np.where(flags.any(axis=1),np.argmax(flags,axis=1)+1,0)
    return pd.Series(event,index=df.index)

def shift_weights(source,target):
    if target is None: raise ValueError('Unlabelled test features required for shift weighting.')
    es=event_groups(source,source);et=event_groups(target,source)
    counts_s=pd.crosstab(source.building_id,es).reindex(index=BUILDINGS,columns=range(7),fill_value=0)
    counts_t=pd.crosstab(target.building_id,et).reindex(index=BUILDINGS,columns=range(7),fill_value=0)
    # Smooth event proportions toward the global event mix within each building.
    ps=(counts_s+15*(np.bincount(es,minlength=7)+1)/(len(es)+7)).div(counts_s.sum(axis=1)+15,axis=0)
    pt=(counts_t+15*(np.bincount(et,minlength=7)+1)/(len(et)+7)).div(counts_t.sum(axis=1)+15,axis=0)
    ratio=pt/ps
    b_ratio=(counts_t.sum(axis=1)/len(target))/(counts_s.sum(axis=1)/len(source))
    return np.clip(np.array([ratio.loc[b,e]*b_ratio[b] for b,e in zip(source.building_id,es)]),.2,10.)

def masked_view(df,test,seed):
    # Exact empirical test masks on COMPLETE validation rows. No extra masks on existing blanks.
    out=df.dropna(subset=NUM).copy()
    rng=np.random.default_rng(seed)
    masks=test[NUM].isna().to_numpy()[rng.integers(0,len(test),len(out))]
    out[NUM]=out[NUM].mask(masks)
    return out

def metrics(df,pred):
    missing=df[NUM].isna().any(axis=1).to_numpy()
    out={'rmse':float(root_mean_squared_error(df.energy_usage,pred))}
    for label,mask in [('complete',~missing),('missing',missing)]:
        out[label]=float(root_mean_squared_error(df.energy_usage.to_numpy()[mask],pred[mask])) if mask.any() else None
    return out


# Reproduces the v3 architecture with its imputer fitted on training rows.
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


class EnergyModel:
    """Selected ensemble. The config is fixed before confirmation evaluation."""
    def __init__(self, config=None):
        self.config = config or json.loads((ROOT / 'model_config.json').read_text())

    def fit(self, train_df, test_features):
        # No prediction targets are used by either model or by shift_weights.
        test_features = test_features.drop(columns=['energy_usage'], errors='ignore')
        self.baseline = NotebookBaseline().fit(train_df)
        self.experts = PatternModel(**self.config['pattern_config']).fit(
            train_df, test_features=test_features)
        return self

    def predict_components(self, df):
        inputs = df.drop(columns=['energy_usage'], errors='ignore')
        return self.baseline.predict(inputs), self.experts.predict(inputs)

    def predict(self, df):
        baseline, experts = self.predict_components(df)
        missing = df[NUM].isna().any(axis=1).to_numpy()
        weights = np.where(missing, self.config['missing_weight'],
                           self.config['complete_weight'])
        return (1 - weights) * baseline + weights * experts


def write_submission(train_path=ROOT / 'train.csv', test_path=ROOT / 'test.csv',
                     output_path=ROOT / 'candidate_v4.csv'):
    """Fit on all labelled rows and preserve the original test ID order."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        prediction = EnergyModel().fit(train, test).predict(test)
    if not np.isfinite(prediction).all():
        raise ValueError('Predictions contain non-finite values.')
    if test.id.duplicated().any():
        raise ValueError('Test IDs are not unique.')
    submission = pd.DataFrame({'id': test.id, 'prediction': prediction})
    submission.to_csv(output_path, index=False)
    return submission


if __name__ == '__main__':
    result = write_submission()
    print(f'Saved {len(result):,} predictions to {ROOT / "candidate_v4.csv"}')
