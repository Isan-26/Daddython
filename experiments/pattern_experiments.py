"""Development experiments. Confirmation split is kept out of model selection."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
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

if __name__=='__main__':
    tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
    dev,confirmation=train_test_split(tr,test_size=.2,random_state=2026,stratify=tr.building_id)
    train,val=train_test_split(dev,test_size=.25,random_state=42,stratify=dev.building_id)
    stress=masked_view(val,te,10)
    patterns=sorted(set(((stress[NUM].isna().to_numpy().astype(int)*(2**np.arange(4))).sum(axis=1)).tolist()) | set(((val[NUM].isna().to_numpy().astype(int)*(2**np.arange(4))).sum(axis=1)).tolist()))
    configs=[dict(harmonics=h,alpha=a,extra=e,tree_weight=t) for h,a,e,t in [
        (1,10,'none',0),(2,10,'none',0),(3,10,'none',0),
        (1,1,'none',0),(1,100,'none',0),(2,100,'none',0),
        (2,10,'weekend',0),(2,10,'interactions',0),(2,100,'interactions',0),
        (2,10,'hinges',0),(2,100,'hinges',0),
        (1,10,'none',.5),(2,10,'none',.5),(2,100,'hinges',.5),
        (2,10,'none',1.)]]
    rows=[]
    with threadpool_limits(limits=4):
        for config in configs:
            start=time.time();m=PatternModel(**config).fit(train,patterns)
            row={'config':config,'raw':metrics(val,m.predict(val)),'test_masks':metrics(stress,m.predict(stress)),'seconds':round(time.time()-start,1)}
            rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'experiments'/'development_results.json').write_text(json.dumps(rows,indent=2))
