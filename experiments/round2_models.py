"""Second-round candidates: hierarchical shrinkage and CatBoost corrections."""
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from energy_model import NUM,BUILDINGS,DAYS,features,shift_weights,PatternModel
TYPES=['Administration','Business','Engineering','LectureHall','Library','Residential','Science','Sports']

class PooledExperts(PatternModel):
    def __init__(self,harmonics=3,alpha=1.,building_scale=.35,type_scale=1.,hour_scale=.5,
                 prev_scale=1.,tree_weight=.35,weight_power=.5,prev_prior=.35):
        super().__init__(harmonics=harmonics,alpha=alpha,tree_weight=tree_weight,
                         weight_power=weight_power,trees=250,min_leaf=25,tree_leaves=15,categorical=True)
        self.building_scale=building_scale;self.type_scale=type_scale
        self.hour_scale=hour_scale;self.prev_scale=prev_scale;self.prev_prior=prev_prior
    def design(self,df,obs,sc):
        a=sc.transform(features(df,obs,self.harmonics))
        b=(df.building_id.to_numpy()[:,None]==np.array(BUILDINGS)[None,:]).astype(float)
        t=(df.building_type.to_numpy()[:,None]==np.array(TYPES)[None,:]).astype(float)
        scales=np.ones(a.shape[1])*self.building_scale
        # Numeric deviations can vary by building; regularize calendar deviations more.
        scales[len(obs):]*=self.hour_scale
        if 'previous_usage' in obs:scales[obs.index('previous_usage')]*=self.prev_scale
        individual=((a*scales)[:,:,None]*b[:,None,:]).reshape(len(a),-1)
        typ=((a*self.type_scale)[:,:,None]*t[:,None,:]).reshape(len(a),-1)
        return np.column_stack([a,b,typ,individual])
    def fit(self,df,patterns=None,test_features=None):
        self.models={}
        weights=shift_weights(df,test_features)**self.weight_power if self.weight_power else np.ones(len(df))
        weights=pd.Series(weights/weights.mean(),index=df.index)
        for p in range(16) if patterns is None else patterns:
            obs=[c for k,c in enumerate(NUM) if not p & (1<<k)]
            sub=df.dropna(subset=obs);sc=StandardScaler().fit(features(sub,obs,self.harmonics))
            x=self.design(sub,obs,sc);sw=weights.loc[sub.index].to_numpy()
            offset=self.prev_prior*sub.previous_usage.to_numpy() if 'previous_usage' in obs else np.zeros(len(sub))
            target=sub.energy_usage.to_numpy()-offset
            m=Ridge(alpha=self.alpha).fit(x,target,sample_weight=sw)
            tree=None
            if self.tree_weight:
                tx=self.tree_features(sub,obs)
                tree=HistGradientBoostingRegressor(max_iter=self.trees,max_leaf_nodes=self.tree_leaves,
                    learning_rate=.05,min_samples_leaf=self.min_leaf,l2_regularization=8.,
                    categorical_features=[False]*(tx.shape[1]-1)+[True],early_stopping=False,random_state=42)
                tree.fit(tx,target-m.predict(x),sample_weight=sw)
            self.models[p]=(obs,sc,m,tree)
        return self
    def predict(self,df):
        patterns=(df[NUM].isna().to_numpy().astype(int)*(2**np.arange(4))).sum(axis=1)
        prediction=np.empty(len(df))
        for p in np.unique(patterns):
            ix=np.flatnonzero(patterns==p);sub=df.iloc[ix];obs,sc,m,t=self.models[p]
            prediction[ix]=m.predict(self.design(sub,obs,sc))
            if 'previous_usage' in obs:prediction[ix]+=self.prev_prior*sub.previous_usage.to_numpy()
            if t is not None:prediction[ix]+=self.tree_weight*t.predict(self.tree_features(sub,obs))
        return prediction


def cat_features(df):
    x=df[['building_id','building_type','hour','day_of_week','month']+NUM].copy()
    x['weekend']=df.day_of_week.isin(['Saturday','Sunday']).astype(int)
    x['hour_sin']=np.sin(2*np.pi*df.hour/24);x['hour_cos']=np.cos(2*np.pi*df.hour/24)
    x['month_sin']=np.sin(2*np.pi*df.month/12);x['month_cos']=np.cos(2*np.pi*df.month/12)
    return x

class CatCorrection:
    """Learn a correction to supervised pattern regressions, with optional masking."""
    def __init__(self,depth=5,iterations=900,l2=10.,augment=1,weight_power=.5):
        self.depth=depth;self.iterations=iterations;self.l2=l2;self.augment=augment;self.weight_power=weight_power
    def fit(self,df,test_features):
        from catboost import CatBoostRegressor
        # Pattern-linear base lets the correction model preserve extrapolation.
        self.base=PatternModel(harmonics=3,alpha=1,tree_weight=0,weight_power=.5).fit(df,test_features=test_features)
        rng=np.random.default_rng(2027)
        copies=[df.copy()]
        for k in range(self.augment):
            q=df.copy()
            mask=test_features[NUM].isna().to_numpy()[rng.integers(0,len(test_features),len(q))]
            q[NUM]=q[NUM].mask(mask);copies.append(q)
        data=pd.concat(copies,ignore_index=True)
        target=data.energy_usage.to_numpy()-self.base.predict(data)
        sw=np.tile(shift_weights(df,test_features)**self.weight_power,len(copies))
        sw/=sw.mean()
        self.model=CatBoostRegressor(iterations=self.iterations,depth=self.depth,learning_rate=.04,
            loss_function='RMSE',l2_leaf_reg=self.l2,thread_count=4,random_seed=2027,
            verbose=False,allow_writing_files=False,one_hot_max_size=20)
        self.model.fit(cat_features(data),target,sample_weight=sw,
            cat_features=['building_id','building_type','day_of_week'])
        return self
    def predict_components(self,df):
        return self.base.predict(df),self.model.predict(cat_features(df))
    def predict(self,df,correction_weight=.5):
        a,b=self.predict_components(df);return a+correction_weight*b

class CatImputedBaseline:
    """Fit missing-input models on unlabelled covariates, then the v3 regressors."""
    def __init__(self,depth=5,iterations=650,augment=1,tree_blend=.2,linear_mode='notebook',distill=False,random_seed=527,imputer_weight=0.):
        self.depth=depth;self.iterations=iterations;self.augment=augment;self.tree_blend=tree_blend
        self.linear_mode=linear_mode;self.distill=distill;self.random_seed=random_seed;self.imputer_weight=imputer_weight
    def fit_imputer(self,df,test_features):
        from catboost import CatBoostRegressor
        joint=pd.concat([df.drop(columns='energy_usage',errors='ignore'),test_features],ignore_index=True)
        rng=np.random.default_rng(self.random_seed)
        self.imputation_models={}
        joint_weights=shift_weights(joint,test_features)**self.imputer_weight if self.imputer_weight else np.ones(len(joint))
        joint_weights/=joint_weights.mean()
        for col in NUM:
            observed=joint.loc[joint[col].notna()].copy()
            data=[observed]
            for k in range(self.augment):
                q=observed.copy()
                mask=test_features[NUM].isna().to_numpy()[rng.integers(0,len(test_features),len(q))]
                # The target of this imputation model remains observed.
                mask[:,NUM.index(col)]=False
                q[NUM]=q[NUM].mask(mask);data.append(q)
            data=pd.concat(data,ignore_index=True)
            x=cat_features(data).drop(columns=col)
            m=CatBoostRegressor(iterations=self.iterations,depth=self.depth,learning_rate=.05,
                loss_function='RMSE',l2_leaf_reg=15.,thread_count=4,random_seed=self.random_seed,
                verbose=False,allow_writing_files=False,one_hot_max_size=20)
            m.fit(x,data[col],sample_weight=np.tile(joint_weights[observed.index.to_numpy()],self.augment+1),
                cat_features=['building_id','building_type','day_of_week'])
            self.imputation_models[col]=m
        return self
    def fill(self,df):
        q=df.copy()
        for col,m in self.imputation_models.items():
            missing=df[col].isna()
            if missing.any():q.loc[missing,col]=m.predict(cat_features(df.loc[missing]).drop(columns=col))
        return q
    def fit(self,df,test_features):
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import RidgeCV
        from experiments.notebook_baseline import linear_features,tree_features
        self.fit_imputer(df,test_features)
        data=self.fill(df);weekend=data.day_of_week.isin(['Saturday','Sunday'])
        self.typical=data.assign(is_weekend=weekend).groupby(['building_id','hour','is_weekend'])[['occupancy','previous_usage']].median()
        self.ridge=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13)))
        if self.linear_mode == 'notebook':
            self.ridge.fit(self.regression_features(data,df),data.energy_usage)
        else:
            self.ridge=PatternModel(harmonics=3,alpha=1,tree_weight=0,weight_power=0).fit(data,patterns=[0])
        self.tree=HistGradientBoostingRegressor(categorical_features='from_dtype',max_iter=1500,
            learning_rate=.03,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,
            early_stopping=True,n_iter_no_change=50,random_state=42)
        self.tree.fit(tree_features(data),data.energy_usage-data.previous_usage)
        return self
    def regression_features(self,filled,raw):
        from experiments.notebook_baseline import linear_features
        x=linear_features(filled,self.typical)
        if self.distill:
            # Learn typical load from available feature values, without energy targets.
            mean_prev=self.imputation_models['previous_usage'].predict(cat_features(raw).drop(columns='previous_usage'))
            mean_occ=self.imputation_models['occupancy'].predict(cat_features(raw).drop(columns='occupancy'))
            x=x.copy()
            x['expected_previous']=mean_prev
            x['expected_occupancy']=mean_occ
            for typ in TYPES:
                flag=(filled.building_type==typ).to_numpy(float)
                x[f'expected_previous_{typ}']=mean_prev*flag
                x[f'expected_occupancy_{typ}']=mean_occ*flag
        return x
    def predict(self,df):
        from experiments.notebook_baseline import linear_features,tree_features
        data=self.fill(df)
        a=self.ridge.predict(self.regression_features(data,df)) if self.linear_mode == 'notebook' else self.ridge.predict(data)
        b=self.tree.predict(tree_features(data))+data.previous_usage.to_numpy()
        return (1-self.tree_blend)*a+self.tree_blend*b
