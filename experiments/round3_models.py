"""Third-round models: supervised mask training and conditional imputers."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from energy_model import NUM, shift_weights
from experiments.round2_models import CatImputedBaseline, cat_features, TYPES
from experiments.notebook_baseline import tree_features


class MaskTrainedEnergy(CatImputedBaseline):
    def __init__(self, energy_augment=1, energy_weight=1., missing_flags=True,
                 probability_events=False, random_seed=527, **kwargs):
        super().__init__(random_seed=random_seed, **kwargs)
        self.energy_augment=energy_augment
        self.energy_weight=energy_weight
        self.missing_flags=missing_flags
        self.probability_events=probability_events

    def fit_weather_models(self, df, test_features):
        from catboost import CatBoostClassifier
        joint=pd.concat([df.drop(columns='energy_usage',errors='ignore'),test_features],ignore_index=True)
        weights=shift_weights(joint,test_features)**self.imputer_weight
        weights/=weights.mean()
        rng=np.random.default_rng(self.random_seed+111)
        self.weather_models={}
        for tag,col,threshold in [('heat','temperature',33.5),('rain','humidity',93.)]:
            observed=joint[joint[col].notna()].copy()
            label=(observed[col].to_numpy()>=threshold).astype(int)
            copies=[observed]
            for k in range(3):
                q=observed.copy()
                mask=test_features[NUM].isna().to_numpy()[rng.integers(0,len(test_features),len(q))]
                mask[:,NUM.index(col)]=False
                q[NUM]=q[NUM].mask(mask)
                copies.append(q)
            data=pd.concat(copies,ignore_index=True)
            model=CatBoostClassifier(iterations=550,depth=4,learning_rate=.05,l2_leaf_reg=15,
                thread_count=4,random_seed=self.random_seed,verbose=False,
                allow_writing_files=False,one_hot_max_size=20)
            model.fit(cat_features(data).drop(columns=col),np.tile(label,len(copies)),
                sample_weight=np.tile(weights[observed.index.to_numpy()],len(copies)),
                cat_features=['building_id','building_type','day_of_week'])
            self.weather_models[tag]=(col,threshold,model)

    def regression_features(self, filled, raw):
        x=super().regression_features(filled,raw).copy()
        if self.probability_events:
            for tag,(col,threshold,model) in self.weather_models.items():
                probability=(raw[col].to_numpy()>=threshold).astype(float)
                missing=raw[col].isna().to_numpy()
                if missing.any():
                    probability[missing]=model.predict_proba(
                        cat_features(raw.iloc[np.flatnonzero(missing)]).drop(columns=col))[:,1]
                x[tag]=probability
                for typ in TYPES:
                    x[f'{tag}_x_bt_{typ}']=probability*(raw.building_type==typ).to_numpy(float)
        if self.missing_flags:
            for col in NUM:
                missing=raw[col].isna().to_numpy(float)
                x[f'missing_{col}']=missing
                for typ in TYPES:
                    x[f'missing_{col}_{typ}']=missing*(raw.building_type==typ).to_numpy(float)
        return x

    def fit(self, df, test_features):
        self.fit_imputer(df,test_features)
        if self.probability_events:
            self.fit_weather_models(df,test_features)
        filled=self.fill(df)
        weekend=filled.day_of_week.isin(['Saturday','Sunday'])
        self.typical=filled.assign(is_weekend=weekend).groupby(
            ['building_id','hour','is_weekend'])[['occupancy','previous_usage']].median()
        sw=shift_weights(df,test_features)**self.energy_weight if self.energy_weight else np.ones(len(df))
        sw/=sw.mean()
        # Choose shrinkage and stopping length without duplicated augmentation rows.
        raw_features=self.regression_features(filled,df)
        selector=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13)))
        selector.fit(raw_features,df.energy_usage,ridgecv__sample_weight=sw)
        self.raw_alpha=float(selector[-1].alpha_)
        tree=HistGradientBoostingRegressor(categorical_features='from_dtype',max_iter=1500,
            learning_rate=.03,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,
            early_stopping=True,n_iter_no_change=50,random_state=42)
        tree.fit(tree_features(filled),df.energy_usage-filled.previous_usage,sample_weight=sw)
        if self.energy_augment:
            rng=np.random.default_rng(self.random_seed+241)
            copies=[df.copy()]
            for k in range(self.energy_augment):
                q=df.copy()
                mask=test_features[NUM].isna().to_numpy()[rng.integers(0,len(test_features),len(q))]
                q[NUM]=q[NUM].mask(mask)
                copies.append(q)
            data=pd.concat(copies,ignore_index=True)
            augmented=self.fill(data)
            weights=np.tile(sw,len(copies))
            self.ridge=make_pipeline(StandardScaler(),Ridge(alpha=self.raw_alpha*len(copies)))
            self.ridge.fit(self.regression_features(augmented,data),data.energy_usage,
                ridge__sample_weight=weights)
            self.tree=HistGradientBoostingRegressor(categorical_features='from_dtype',
                max_iter=tree.n_iter_,learning_rate=.03,max_leaf_nodes=15,min_samples_leaf=30,
                l2_regularization=float(len(copies)),early_stopping=False,random_state=42)
            self.tree.fit(tree_features(augmented),data.energy_usage-augmented.previous_usage,
                sample_weight=weights)
        else:
            self.ridge=selector
            self.tree=tree
        return self


class ConditionalImputedEnergy(CatImputedBaseline):
    def __init__(self, conditional_depth=4, conditional_iterations=750, **kwargs):
        super().__init__(**kwargs)
        self.conditional_depth=conditional_depth
        self.conditional_iterations=conditional_iterations

    def fit_imputer(self, df, test_features):
        from catboost import CatBoostRegressor
        super().fit_imputer(df,test_features)
        joint=pd.concat([df.drop(columns='energy_usage',errors='ignore'),test_features],ignore_index=True)
        weights=shift_weights(joint,test_features)**self.imputer_weight
        weights/=weights.mean()
        self.conditional_models={}
        for col in ['occupancy','previous_usage']:
            ci=NUM.index(col)
            for pattern in range(16):
                if pattern & (1<<ci):
                    continue
                observed=[c for i,c in enumerate(NUM) if c!=col and not pattern & (1<<i)]
                dropped=[c for c in NUM if c not in observed]
                data=joint.dropna(subset=observed+[col])
                model=CatBoostRegressor(iterations=self.conditional_iterations,
                    depth=self.conditional_depth,learning_rate=.05,l2_leaf_reg=20.,
                    thread_count=4,random_seed=self.random_seed,verbose=False,
                    allow_writing_files=False,one_hot_max_size=20)
                model.fit(cat_features(data).drop(columns=dropped),data[col],
                    sample_weight=weights[data.index.to_numpy()],
                    cat_features=['building_id','building_type','day_of_week'])
                self.conditional_models[(col,pattern)]=(dropped,model)
        return self

    def expected(self, df, col):
        ci=NUM.index(col)
        patterns=(df[NUM].isna().to_numpy().astype(int)*(2**np.arange(4))).sum(axis=1)
        patterns=patterns & ~(1<<ci)
        pred=np.empty(len(df))
        for pattern in np.unique(patterns):
            ix=np.flatnonzero(patterns==pattern)
            dropped,model=self.conditional_models[(col,int(pattern))]
            pred[ix]=model.predict(cat_features(df.iloc[ix]).drop(columns=dropped))
        return np.maximum(pred,0.)

    def fill(self, df):
        filled=super().fill(df)
        for col in ['occupancy','previous_usage']:
            missing=df[col].isna().to_numpy()
            if missing.any():
                ix=np.flatnonzero(missing)
                filled.iloc[ix,filled.columns.get_loc(col)]=self.expected(df.iloc[ix],col)
        return filled

    def regression_features(self, filled, raw):
        x=super().regression_features(filled,raw).copy()
        if self.distill:
            for col,name in [('previous_usage','expected_previous'),('occupancy','expected_occupancy')]:
                mean=self.expected(raw,col)
                x[name]=mean
                for typ in TYPES:
                    x[f'{name}_{typ}']=mean*(raw.building_type==typ).to_numpy(float)
        return x
