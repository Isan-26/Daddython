"""V5: covariate-trained imputers and typical-load features blended with v4."""
from pathlib import Path
import hashlib,json,warnings
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from energy_model import ROOT,EnergyModel
from experiments.round2_models import CatImputedBaseline

class EnergyModelV5:
    def __init__(self,config=None):
        self.config=config or json.loads((ROOT/'model_v5_config.json').read_text())
    def fit(self,train,test_features,fit_baseline=True):
        # Imputation targets are observed feature values; no test energy targets.
        inputs=test_features.drop(columns='energy_usage',errors='ignore')
        self.new_model=CatImputedBaseline(**self.config['config']).fit(train,inputs)
        self.baseline=EnergyModel().fit(train,inputs) if fit_baseline else None
        return self
    def predict(self,df,baseline_prediction=None):
        if baseline_prediction is None:
            if self.baseline is None:raise ValueError('Supply baseline predictions or fit the baseline.')
            baseline_prediction=self.baseline.predict(df)
        baseline_prediction=np.asarray(baseline_prediction,float)
        if len(baseline_prediction)!=len(df):raise ValueError('Baseline row count differs.')
        new=self.new_model.predict(df)
        w=self.config['blend']
        return (1-w)*baseline_prediction+w*new


def write_submission(output_path=ROOT/'candidate_v5.csv'):
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    config=json.loads((ROOT/'model_v5_config.json').read_text())
    baseline_path=ROOT/config['baseline_file']
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=config['baseline_sha256']:
        raise ValueError('The confirmed baseline file has changed; verify it before blending.')
    baseline=pd.read_csv(baseline_path)
    if not baseline.id.equals(test.id):raise ValueError('Baseline IDs differ from test IDs.')
    with threadpool_limits(limits=4):
        model=EnergyModelV5(config).fit(train,test,fit_baseline=False)
        pred=model.predict(test,baseline_prediction=baseline.prediction)
    if not np.isfinite(pred).all():raise ValueError('Non-finite prediction.')
    result=pd.DataFrame({'id':test.id,'prediction':pred})
    result.to_csv(output_path,index=False)
    return result

if __name__=='__main__':
    out=write_submission()
    print(f'Saved {len(out):,} rows to {ROOT / "candidate_v5.csv"}')
