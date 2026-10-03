"""Compare the notebook's v3 model using training-only preprocessing."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT, NUM, masked_view, metrics

n=json.loads((ROOT/'linearregression.ipynb').read_text())
source=''.join(n['cells'][35]['source'])
source=source.replace('n_jobs=-1','n_jobs=4').replace('fit_imputer(pd.concat([train_df, pred_df]))','fit_imputer(train_df)')
exec(compile(source,'notebook_v3_baseline','exec'))
dev,confirmation=train_test_split(train_full,test_size=.2,random_state=2026,stratify=train_full.building_id)
train,val=train_test_split(dev,test_size=.25,random_state=42,stratify=dev.building_id)
stress=masked_view(val,test_full,10)
with warnings.catch_warnings(), threadpool_limits(limits=4):
    warnings.simplefilter('ignore')
    prediction=fit_and_predict(train,pd.concat([val,stress]))
result={'description':'Current v3 architecture, preprocessing fitted only on training fold.',
        'raw':metrics(val,prediction[:len(val)]),'test_masks':metrics(stress,prediction[len(val):])}
print(json.dumps(result),flush=True)
(ROOT/'experiments'/'baseline_results.json').write_text(json.dumps(result,indent=2))
