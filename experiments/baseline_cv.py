import json,warnings,time
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT,NUM,masked_view,shift_weights

n=json.loads((ROOT/'linearregression.ipynb').read_text())
source=''.join(n['cells'][35]['source']).replace('n_jobs=-1','n_jobs=4').replace('fit_imputer(pd.concat([train_df, pred_df]))','fit_imputer(train_df)')
exec(compile(source,'notebook_v3_baseline','exec'))
dev,_=train_test_split(train_full,test_size=.2,random_state=2026,stratify=train_full.building_id)
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=79).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  views=[masked_view(val,test_full,100+fold*10+s) for s in range(3)]
  prediction=fit_and_predict(train,pd.concat([val]+views))
  raw=prediction[:len(val)]-val.energy_usage.to_numpy();offset=len(val)
  errors=[];weights=[]
  for v in views:
   errors.append((prediction[offset:offset+len(v)]-v.energy_usage.to_numpy())**2)
   weights.append(shift_weights(val.loc[v.index],test_full));offset+=len(v)
  row={'fold':fold,'raw_mse':float(np.mean(raw**2)),'stress_mse':float(np.concatenate(errors).mean()),
        'shift_mse':float(np.average(np.concatenate(errors),weights=np.concatenate(weights)))}
  print(json.dumps(row),flush=True);rows.append(row)
summary={label:float(np.sqrt(np.mean([r[k] for r in rows]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('shift_rmse','shift_mse')]}
print(json.dumps(summary),flush=True)
(ROOT/'experiments'/'baseline_cv_results.json').write_text(json.dumps({'summary':summary,'folds':rows},indent=2))
