import json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT,PatternModel,masked_view
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2026,stratify=tr.building_id)
config=json.loads((ROOT/'experiments/harmonic_results.json').read_text())['summary'][0]['config']
rows=[]
with threadpool_limits(limits=4):
 for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=79).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b];cache=np.load(ROOT/'experiments'/f'blend_fold_{fold}.npz')
  full=pd.concat([val]+[masked_view(val,te,100+fold*10+s) for s in range(3)])
  assert np.array_equal(full.energy_usage.to_numpy(),cache['y'])
  new=PatternModel(**config).fit(train,test_features=te).predict(full)
  for wc in [0,.25,.5,.75,1.]:
   for wm in [0,.25,.5,.75,1.]:
    w=np.where(cache['missing'],wm,wc);e=(new*w+cache['old']*(1-w)-cache['y'])**2;k=int(cache['n_raw'])
    rows.append({'fold':fold,'complete_weight':wc,'missing_weight':wm,
        'raw_mse':float(e[:k].mean()),'stress_mse':float(e[k:].mean()),
        'shift_mse':float(np.average(e[k:],weights=cache['weights']))})
  print('Final selection fold',fold+1,'completed',flush=True)
summary=[]
for wc in [0,.25,.5,.75,1.]:
 for wm in [0,.25,.5,.75,1.]:
  r=[v for v in rows if v['complete_weight']==wc and v['missing_weight']==wm]
  summary.append({'complete_weight':wc,'missing_weight':wm,**{label:float(np.sqrt(np.mean([v[k] for v in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('shift_rmse','shift_mse')]}})
summary.sort(key=lambda v:v['shift_rmse'])
previous=json.loads((ROOT/'experiments/blend_results.json').read_text())
if summary[0]['shift_rmse'] < previous['summary'][0]['shift_rmse']:
 best=dict(summary[0],pattern_config=config)
else:
 best=dict(previous['summary'][0],pattern_config=previous['pattern_config'])
print(json.dumps({'selected':best,'new_summary':summary[:5]},indent=2),flush=True)
(ROOT/'model_config.json').write_text(json.dumps(best,indent=2))
(ROOT/'experiments'/'final_selection_results.json').write_text(json.dumps({'selected':best,'summary':summary,'folds':rows},indent=2))
