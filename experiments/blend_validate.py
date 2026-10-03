import json,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT,NUM,PatternModel,masked_view,shift_weights
from notebook_baseline import NotebookBaseline
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2026,stratify=tr.building_id)
config=json.loads((ROOT/'experiments/tree_results.json').read_text())['summary'][0]['config']
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=79).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  views=[val]+[masked_view(val,te,100+fold*10+s) for s in range(3)]
  old=NotebookBaseline().fit(train);new=PatternModel(**config).fit(train,test_features=te)
  full=pd.concat(views);p_old=old.predict(full);p_new=new.predict(full)
  missing=full[NUM].isna().any(axis=1).to_numpy()
  weights=np.concatenate([shift_weights(val.loc[v.index],te) for v in views[1:]])
  y=full.energy_usage.to_numpy();k=len(val)
  np.savez(ROOT/'experiments'/f'blend_fold_{fold}.npz',old=p_old,new=p_new,y=y,missing=missing,weights=weights,n_raw=k)
  for wc in [0,.25,.5,.75,1.]:
   for wm in [0,.25,.5,.75,1.]:
    w=np.where(missing,wm,wc);e=(p_new*w+p_old*(1-w)-y)**2
    rows.append({'fold':fold,'complete_weight':wc,'missing_weight':wm,
        'raw_mse':float(e[:k].mean()),'stress_mse':float(e[k:].mean()),
        'shift_mse':float(np.average(e[k:],weights=weights))})
  print('Blend fold',fold+1,'completed',flush=True)
summary=[]
for wc in [0,.25,.5,.75,1.]:
 for wm in [0,.25,.5,.75,1.]:
  r=[v for v in rows if v['complete_weight']==wc and v['missing_weight']==wm]
  summary.append({'complete_weight':wc,'missing_weight':wm,**{label:float(np.sqrt(np.mean([v[k] for v in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('shift_rmse','shift_mse')]}})
summary.sort(key=lambda v:v['shift_rmse'])
print(json.dumps(summary[:8],indent=2),flush=True)
(ROOT/'experiments'/'blend_results.json').write_text(json.dumps({'pattern_config':config,'summary':summary,'folds':rows},indent=2))
