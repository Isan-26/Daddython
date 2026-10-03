"""Use only the development partition to select a model."""
import json, time
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT,NUM,PatternModel,masked_view

tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2026,stratify=tr.building_id)
configs=[dict(harmonics=h,alpha=a,extra=e,tree_weight=t) for h,a,e,t in [
 (1,.1,'none',0),(1,1,'none',0),(2,.1,'none',0),(2,1,'none',0),(2,3,'none',0),
 (3,1,'none',0),(2,1,'weekend',0),(3,1,'weekend',0),
 (2,1,'interactions',0),(2,1,'hinges',0),
 (1,1,'none',.35),(2,1,'none',.35),(2,3,'none',.35),
 (3,1,'none',.35),(2,1,'weekend',.35),(2,1,'interactions',.35),
 (2,1,'hinges',.35),(2,1,'none',.65)]]
rows=[]
with threadpool_limits(limits=4):
 for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=79).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  views=[masked_view(val,te,100+fold*10+s) for s in range(3)]
  patterns=range(16)
  for config in configs:
   m=PatternModel(**config).fit(train,patterns)
   ps=[m.predict(v) for v in views]
   errors=np.concatenate([(p-v.energy_usage.to_numpy())**2 for p,v in zip(ps,views)])
   raw=m.predict(val)-val.energy_usage.to_numpy()
   row={'fold':fold,'config':config,'raw_mse':float(np.mean(raw**2)), 'stress_mse':float(errors.mean())}
   rows.append(row)
  print('Fold',fold+1,'completed',flush=True)
summary=[]
for config in configs:
 r=[v for v in rows if v['config']==config]
 summary.append({'config':config,'raw_rmse':float(np.sqrt(np.mean([v['raw_mse'] for v in r]))),
                 'test_masks_rmse':float(np.sqrt(np.mean([v['stress_mse'] for v in r]))),
                 'fold_stress_rmse':[round(np.sqrt(v['stress_mse']),4) for v in r]})
summary.sort(key=lambda v:v['test_masks_rmse'])
print(json.dumps(summary,indent=2),flush=True)
(ROOT/'experiments'/'cv_results.json').write_text(json.dumps({'summary':summary,'folds':rows},indent=2))
