"""Development validation with feature-only event/building reweighting."""
import json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT,NUM,PatternModel,masked_view,shift_weights,event_groups

tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2026,stratify=tr.building_id)
configs=[dict(harmonics=h,alpha=a,extra=e,tree_weight=t,weight_power=w) for h,a,e,t,w in [
 (1,1,'none',0,0),(2,1,'none',0,0),(2,1,'weekend',0,0),(2,1,'none',.35,0),
 (2,1,'none',.35,.5),(2,1,'none',.35,1),(2,1,'none',0,.5),(2,1,'none',0,1),
 (2,3,'none',.35,.5),(2,1,'weekend',.35,.5),
 (2,1,'interactions',.35,.5),(2,1,'hinges',.35,.5),
 (2,1,'weekend',0,.5),(2,1,'weekend',0,1),
 (2,.1,'none',.35,.5),(2,1,'none',.65,.5)]]
rows=[]
with threadpool_limits(limits=4):
 for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=79).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  views=[masked_view(val,te,100+fold*10+s) for s in range(3)]
  weights=[shift_weights(val.loc[v.index],te) for v in views]
  for config in configs:
   m=PatternModel(**config).fit(train,test_features=te)
   errors=[(m.predict(v)-v.energy_usage.to_numpy())**2 for v in views]
   mse=np.average(np.concatenate(errors),weights=np.concatenate(weights))
   row={'fold':fold,'config':config,'stress_mse':float(np.concatenate(errors).mean()),'shift_mse':float(mse)}
   rows.append(row)
  print('Shift fold',fold+1,'completed',flush=True)
summary=[]
for config in configs:
 r=[v for v in rows if v['config']==config]
 summary.append({'config':config,'test_masks_rmse':float(np.sqrt(np.mean([v['stress_mse'] for v in r]))),
                 'shift_rmse':float(np.sqrt(np.mean([v['shift_mse'] for v in r]))),
                 'fold_shift_rmse':[round(np.sqrt(v['shift_mse']),4) for v in r]})
summary.sort(key=lambda v:v['shift_rmse'])
print(json.dumps(summary,indent=2),flush=True)
(ROOT/'experiments'/'shift_results.json').write_text(json.dumps({'summary':summary,'folds':rows},indent=2))
