import json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
from pattern_experiments import ROOT,PatternModel,masked_view,shift_weights
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2026,stratify=tr.building_id)
base=json.loads((ROOT/'experiments/tree_results.json').read_text())['summary'][0]['config']
configs=[dict(base,harmonics=h) for h in [2,3,4,6]]
rows=[]
with threadpool_limits(limits=4):
 for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=79).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  views=[masked_view(val,te,100+fold*10+s) for s in range(3)]
  weights=[shift_weights(val.loc[v.index],te) for v in views]
  for config in configs:
   m=PatternModel(**config).fit(train,test_features=te)
   e=np.concatenate([(m.predict(v)-v.energy_usage.to_numpy())**2 for v in views])
   rows.append({'fold':fold,'config':config,'stress_mse':float(e.mean()),
    'shift_mse':float(np.average(e,weights=np.concatenate(weights)))})
  print('Harmonic fold',fold+1,'completed',flush=True)
summary=[]
for c in configs:
 r=[v for v in rows if v['config']==c]
 summary.append({'config':c,'test_masks_rmse':float(np.sqrt(np.mean([v['stress_mse'] for v in r]))),
       'shift_rmse':float(np.sqrt(np.mean([v['shift_mse'] for v in r])))})
summary.sort(key=lambda v:v['shift_rmse'])
print(json.dumps(summary,indent=2),flush=True)
(ROOT/'experiments'/'harmonic_results.json').write_text(json.dumps({'summary':summary,'folds':rows},indent=2))
