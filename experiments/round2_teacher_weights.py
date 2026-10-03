import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_model import ROOT,NUM,masked_view
from round2_models import CatImputedBaseline
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2027,stratify=tr.building_id)
base=json.loads((ROOT/'experiments/round2/refined_results.json').read_text())['summary'][0]['config']
configs=[dict(base,imputer_weight=w) for w in [0,.5,1,1.5]]
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=811).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  full=pd.concat([val]+[masked_view(val,te,20270+fold*10+k) for k in range(4)])
  q=np.load(ROOT/'experiments/round2'/f'baseline_fold_{fold}.npz');old=q['pred'];y=q['y'];n=int(q['n_raw']);weights=q['weights']
  for ci,config in enumerate(configs):
   m=CatImputedBaseline(**config).fit(train,test_features=te);p=m.predict(full)
   np.savez(ROOT/'experiments/round2'/f'weighted_teacher_fold_{fold}_model_{ci}.npz',prediction=p)
   for blend in [.25,.5,.75,1.]:
    e=(blend*p+(1-blend)*old-y)**2
    rows.append({'fold':fold,'config':config,'blend':blend,'raw_mse':float(e[:n].mean()),
      'stress_mse':float(e[n:].mean()),'shift_mse':float(np.average(e[n:],weights=weights))})
  print('Teacher weighting fold',fold+1,'complete',flush=True)
summary=[]
for c in configs:
 for blend in [.25,.5,.75,1.]:
  r=[v for v in rows if v['config']==c and v['blend']==blend]
  summary.append({'config':c,'blend':blend,**{label:float(np.sqrt(np.mean([x[k] for x in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','shift_mse')]}})
summary.sort(key=lambda v:v['weighted_rmse'])
print(json.dumps(summary[:8],indent=2),flush=True)
(ROOT/'experiments/round2/teacher_weight_results.json').write_text(json.dumps({'summary':summary,'folds':rows,'configs':configs},indent=2))
