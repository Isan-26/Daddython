import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_model import ROOT,NUM,EnergyModel,masked_view,shift_weights
from round2_models import CatCorrection
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2027,stratify=tr.building_id)
configs=[dict(depth=d,iterations=n,l2=l,augment=a,weight_power=w) for d,n,l,a,w in [
 (4,1000,10,0,.5),(5,1000,10,1,.5),(6,1100,15,1,.5),
 (5,1000,20,3,.5),(5,1000,10,1,0),(4,1600,20,3,.5)]]
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=811).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  full=pd.concat([val]+[masked_view(val,te,20270+fold*10+k) for k in range(4)])
  cache=np.load(ROOT/'experiments/round2'/f'baseline_fold_{fold}.npz')
  assert np.array_equal(full.energy_usage.to_numpy(),cache['y'])
  y=cache['y'];base=cache['pred'];n=int(cache['n_raw']);weights=cache['weights']
  for config in configs:
   m=CatCorrection(**config).fit(train,test_features=te);p,c=m.predict_components(full)
   for correction in [.25,.5,.75,1.]:
    pred=p+correction*c
    for blend in [.5,1.]:
     e=(blend*pred+(1-blend)*base-y)**2
     rows.append({'fold':fold,'config':config,'correction_weight':correction,'blend':blend,
      'raw_mse':float(e[:n].mean()),'stress_mse':float(e[n:].mean()),
      'shift_mse':float(np.average(e[n:],weights=weights))})
  print('CatBoost fold',fold+1,'complete',flush=True)
summary=[]
for c in configs:
 for correction in [.25,.5,.75,1.]:
  for blend in [.5,1.]:
   r=[x for x in rows if x['config']==c and x['correction_weight']==correction and x['blend']==blend]
   summary.append({'config':c,'correction_weight':correction,'blend':blend,**{label:float(np.sqrt(np.mean([x[k] for x in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','shift_mse')]}})
summary.sort(key=lambda v:v['weighted_rmse'])
print(json.dumps(summary[:8],indent=2),flush=True)
(ROOT/'experiments/round2/cat_results.json').write_text(json.dumps({'summary':summary,'folds':rows},indent=2))
