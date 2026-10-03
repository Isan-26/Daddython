import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_model import ROOT,NUM,EnergyModel,masked_view,shift_weights
from round2_models import PooledExperts
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,confirmation=train_test_split(tr,test_size=.2,random_state=2027,stratify=tr.building_id)
configs=[dict(harmonics=h,alpha=a,building_scale=b,type_scale=t,hour_scale=u,prev_scale=p,tree_weight=w,weight_power=.5) for h,a,b,t,u,p,w in [
 (3,1,.35,1,.5,1,0),(3,1,.35,1,.5,1,.35),(3,1,.35,1,.5,1,.65),
 (3,1,.2,1,.5,.3,.35),(3,1,.6,1,.5,.3,.35),(3,1,.35,1,.5,0,.35),
 (2,1,.35,1,.5,1,.35),(3,3,.6,1,.5,1,.35),(3,.3,.35,1,.5,1,.35),
 (3,1,.35,1,0,1,.35),(3,1,0,1,0,0,.35),(3,1,.35,0,1,1,.35)]]
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=811).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  views=[val]+[masked_view(val,te,20270+fold*10+k) for k in range(4)]
  full=pd.concat(views);base=EnergyModel().fit(train,te).predict(full)
  y=full.energy_usage.to_numpy();n=len(val)
  weights=np.concatenate([shift_weights(val.loc[v.index],te) for v in views[1:]])
  np.savez(ROOT/'experiments/round2'/f'baseline_fold_{fold}.npz',pred=base,y=y,weights=weights,n_raw=n,indices=full.index.to_numpy())
  for config in configs:
   m=PooledExperts(**config).fit(train,test_features=te);pred=m.predict(full)
   for blend in [.5,1.]:
    e=(blend*pred+(1-blend)*base-y)**2
    rows.append({'fold':fold,'config':config,'blend':blend,'raw_mse':float(e[:n].mean()),
     'stress_mse':float(e[n:].mean()),'shift_mse':float(np.average(e[n:],weights=weights))})
  print('Pooling fold',fold+1,'complete',flush=True)
summary=[]
for c in configs:
 for blend in [.5,1.]:
  r=[x for x in rows if x['config']==c and x['blend']==blend]
  summary.append({'config':c,'blend':blend,**{label:float(np.sqrt(np.mean([x[k] for x in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','shift_mse')]}})
summary.sort(key=lambda v:v['weighted_rmse'])
base=[]
for f in range(3):
 q=np.load(ROOT/'experiments/round2'/f'baseline_fold_{f}.npz');n=int(q['n_raw']);e=(q['pred']-q['y'])**2
 base.append({'raw_mse':e[:n].mean(),'stress_mse':e[n:].mean(),'shift_mse':np.average(e[n:],weights=q['weights'])})
bs={label:float(np.sqrt(np.mean([x[k] for x in base]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','shift_mse')]}
result={'baseline':bs,'summary':summary,'folds':rows,'confirmation_seed':2027}
print(json.dumps({'baseline':bs,'best':summary[:6]},indent=2),flush=True)
(ROOT/'experiments/round2/pooling_results.json').write_text(json.dumps(result,indent=2))
