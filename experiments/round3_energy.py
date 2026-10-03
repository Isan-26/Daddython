import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_model import ROOT,NUM,masked_view,shift_weights
from energy_model_v5 import EnergyModelV5
from round3_models import MaskTrainedEnergy

tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2028,stratify=tr.building_id)
base=json.loads((ROOT/'model_v5_config.json').read_text())['config']
configs=[dict(base,energy_augment=a,energy_weight=w,missing_flags=m,probability_events=p) for a,w,m,p in [
 (0,.5,False,False),(0,1,False,False),(1,.5,False,False),
 (1,1,True,False),(1,1.5,True,False),(3,1,True,False),
 (3,1,True,True),(3,.5,True,True)]]
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=812).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  full=pd.concat([val]+[masked_view(val,te,42000+fold*10+k) for k in range(4)])
  base_model=EnergyModelV5().fit(train,te);old=base_model.predict(full)
  y=full.energy_usage.to_numpy();n=len(val)
  weights=np.concatenate([shift_weights(val.loc[v.index],te) for v in [masked_view(val,te,42000+fold*10+k) for k in range(4)]])
  np.savez(ROOT/'experiments/round3'/f'baseline_fold_{fold}.npz',prediction=old,y=y,n_raw=n,weights=weights,indices=full.index.to_numpy())
  for ci,config in enumerate(configs):
   model=MaskTrainedEnergy(**config).fit(train,te);pred=model.predict(full)
   np.savez(ROOT/'experiments/round3'/f'energy_fold_{fold}_model_{ci}.npz',prediction=pred)
   for blend in [.25,.5,.75,1.]:
    error=(blend*pred+(1-blend)*old-y)**2
    rows.append({'fold':fold,'config':config,'blend':blend,'raw_mse':float(error[:n].mean()),
      'stress_mse':float(error[n:].mean()),'weighted_mse':float(np.average(error[n:],weights=weights))})
  print('Third-round energy fold',fold+1,'complete',flush=True)
summary=[]
for c in configs:
 for blend in [.25,.5,.75,1.]:
  r=[v for v in rows if v['config']==c and v['blend']==blend]
  summary.append({'config':c,'blend':blend,**{label:float(np.sqrt(np.mean([x[k] for x in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','weighted_mse')]}})
summary.sort(key=lambda v:v['weighted_rmse'])
baselines=[]
for f in range(3):
 q=np.load(ROOT/'experiments/round3'/f'baseline_fold_{f}.npz');n=int(q['n_raw']);e=(q['prediction']-q['y'])**2
 baselines.append({'raw_mse':e[:n].mean(),'stress_mse':e[n:].mean(),'weighted_mse':np.average(e[n:],weights=q['weights'])})
bs={label:float(np.sqrt(np.mean([x[k] for x in baselines]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','weighted_mse')]}
result={'baseline':bs,'summary':summary,'folds':rows,'configs':configs,'split_seed':2028}
print(json.dumps({'baseline':bs,'best':summary[:8]},indent=2),flush=True)
(ROOT/'experiments/round3/energy_results.json').write_text(json.dumps(result,indent=2))
