import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_model import ROOT,masked_view
from round3_models import ConditionalImputedEnergy
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2028,stratify=tr.building_id)
base=json.loads((ROOT/'model_v5_config.json').read_text())['config']
configs=[dict(base,conditional_depth=d,conditional_iterations=n) for d,n in [(4,750),(5,1000)]]
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=812).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  full=pd.concat([val]+[masked_view(val,te,42000+fold*10+k) for k in range(4)])
  q=np.load(ROOT/'experiments/round3'/f'baseline_fold_{fold}.npz')
  assert np.array_equal(q['y'],full.energy_usage.to_numpy())
  old=q['prediction'];y=q['y'];n=int(q['n_raw']);weights=q['weights']
  for ci,c in enumerate(configs):
   model=ConditionalImputedEnergy(**c).fit(train,te);pred=model.predict(full)
   np.savez(ROOT/'experiments/round3'/f'conditional_fold_{fold}_model_{ci}.npz',prediction=pred)
   for blend in [.25,.5,.75,1.]:
    error=(blend*pred+(1-blend)*old-y)**2
    rows.append({'fold':fold,'config':c,'blend':blend,'raw_mse':float(error[:n].mean()),
     'stress_mse':float(error[n:].mean()),'weighted_mse':float(np.average(error[n:],weights=weights))})
  print('Conditional-imputer fold',fold+1,'complete',flush=True)
summary=[]
for c in configs:
 for blend in [.25,.5,.75,1.]:
  r=[v for v in rows if v['config']==c and v['blend']==blend]
  summary.append({'config':c,'blend':blend,**{label:float(np.sqrt(np.mean([x[k] for x in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','weighted_mse')]}})
summary.sort(key=lambda v:v['weighted_rmse'])
print(json.dumps(summary[:5],indent=2),flush=True)
(ROOT/'experiments/round3/conditional_results.json').write_text(json.dumps({'summary':summary,'folds':rows,'configs':configs},indent=2))
