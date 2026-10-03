import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_model import ROOT,masked_view
from round2_models import CatImputedBaseline
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2028,stratify=tr.building_id)
base=json.loads((ROOT/'model_v5_config.json').read_text())['config']
seeds=[1223,2041,3301]
rows=[]
with warnings.catch_warnings(),threadpool_limits(limits=4):
 warnings.simplefilter('ignore')
 for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=812).split(dev)):
  train=dev.iloc[a];val=dev.iloc[b]
  full=pd.concat([val]+[masked_view(val,te,42000+fold*10+k) for k in range(4)])
  q=np.load(ROOT/'experiments/round3'/f'baseline_fold_{fold}.npz');old=q['prediction'];y=q['y'];n=int(q['n_raw']);weights=q['weights']
  predictions=[]
  for seed in seeds:
   model=CatImputedBaseline(**base,random_seed=seed).fit(train,te);pred=model.predict(full)
   predictions.append(pred)
   np.savez(ROOT/'experiments/round3'/f'bag_fold_{fold}_seed_{seed}.npz',prediction=pred)
  for count in [1,2,3]:
   new=np.mean(predictions[:count],axis=0)
   for blend in [.25,.5,.75,1.]:
    e=(blend*new+(1-blend)*old-y)**2
    rows.append({'fold':fold,'seeds':seeds[:count],'config':base,'blend':blend,
     'raw_mse':float(e[:n].mean()),'stress_mse':float(e[n:].mean()),
     'weighted_mse':float(np.average(e[n:],weights=weights))})
  print('Imputer-averaging fold',fold+1,'complete',flush=True)
summary=[]
for count in [1,2,3]:
 for blend in [.25,.5,.75,1.]:
  r=[v for v in rows if v['seeds']==seeds[:count] and v['blend']==blend]
  summary.append({'seeds':seeds[:count],'config':base,'blend':blend,**{label:float(np.sqrt(np.mean([x[k] for x in r]))) for label,k in [('raw_rmse','raw_mse'),('test_masks_rmse','stress_mse'),('weighted_rmse','weighted_mse')]}})
summary.sort(key=lambda v:v['weighted_rmse'])
print(json.dumps(summary[:8],indent=2),flush=True)
(ROOT/'experiments/round3/bagging_results.json').write_text(json.dumps({'summary':summary,'folds':rows},indent=2))
