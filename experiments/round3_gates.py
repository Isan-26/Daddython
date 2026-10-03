"""Choose two coarse blend weights using development out-of-fold errors."""
import json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from energy_model import ROOT,NUM,masked_view
tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
dev,_=train_test_split(tr,test_size=.2,random_state=2028,stratify=tr.building_id)
base=json.loads((ROOT/'model_v5_config.json').read_text())['config']
sources=[]
for name,kind in [('energy','mask'),('conditional','conditional')]:
 r=json.loads((ROOT/'experiments/round3'/f'{name}_results.json').read_text())
 for i,c in enumerate(r['configs']):sources.append({'kind':kind,'config':c,'index':i,'prefix':name})
for seeds in [[1223],[1223,2041],[1223,2041,3301]]:
 sources.append({'kind':'bag','config':base,'seeds':seeds})
fold_data=[]
for f,(_,b) in enumerate(KFold(3,shuffle=True,random_state=812).split(dev)):
 val=dev.iloc[b];full=pd.concat([val]+[masked_view(val,te,42000+f*10+k) for k in range(4)])
 q=np.load(ROOT/'experiments/round3'/f'baseline_fold_{f}.npz');n=int(q['n_raw'])
 count=full[NUM].isna().sum(axis=1).to_numpy()
 weights=np.concatenate([np.zeros(n),q['weights']])
 fold_data.append({'y':q['y'],'old':q['prediction'],'weights':weights,'count':count,'n_raw':n})
summary=[]
for source in sources:
 ds=[]
 for f,d in enumerate(fold_data):
  if source['kind']=='bag':
   pred=np.mean([np.load(ROOT/'experiments/round3'/f'bag_fold_{f}_seed_{s}.npz')['prediction'] for s in source['seeds']],axis=0)
  else:
   pred=np.load(ROOT/'experiments/round3'/f"{source['prefix']}_fold_{f}_model_{source['index']}.npz")['prediction']
  ds.append(dict(d,pred=pred))
 learned=[0.]
 for group in [1,2]:
  numerator=denominator=0.
  for d in ds:
   mask=(d['count']==1) if group==1 else (d['count']>=2)
   delta=d['pred']-d['old'];w=d['weights']*mask
   numerator+=np.sum(w*delta*(d['y']-d['old']))
   denominator+=np.sum(w*delta**2)
  learned.append(float(np.clip(numerator/denominator,0,1)) if denominator else 0.)
 errors=[];weights=[];raw_errors=[]
 for d in ds:
  gate=np.select([d['count']==1,d['count']>=2],[learned[1],learned[2]],default=0.)
  e=(d['old']+gate*(d['pred']-d['old'])-d['y'])**2;n=d['n_raw']
  raw_errors.extend(e[:n]);errors.extend(e[n:]);weights.extend(d['weights'][n:])
 summary.append({'source':source,'weights_by_missing_count':learned,
   'raw_rmse':float(np.sqrt(np.mean(raw_errors))),
   'test_masks_rmse':float(np.sqrt(np.mean(errors))),
   'weighted_rmse':float(np.sqrt(np.average(errors,weights=weights)))})
summary.sort(key=lambda v:v['weighted_rmse'])
print(json.dumps(summary[:8],indent=2),flush=True)
(ROOT/'experiments/round3/gated_results.json').write_text(json.dumps({'summary':summary},indent=2))
