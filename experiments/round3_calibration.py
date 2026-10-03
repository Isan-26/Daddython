"""Cross-fit a small bias correction by missing inputs and building type."""
import json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import Ridge
from energy_model import ROOT,NUM,masked_view
from experiments.round2_models import TYPES

def calibration_features(df):
    m=df[NUM].isna().to_numpy(float)
    arrays=[m]
    arrays.append(np.column_stack([m[:,i]*m[:,j] for i in range(4) for j in range(i+1,4)]))
    types=(df.building_type.to_numpy()[:,None]==np.array(TYPES)[None,:]).astype(float)
    arrays.append((m[:,:,None]*types[:,None,:]).reshape(len(df),-1))
    return np.column_stack(arrays)

if __name__=='__main__':
 tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
 dev,_=train_test_split(tr,test_size=.2,random_state=2028,stratify=tr.building_id)
 gate=json.loads((ROOT/'experiments/round3/gated_results.json').read_text())['summary'][0]
 data=[]
 for f,(_,b) in enumerate(KFold(3,shuffle=True,random_state=812).split(dev)):
  val=dev.iloc[b];full=pd.concat([val]+[masked_view(val,te,42000+f*10+k) for k in range(4)])
  q=np.load(ROOT/'experiments/round3'/f'baseline_fold_{f}.npz');n=int(q['n_raw'])
  p=np.load(ROOT/'experiments/round3'/f"conditional_fold_{f}_model_{gate['source']['index']}.npz")['prediction']
  count=full[NUM].isna().sum(axis=1).to_numpy()
  w=np.select([count==1,count>=2],gate['weights_by_missing_count'][1:],default=0.)
  predicted=q['prediction']+w*(p-q['prediction'])
  data.append({'x':calibration_features(full),'y':q['y'],'pred':predicted,'old':q['prediction'],
   'weights':np.concatenate([np.zeros(n),q['weights']]),'n_raw':n})
 rows=[]
 for alpha in [100.,500.,1500.,5000.]:
  for scale in [.5,1.]:
   errs=[];weight=[];raw=[]
   for f,d in enumerate(data):
    others=[z for k,z in enumerate(data) if k!=f]
    x=np.concatenate([z['x'] for z in others]);res=np.concatenate([z['y']-z['pred'] for z in others])
    sw=np.concatenate([z['weights'] for z in others]);positive=sw>0
    model=Ridge(alpha=alpha,fit_intercept=False).fit(x[positive],res[positive],sample_weight=sw[positive])
    pred=d['pred']+scale*model.predict(d['x'])
    e=(pred-d['y'])**2;n=d['n_raw'];errs.extend(e[n:]);raw.extend(e[:n]);weight.extend(d['weights'][n:])
   rows.append({'alpha':alpha,'scale':scale,'raw_rmse':float(np.sqrt(np.mean(raw))),
     'test_masks_rmse':float(np.sqrt(np.mean(errs))),
     'weighted_rmse':float(np.sqrt(np.average(errs,weights=weight)))})
 rows.sort(key=lambda v:v['weighted_rmse']);chosen=rows[0]
 x=np.concatenate([z['x'] for z in data]);res=np.concatenate([z['y']-z['pred'] for z in data]);sw=np.concatenate([z['weights'] for z in data]);positive=sw>0
 model=Ridge(alpha=chosen['alpha'],fit_intercept=False).fit(x[positive],res[positive],sample_weight=sw[positive])
 result={'gate':gate,'summary':rows,'selected':chosen,'coefficients':model.coef_.tolist(),
  'note':'Calibration hyperparameters were compared by leaving each development fold out. The base-model OOF predictions are shared across this stage; reserved third-round holdout is the main validation.'}
 print(json.dumps({'selected':chosen,'summary':rows},indent=2),flush=True)
 (ROOT/'experiments/round3/calibration_results.json').write_text(json.dumps(result,indent=2))
