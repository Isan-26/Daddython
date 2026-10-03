"""Score fixed second-round settings on the second-round reserved partition."""
import json,hashlib,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from energy_model import ROOT,NUM,EnergyModel,masked_view,metrics,shift_weights
from energy_model_v5 import EnergyModelV5


def validate():
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    dev,holdout=train_test_split(train,test_size=.2,random_state=2027,stratify=train.building_id)
    views=[masked_view(holdout,test,31000+k) for k in range(12)]
    full=pd.concat([holdout]+views)
    with warnings.catch_warnings(),threadpool_limits(limits=4):
        warnings.simplefilter('ignore')
        model=EnergyModelV5().fit(dev,test)
        old=model.baseline.predict(full)
        new=model.predict(full,baseline_prediction=old)
    n=len(holdout);m=len(views[0]);y=full.energy_usage.to_numpy()
    w=shift_weights(holdout.loc[views[0].index],test)
    def score(p):
        e=(p[n:]-y[n:])**2
        return {'ordinary':metrics(holdout,p[:n]),'test_masks_rmse':float(np.sqrt(e.mean())),
                'weighted_rmse':float(np.sqrt(np.average(e,weights=np.tile(w,len(views)))))}
    result={'baseline_file':'candidate_v4.csv','baseline_competition_rmse':3.37,
        'split_seed':2027,'holdout_rows':n,'complete_holdout_rows':m,'mask_trials':len(views),
        'note':'Held out from second-round selection. These labelled rows may have been used in the first round.',
        'config':model.config,'baseline':score(old),'candidate':score(new)}
    rng=np.random.default_rng(91)
    eo=((old[n:]-y[n:])**2).reshape(len(views),m).mean(axis=0)
    en=((new[n:]-y[n:])**2).reshape(len(views),m).mean(axis=0)
    deltas=[];weighted=[]
    for k in range(1000):
        ix=rng.integers(0,m,m)
        deltas.append(np.sqrt(eo[ix].mean())-np.sqrt(en[ix].mean()))
        weighted.append(np.sqrt(np.average(eo[ix],weights=w[ix]))-np.sqrt(np.average(en[ix],weights=w[ix])))
    result['bootstrap_gain_intervals']={'positive_favors_v5':True,
        'masked_95pct':np.quantile(deltas,[.025,.975]).tolist(),
        'weighted_95pct':np.quantile(weighted,[.025,.975]).tolist()}
    result['data_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv']}
    np.savez(ROOT/'experiments/round2/confirmation_predictions.npz',baseline=old,candidate=new,y=y,n_raw=n)
    (ROOT/'validation_v5_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':validate()
