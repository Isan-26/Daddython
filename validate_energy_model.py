"""Confirm the fixed model on rows never used in development selection.

The repeated masking trials share validation rows; uncertainty is estimated by
resampling rows, rather than treating repeated views as independent examples.
"""
from pathlib import Path
import hashlib,json,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from threadpoolctl import threadpool_limits
from energy_model import ROOT,NUM,EnergyModel,masked_view,shift_weights,metrics


def validate():
    train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv')
    development,confirmation=train_test_split(train,test_size=.2,random_state=2026,stratify=train.building_id)
    views=[masked_view(confirmation,test,11000+k) for k in range(10)]
    with warnings.catch_warnings(),threadpool_limits(limits=4):
        warnings.simplefilter('ignore')
        model=EnergyModel().fit(development,test)
        old,new=model.predict_components(pd.concat([confirmation]+views))
    missing=pd.concat([confirmation]+views)[NUM].isna().any(axis=1).to_numpy()
    cfg=model.config
    blend_weights=np.where(missing,cfg['missing_weight'],cfg['complete_weight'])
    selected=(1-blend_weights)*old+blend_weights*new
    n=len(confirmation);m=len(views[0])
    ys=np.concatenate([v.energy_usage.to_numpy() for v in views])
    source=confirmation.loc[views[0].index]
    weights=shift_weights(source,test)
    repeated_weights=np.tile(weights,len(views))
    def evaluate(prediction):
        err=(prediction[n:]-ys)**2
        return {'ordinary_validation':metrics(confirmation,prediction[:n]),
                'test_masks_rmse':float(np.sqrt(err.mean())),
                'building_event_weighted_rmse':float(np.sqrt(np.average(err,weights=repeated_weights)))}
    result={'split':{'seed':2026,'development_rows':len(development),'confirmation_rows':n,
                    'complete_confirmation_rows':m,'masking_trials':len(views)},
            'config':cfg,'baseline':evaluate(old),'candidate':evaluate(selected)}
    rng=np.random.default_rng(89)
    old_errors=((old[n:]-ys)**2).reshape(len(views),m).mean(axis=0)
    new_errors=((selected[n:]-ys)**2).reshape(len(views),m).mean(axis=0)
    deltas=[];weighted_deltas=[]
    for k in range(1000):
        ix=rng.integers(0,m,m)
        deltas.append(np.sqrt(old_errors[ix].mean())-np.sqrt(new_errors[ix].mean()))
        weighted_deltas.append(np.sqrt(np.average(old_errors[ix],weights=weights[ix]))-
                               np.sqrt(np.average(new_errors[ix],weights=weights[ix])))
    result['paired_row_bootstrap']={'meaning':'positive values favor candidate model',
            'test_masks_gain_95pct_interval':np.quantile(deltas,[.025,.975]).tolist(),
            'weighted_gain_95pct_interval':np.quantile(weighted_deltas,[.025,.975]).tolist()}
    result['data_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv']}
    (ROOT/'validation_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    validate()
