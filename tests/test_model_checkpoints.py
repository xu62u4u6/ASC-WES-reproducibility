import numpy as np
import pandas as pd
import pytest

from asc_reproduction.model_checkpoints import save_checkpoint, verify_checkpoint, CANDIDATES
from asc_reproduction.model_evaluation import MODALITIES
from asc_reproduction.model_learners import BRANCHES


def payload():
    return dict(metrics=pd.DataFrame([dict(model=n,seed=0,fold=0,acc=.8,f1=.8,auc=.9) for n in sorted(CANDIDATES)]),
                selections={'standalone':{n:['feature'] for n in MODALITIES},
                            'full':{n:['feature'] for n in BRANCHES},
                            'inner':[{n:['feature'] for n in BRANCHES} for _ in range(5)]},
                curves={n:pd.DataFrame({'n_features':[10],'mean_test_score':[.8],'std_test_score':[0.],
                                       **{f'split{i}_test_score':[.8] for i in range(5)}}) for n in ('snv','cnv')},
                identity={'inputs':'hash','source':'hash'},seed=0,fold=0,resources={'elapsed_seconds':1})


def test_verified_checkpoint_and_immutable_attempt(tmp_path):
    args=payload();folder=tmp_path/'fold'
    save_checkpoint(folder,**args)
    metrics=verify_checkpoint(folder,identity=args['identity'],seed=0,fold=0)
    pd.testing.assert_frame_equal(metrics,args['metrics'])
    with pytest.raises(FileExistsError): save_checkpoint(folder,**args)
    with pytest.raises(ValueError,match='identity'):
        verify_checkpoint(folder,identity={'source':'different'},seed=0,fold=0)
    (folder/'metrics.csv').write_text('corrupted')
    with pytest.raises(ValueError,match='artifact'):
        verify_checkpoint(folder,identity=args['identity'],seed=0,fold=0)


def test_reject_incomplete_or_invalid_results_before_creation(tmp_path):
    args=payload();args['metrics'].loc[0,'auc']=np.nan
    with pytest.raises(ValueError,match='metric'): save_checkpoint(tmp_path/'bad',**args)
    assert not (tmp_path/'bad').exists()


def test_invalid_rfecv_curve_rejected(tmp_path):
    args=payload();args['curves']['snv'].loc[0,'mean_test_score']=.99
    with pytest.raises(ValueError,match='summaries'): save_checkpoint(tmp_path/'bad',**args)
    assert not (tmp_path/'bad').exists()
    args=payload();args['curves']['cnv'].loc[0,'split0_test_score']=np.nan
    with pytest.raises(ValueError,match='scores'): save_checkpoint(tmp_path/'bad',**args)
    args=payload();args['selections']['inner'].pop()
    with pytest.raises(ValueError,match='Incomplete'): save_checkpoint(tmp_path/'bad',**args)
    assert not (tmp_path/'bad').exists()
