import numpy as np
import pandas as pd

from asc_reproduction import model_evaluation as module
from asc_reproduction.model_learners import COMBINATIONS, random_forest
from asc_reproduction.model_splits import reference_split, outer_folds


def test_standalone_matches_legacy_construction():
    rng=np.random.default_rng(41)
    matrices={k:rng.normal(size=(30,12)) for k in ('gene','arm','cnv')}
    matrices['snv']=rng.integers(0,2,(30,12))
    matrices['snv'][:20,-1]=0
    y=np.tile([0,1],15)
    fitted=module.fit_standalone({k:v[:20] for k,v in matrices.items()},y[:20],seed=6)
    result=fitted.predict({k:v[20:] for k,v in matrices.items()})
    for name,modality in module.MODALITIES.items():
        x=matrices[modality]
        ids=np.flatnonzero(x[:20].mean(axis=0)>=.05) if modality=='snv' else np.arange(x.shape[1])
        old=random_forest(6).fit(x[:20,ids],y[:20])
        np.testing.assert_array_equal(result[name]['probability'],old.predict_proba(x[20:,ids])[:,1])
        np.testing.assert_array_equal(result[name]['predicted'],old.predict(x[20:,ids]))


def test_outer_evaluator_excludes_holdout_and_uses_same_test(monkeypatch):
    ids=[f'synthetic-{i:03}' for i in range(100)]
    meta=pd.DataFrame({'case_ID':ids,'subtype':['LUAD']*50+['LUSC']*50,'sample_type':'T'},index=ids)
    split=reference_split(meta)
    data={'patients':np.array(ids),'y':split.labels}
    for mod in module.MODALITIES.values(): data[mod]=np.arange(100)[:,None]
    tr,te=list(outer_folds(split,0))[0]
    expected_train=split.development[tr];expected_test=split.development[te]
    calls=[]
    class Fake:
        def __init__(self,names):self.names=names
        def predict(self,*matrices):
            values=matrices[0].values() if isinstance(matrices[0],dict) else matrices
            for x in values: np.testing.assert_array_equal(x[:,0],expected_test)
            return {n:{'probability':np.full(len(te),.4),'predicted':np.zeros(len(te))} for n in self.names}
    def standalone(matrices,y,*,seed):
        for x in matrices.values(): np.testing.assert_array_equal(x[:,0],expected_train)
        np.testing.assert_array_equal(y,split.labels[expected_train]);calls.append('standalone')
        return Fake(module.MODALITIES)
    def stacking(snv,cnv,y,*,seed,on_inner_fit=None):
        np.testing.assert_array_equal(snv[:,0],expected_train)
        np.testing.assert_array_equal(cnv[:,0],expected_train)
        np.testing.assert_array_equal(y,split.labels[expected_train]);calls.append('stacking')
        result = Fake(COMBINATIONS)
        result.base = object()
        return result
    monkeypatch.setattr(module,'fit_standalone',standalone)
    monkeypatch.setattr(module,'fit_stacking',stacking)
    from asc_reproduction import model_importance
    monkeypatch.setattr(model_importance,'collect_importance',lambda *a,**k:pd.DataFrame())
    result=module.evaluate_development_fold(data,split,seed=0,fold=0)
    assert calls==['standalone','stacking']
    assert len(result['metrics'])==8
    assert set(expected_train).isdisjoint(split.holdout)
    assert set(expected_test).isdisjoint(split.holdout)
