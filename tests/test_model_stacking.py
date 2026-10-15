import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from asc_reproduction import model_stacking as module
from asc_reproduction.model_learners import BRANCHES, COMBINATIONS


def test_stacking_oof_refits_selection_and_meta_matches(monkeypatch):
    # Encode row identity in CNV to audit every call without using patient IDs.
    y = np.tile([0, 1], 20)
    snv = np.zeros((40,12))
    cnv = np.column_stack([np.arange(40),np.zeros((40,11))])
    calls = []
    class FakeFit:
        def __init__(self, train):
            self.train = set(train)
            self.selected = {name: np.array([min(train)]) for name in BRANCHES}
        def probabilities(self, s, c):
            rows = c[:,0].astype(int)
            assert self.train.isdisjoint(rows)
            return {name: (rows % (i+3))/(i+3) for i,name in enumerate(BRANCHES)}
    def fit(s,c,labels,*,seed):
        rows=c[:,0].astype(int)
        np.testing.assert_array_equal(labels,y[rows])
        calls.append(rows)
        return FakeFit(rows)
    monkeypatch.setattr(module,'fit_base_learners',fit)
    result=module.fit_stacking(snv,cnv,y,seed=9)
    assert len(calls)==6
    assert all(len(rows)==32 for rows in calls[:5])
    assert len(calls[5])==40
    for i,name in enumerate(BRANCHES):
        np.testing.assert_array_equal(result.oof[name],(np.arange(40)%(i+3))/(i+3))
    for name,(s,c) in COMBINATIONS.items():
        expected=LogisticRegression(max_iter=1000,random_state=9).fit(
            np.column_stack([result.oof[s],result.oof[c]]),y)
        np.testing.assert_array_equal(result.meta_models[name].coef_,expected.coef_)
    assert len(result.inner_selected)==5


def test_invalid_oof_predictions_rejected(monkeypatch):
    class BadFit:
        def probabilities(self,s,c):
            return {name:np.full(len(s),np.nan) for name in BRANCHES}
    monkeypatch.setattr(module,'fit_base_learners',lambda *args,**kwargs:BadFit())
    with pytest.raises(ValueError,match='Invalid inner'):
        module.fit_stacking(np.ones((40,12)),np.ones((40,12)),np.tile([0,1],20),seed=0)
