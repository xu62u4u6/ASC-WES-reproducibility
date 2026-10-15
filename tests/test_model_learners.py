import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from asc_reproduction.model_learners import fit_base_learners, random_forest, BRANCHES


def test_base_fit_matches_original_settings_and_ignores_nontraining_rows():
    rng = np.random.default_rng(8)
    snv = rng.integers(0, 2, (50, 13)).astype(float)
    cnv = rng.normal(size=(50, 12))
    y = np.tile([0, 1], 25)
    snv[:40, -1] = 0  # Only non-training samples carry this feature.
    snv[:40, -2] = 0
    snv[:2, -2] = 1  # Exactly 5%, must be retained.
    model = fit_base_learners(snv[:40], cnv[:40], y[:40], seed=3)
    assert 12 not in model.selected['snv_top']
    assert 11 in model.selected['snv_top']
    top = np.flatnonzero(snv[:40].mean(axis=0) >= .05)
    for modality, x in [('snv', snv[:40, top]), ('cnv', cnv[:40])]:
        original = RFECV(random_forest(42), step=10, min_features_to_select=10,
                         cv=StratifiedKFold(5, shuffle=False), scoring='accuracy', n_jobs=1).fit(x,y[:40])
        expected = np.flatnonzero(original.support_)
        if modality == 'snv': expected = top[expected]
        np.testing.assert_array_equal(model.selected[modality+'_rfe'], expected)
        pd.testing.assert_frame_equal(model.curves[modality], pd.DataFrame(
            {k:v for k,v in original.cv_results_.items() if np.asarray(v).ndim==1}))
    before = model.probabilities(snv[40:], cnv[40:])
    for branch in BRANCHES:
        x = snv if branch.startswith('snv') else cnv
        ids = model.selected[branch]
        expected = random_forest(3).fit(x[:40,ids],y[:40]).predict_proba(x[40:,ids])[:,1]
        np.testing.assert_array_equal(before[branch], expected)
    snv[40:] = 1-snv[40:]
    cnv[40:] += 100
    y[40:] = 1-y[40:]
    after = fit_base_learners(snv[:40], cnv[:40], y[:40], seed=3)
    for name in BRANCHES:
        np.testing.assert_array_equal(model.selected[name],after.selected[name])
        np.testing.assert_array_equal(model.models[name].feature_importances_,after.models[name].feature_importances_)
    with pytest.raises(ValueError, match='schema'):
        model.probabilities(snv[40:,:-1],cnv[40:])


def test_training_validation():
    with pytest.raises(ValueError, match='binary'):
        fit_base_learners(np.full((20,12),.5),np.ones((20,12)),np.tile([0,1],10),seed=0)
    with pytest.raises(ValueError, match='ten candidate'):
        fit_base_learners(np.zeros((20,12)),np.ones((20,12)),np.tile([0,1],10),seed=0)
