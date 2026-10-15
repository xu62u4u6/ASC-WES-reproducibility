"""Training-only TOP/RFECV and base learners with the frozen experiment settings."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

BRANCHES = ("snv_top", "snv_rfe", "cnv_all", "cnv_rfe")
COMBINATIONS = {
    "SNV_TOP-CNV_ALL": ("snv_top", "cnv_all"),
    "SNV_TOP-CNV_RFE": ("snv_top", "cnv_rfe"),
    "SNV_RFE-CNV_ALL": ("snv_rfe", "cnv_all"),
    "SNV_RFE-CNV_RFE": ("snv_rfe", "cnv_rfe"),
}


def random_forest(seed):
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)


def select_rfe(x, y):
    selector = RFECV(random_forest(42), step=10, min_features_to_select=10,
                     cv=StratifiedKFold(5, shuffle=False), scoring="accuracy", n_jobs=1)
    selector.fit(x, y)
    curve = pd.DataFrame({k: v for k, v in selector.cv_results_.items()
                          if np.asarray(v).ndim == 1})
    return np.flatnonzero(selector.support_), curve


@dataclass
class BaseFit:
    selected: dict
    curves: dict
    models: dict
    feature_counts: dict

    def probabilities(self, snv, cnv):
        matrices = {"snv": np.asarray(snv, dtype=float), "cnv": np.asarray(cnv, dtype=float)}
        for modality, x in matrices.items():
            if x.ndim != 2 or x.shape[1] != self.feature_counts[modality] or not np.isfinite(x).all():
                raise ValueError("Prediction matrix schema differs from training")
        if len(matrices["snv"]) != len(matrices["cnv"]):
            raise ValueError("Prediction modalities must have aligned rows")
        with threadpool_limits(limits=1):
            return {name: self.models[name].predict_proba(
                matrices["snv" if name.startswith("snv") else "cnv"][:, self.selected[name]])[:, 1]
                for name in BRANCHES}


def fit_base_learners(snv_train, cnv_train, y_train, *, seed):
    """Receive training rows only: no holdout, ASC or legacy feature-list argument.

    TOP is fitted once per supplied training split before RFECV, matching the
    recorded experiment. This does not claim TOP is refit inside RFECV's own
    feature-count-selection CV. Outer/stacking evaluation splits remain excluded.
    """
    snv = np.asarray(snv_train, dtype=float)
    cnv = np.asarray(cnv_train, dtype=float)
    y = np.asarray(y_train)
    if snv.ndim != 2 or cnv.ndim != 2 or y.ndim != 1 or len(snv) != len(cnv) or len(y) != len(snv):
        raise ValueError("Training matrix shapes are inconsistent")
    if not np.isfinite(snv).all() or not np.isfinite(cnv).all() or not np.isin(snv, [0, 1]).all():
        raise ValueError("Training requires finite CNV and binary SNV")
    if set(y) != {0, 1} or min(np.count_nonzero(y == v) for v in (0, 1)) < 5:
        raise ValueError("RFECV requires both classes with at least five training rows each")
    top = np.flatnonzero(snv.mean(axis=0) >= .05)
    if len(top) < 10 or cnv.shape[1] < 10:
        raise ValueError("Fewer than ten candidate features")
    with threadpool_limits(limits=1):
        s, scurve = select_rfe(snv[:, top], y)
        c, ccurve = select_rfe(cnv, y)
        selected = {"snv_top": top, "snv_rfe": top[s],
                    "cnv_all": np.arange(cnv.shape[1]), "cnv_rfe": c}
        models = {}
        for name in BRANCHES:
            matrix = snv if name.startswith("snv") else cnv
            models[name] = random_forest(seed).fit(matrix[:, selected[name]], y)
    return BaseFit(selected, {"snv": scurve, "cnv": ccurve}, models,
                   {"snv": snv.shape[1], "cnv": cnv.shape[1]})
