"""OOF stacking within supplied training data; evaluation is a separate operation."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .model_learners import BaseFit, BRANCHES, COMBINATIONS, fit_base_learners
from .model_splits import stacking_folds


@dataclass
class StackFit:
    base: BaseFit
    meta_models: dict
    oof: dict
    inner_selected: list

    def predict(self, snv, cnv):
        base = self.base.probabilities(snv, cnv)
        with threadpool_limits(limits=1):
            results = {}
            for name, (s, c) in COMBINATIONS.items():
                x = np.column_stack([base[s], base[c]])
                model = self.meta_models[name]
                results[name] = {"probability": model.predict_proba(x)[:, 1],
                                 "predicted": model.predict(x)}
        return results


def fit_stacking(snv_train, cnv_train, y_train, *, seed, progress=None, on_inner_fit=None):
    """Fit all four candidate stacks, with new selection in each inner fold.

    Inputs must already be restricted to one outer training set (or to the
    development set for final fitting). No evaluation labels are accepted.
    OOF arrays stay in memory; this function does not write individual records.
    """
    snv = np.asarray(snv_train)
    cnv = np.asarray(cnv_train)
    y = np.asarray(y_train)
    if snv.ndim != 2 or cnv.ndim != 2 or y.ndim != 1 or len(snv) != len(cnv) or len(y) != len(snv):
        raise ValueError("Stacking training shapes are inconsistent")
    if set(y) != {0, 1} or min(np.count_nonzero(y == v) for v in (0, 1)) < 7:
        raise ValueError("Stacking requires enough training rows for inner five-fold RFECV")
    oof = {name: np.full(len(y), np.nan) for name in BRANCHES}
    seen = np.zeros(len(y), dtype=int)
    selections = []
    for inner_index, (train, validation) in enumerate(stacking_folds(y, np.arange(len(y)))):
        if progress is not None:
            progress(f"inner_{inner_index}_fit")
        inner = fit_base_learners(snv[train], cnv[train], y[train], seed=seed)
        predictions = inner.probabilities(snv[validation], cnv[validation])
        seen[validation] += 1
        for name in BRANCHES:
            values = np.asarray(predictions[name])
            if values.shape != (len(validation),) or not np.isfinite(values).all():
                raise ValueError("Invalid inner-fold prediction")
            oof[name][validation] = values
        selections.append({name: ids.copy() for name, ids in inner.selected.items()})
        if on_inner_fit is not None:
            on_inner_fit(inner_index, inner)
    if not np.all(seen == 1) or not all(np.isfinite(values).all() for values in oof.values()):
        raise ValueError("Incomplete or overlapping stacking OOF coverage")
    with threadpool_limits(limits=1):
        meta = {name: LogisticRegression(max_iter=1000, random_state=seed).fit(
            np.column_stack([oof[s], oof[c]]), y)
                for name, (s, c) in COMBINATIONS.items()}
    if progress is not None:
        progress("full_training_fit")
    full = fit_base_learners(snv, cnv, y, seed=seed)
    return StackFit(full, meta, oof, selections)
