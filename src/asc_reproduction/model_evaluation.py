"""Evaluate all eight candidates on one identical development-only outer fold."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

from .model_learners import random_forest
from .model_splits import outer_folds
from .model_stacking import fit_stacking

MODALITIES = {"SNV-TOP": "snv", "CNV-gene": "gene", "CNV-arm": "arm", "CNV-cluster": "cnv"}


@dataclass
class StandaloneFit:
    models: dict
    selected: dict
    widths: dict

    def predict(self, matrices):
        result = {}
        row_count = None
        with threadpool_limits(limits=1):
            for name, modality in MODALITIES.items():
                x = np.asarray(matrices[modality], dtype=float)
                if x.ndim != 2 or x.shape[1] != self.widths[modality] or not np.isfinite(x).all():
                    raise ValueError("Standalone prediction schema differs from training")
                if row_count is not None and len(x) != row_count:
                    raise ValueError("Prediction modalities have different row counts")
                row_count = len(x)
                model = self.models[name]
                x = x[:, self.selected[name]]
                result[name] = {"probability": model.predict_proba(x)[:,1], "predicted": model.predict(x)}
        return result


def fit_standalone(matrices, y, *, seed):
    y = np.asarray(y)
    if y.ndim != 1 or set(y) != {0,1}:
        raise ValueError("Standalone training requires binary labels")
    models, selected, widths = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, modality in MODALITIES.items():
            x = np.asarray(matrices[modality], dtype=float)
            if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
                raise ValueError("Invalid standalone training matrix")
            if modality == "snv" and not np.isin(x, [0,1]).all():
                raise ValueError("Standalone SNV must be binary")
            ids = np.flatnonzero(x.mean(axis=0) >= .05) if modality == "snv" else np.arange(x.shape[1])
            if not len(ids):
                raise ValueError("No standalone candidate features")
            selected[name] = ids
            widths[modality] = x.shape[1]
            models[name] = random_forest(seed).fit(x[:,ids],y)
    return StandaloneFit(models, selected, widths)


def evaluate_development_fold(data, split, *, seed, fold):
    """Subset development before calling any learner. Results stay in memory.

    Predictions are temporary verification data, not an export format. The caller
    must not serialize this dictionary as a public/patient-level artifact.
    """
    if not isinstance(seed, int) or not 0 <= seed < 100 or not isinstance(fold, int) or not 0 <= fold < 5:
        raise ValueError("Expected seed0–99 and fold0–4")
    if not np.array_equal(data['patients'], split.metadata.index.to_numpy(dtype=str)) or not np.array_equal(data['y'],split.labels):
        raise ValueError("Model data and partition cohort differ")
    dev = {k: np.asarray(data[k])[split.development] for k in MODALITIES.values()}
    y = split.labels[split.development]
    train, test = list(outer_folds(split,seed))[fold]
    training = {k:v[train] for k,v in dev.items()}
    testing = {k:v[test] for k,v in dev.items()}
    standalone = fit_standalone(training,y[train],seed=seed)
    from .model_importance import collect_importance
    importance = [collect_importance(standalone, data, seed=seed, fold=fold,
                                    scope='standalone', inner_fold=-1)]
    def record_inner(index, fit):
        importance.append(collect_importance(fit, data, seed=seed, fold=fold,
                                            scope='stack_inner', inner_fold=index))
    stack = fit_stacking(training['snv'],training['cnv'],y[train],seed=seed,
                        on_inner_fit=record_inner)
    importance.append(collect_importance(stack.base, data, seed=seed, fold=fold,
                                        scope='stack_full', inner_fold=-1))
    predictions = standalone.predict(testing)
    predictions.update(stack.predict(testing['snv'],testing['cnv']))
    rows = []
    for name, result in predictions.items():
        rows.append(dict(model=name,seed=seed,fold=fold,
                         acc=accuracy_score(y[test],result['predicted']),
                         f1=f1_score(y[test],result['predicted'],average='macro'),
                         auc=roc_auc_score(y[test],result['probability'])))
    return {"metrics":pd.DataFrame(rows), "predictions":predictions,
            "importance":pd.concat(importance, ignore_index=True),
            "test_positions":test, "standalone":standalone, "stack":stack}
