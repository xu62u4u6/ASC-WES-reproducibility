"""Capture impurity-based RF importance without retaining patient data or refitting."""

import numpy as np
import pandas as pd

from .model_evaluation import MODALITIES
from .model_learners import BRANCHES

COLUMNS = ['seed', 'fold', 'scope', 'inner_fold', 'model', 'feature', 'importance']


def collect_importance(fit, data, *, seed, fold, scope, inner_fold):
    rows = []
    for name, ids in fit.selected.items():
        modality = MODALITIES[name] if scope == 'standalone' else ('snv' if name.startswith('snv') else 'cnv')
        names = np.asarray(data[modality + '_features'])[ids]
        values = np.asarray(fit.models[name].feature_importances_, dtype=float)
        if values.shape != names.shape:
            raise ValueError('Importance length differs from selected feature names')
        rows.extend((seed, fold, scope, inner_fold, name, str(feature), float(value))
                    for feature, value in zip(names, values))
    return pd.DataFrame(rows, columns=COLUMNS)


def validate_importance(frame, selections, seed, fold):
    if list(frame.columns) != COLUMNS or frame.empty or frame.isna().any().any():
        raise ValueError('Incomplete importance table')
    if not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all():
        raise ValueError('Importance partition differs')
    keys = ['scope', 'inner_fold', 'model', 'feature']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate importance records')
    expected = set()
    groups = [('standalone', -1, selections['standalone']), ('stack_full', -1, selections['full'])]
    groups += [('stack_inner', i, group) for i, group in enumerate(selections['inner'])]
    for scope, inner, group in groups:
        if set(group) != (set(MODALITIES) if scope == 'standalone' else set(BRANCHES)):
            raise ValueError('Importance model branches differ')
        expected.update((scope, inner, name, feature) for name, features in group.items() for feature in features)
    if set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Importance coverage differs from selection manifest')
    values = frame.importance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Invalid importance values')
    totals = frame.groupby(keys[:3]).importance.sum().to_numpy()
    # sklearn's no-split forest is allowed to have all-zero importance.
    if not np.all(np.isclose(totals, 1, atol=1e-8) | np.isclose(totals, 0, atol=1e-12)):
        raise ValueError('Importance sums must equal one (or all-zero forest)')
