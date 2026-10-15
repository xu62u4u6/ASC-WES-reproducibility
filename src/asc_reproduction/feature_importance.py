"""Selection-aware S20 summaries; never infer missing measurements as zero."""

import numpy as np
import pandas as pd


def summarize_importance(records, selections, *, expected_folds, features):
    """Summarize ONE model's importance over a declared complete CV grid.

    records has seed/fold/feature/importance columns. selections maps each
    (seed, fold) to its selected feature names, independently of records.
    Every selected feature requires an observed finite nonnegative importance.
    Only features explicitly absent from a completed fold's selection get zero.
    Callers must verify checkpoint provenance before passing selections.
    Returns feature-by-seed means and a ranked overall summary. No renormalizing.
    """
    folds = list(expected_folds)
    names = list(features)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError('Expected folds must be nonempty and unique')
    if not names or len(set(names)) != len(names):
        raise ValueError('Feature universe must be nonempty and unique')
    if set(selections) != set(folds):
        raise ValueError('Missing or unexpected fold selection manifest')
    sizes = pd.Series([seed for seed, _ in folds]).value_counts()
    if sizes.nunique() != 1:
        raise ValueError('Seeds must have equal numbers of folds')
    required = ['seed', 'fold', 'feature', 'importance']
    if not set(required).issubset(records.columns):
        raise ValueError('Missing importance columns')
    data = records[required].copy()
    if data.isna().any().any() or data.duplicated(required[:3]).any():
        raise ValueError('Missing values or duplicate importance records')
    values = data.importance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid feature importance')
    allowed = set(names)
    expected = set()
    for (seed, fold), selected in selections.items():
        selected = list(selected)
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= allowed:
            raise ValueError('Invalid selected feature manifest')
        expected.update((seed, fold, feature) for feature in selected)
    observed = set(data[required[:3]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError('Importance records do not match selected features; cannot zero-fill')
    # Sparse sums avoid constructing a genes × 500-fold dense table.
    seeds = sorted(sizes.index)
    seed_means = (data.groupby(['feature', 'seed']).importance.sum().unstack('seed')
                  .reindex(index=names, columns=seeds).fillna(0.0))
    seed_means = seed_means.div(sizes.reindex(seeds), axis='columns')
    summary = pd.DataFrame(index=pd.Index(names, name='feature'))
    summary['mean_importance'] = seed_means.mean(axis=1)
    summary['selected_folds'] = data.groupby('feature').size().reindex(names, fill_value=0)
    summary['total_folds'] = len(folds)
    summary['selection_frequency'] = summary.selected_folds / len(folds)
    summary = summary.sort_values('mean_importance', ascending=False, kind='stable')
    return seed_means.loc[summary.index], summary
