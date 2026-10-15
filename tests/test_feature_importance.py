import pandas as pd
import pytest

from asc_reproduction.feature_importance import summarize_importance


def example():
    folds = [(0, i) for i in range(5)]
    selections = {key: ['stable'] for key in folds}
    selections[(0, 0)] = ['rare', 'stable']
    records = pd.DataFrame(
        [(0, 0, 'rare', .8)] + [(0, i, 'stable', .2) for i in range(5)],
        columns=['seed', 'fold', 'feature', 'importance'])
    return records, selections, dict(expected_folds=folds, features=['rare', 'stable', 'never'])


def test_unselected_folds_are_zero_not_excluded():
    records, selections, kwargs = example()
    matrix, summary = summarize_importance(records, selections, **kwargs)
    assert matrix.loc['rare', 0] == pytest.approx(.16)
    assert summary.loc['stable', 'mean_importance'] == pytest.approx(.2)
    assert summary.loc['rare', 'selection_frequency'] == pytest.approx(.2)
    assert summary.loc['never', 'mean_importance'] == 0
    assert summary.index[0] == 'stable'


def test_missing_selected_measurement_is_error():
    records, selections, kwargs = example()
    with pytest.raises(ValueError, match='cannot zero-fill'):
        summarize_importance(records.iloc[1:], selections, **kwargs)


def test_missing_fold_manifest_is_error():
    records, selections, kwargs = example()
    del selections[(0, 4)]
    with pytest.raises(ValueError, match='manifest'):
        summarize_importance(records, selections, **kwargs)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1])
def test_invalid_measurement_is_error(bad):
    records, selections, kwargs = example()
    records.loc[0, 'importance'] = bad
    with pytest.raises(ValueError):
        summarize_importance(records, selections, **kwargs)


def test_duplicate_measurement_is_error():
    records, selections, kwargs = example()
    with pytest.raises(ValueError, match='duplicate'):
        summarize_importance(pd.concat([records, records.iloc[:1]]), selections, **kwargs)
