import numpy as np
import pandas as pd
import pytest

from asc_reproduction.model_splits import reference_split, outer_folds, stacking_folds


def metadata():
    n = 171
    return pd.DataFrame({"case_ID": [f"synthetic-{i:03}" for i in range(n)],
                         "subtype": ["LUAD"]*89+["LUSC"]*82,
                         "sample_type": "T"}, index=[f"sample-{i:03}" for i in range(n)])


def test_split_order_and_nested_boundaries():
    frame = metadata()
    first = reference_split(frame)
    shuffled = reference_split(frame.sample(frac=1, random_state=9))
    assert first.metadata.equals(shuffled.metadata)
    np.testing.assert_array_equal(first.development, shuffled.development)
    assert np.bincount(first.labels[first.development]).tolist() == [71, 65]
    assert np.bincount(first.labels[first.holdout]).tolist() == [18, 17]
    assert set(first.development).isdisjoint(first.holdout)
    for seed in range(100):
        seen = np.zeros(136, dtype=int)
        for train, test in outer_folds(first, seed):
            seen[test] += 1
            assert set(train).isdisjoint(test)
            inner_seen = np.zeros(136, dtype=int)
            for inner_train, validation in stacking_folds(first.labels[first.development], train):
                assert set(inner_train).isdisjoint(validation)
                assert set(inner_train).isdisjoint(test)
                assert set(validation).issubset(train)
                inner_seen[validation] += 1
            assert np.all(inner_seen[train] == 1)
            assert np.all(inner_seen[test] == 0)
        assert np.all(seen == 1)


def test_reject_patient_overlap_and_exclude_asc():
    frame = metadata()
    asc = pd.DataFrame({"case_ID": ["asc-only"], "subtype": ["ASC"], "sample_type": ["A"]}, index=["asc-sample"])
    assert len(reference_split(pd.concat([frame, asc])).metadata) == 171
    asc.loc["asc-sample", "case_ID"] = frame.iloc[0].case_ID
    with pytest.raises(ValueError, match="overlaps"):
        reference_split(pd.concat([frame, asc]))
    frame.iloc[1, 0] = frame.iloc[0].case_ID
    with pytest.raises(ValueError, match="Multiple reference"):
        reference_split(frame)
