"""Keyed frozen-model inputs and score alignment shared by figure analyses."""

import numpy as np
import pandas as pd

from .cohort_inputs import as_sample_mask


def keyed_matrices(data: dict, *, asc: bool) -> dict[str, pd.DataFrame]:
    """Retain sample keys while aligning the two frozen-model modalities."""
    identifiers = data["asc_samples" if asc else "patients"]
    prefix = "asc_" if asc else ""
    return {
        modality: pd.DataFrame(
            data[prefix + modality],
            index=identifiers,
            columns=data[modality + "_features"],
        )
        for modality in ("snv", "cnv")
    }


def select_score_panels(reference, asc, metadata):
    """Select 200 T samples and 11 complete ASC triples by explicit keys.

    Reject duplicate case/components and disagreement between AS_sample and
    complete-triplet selection. Return two in-memory plotting frames.
    """
    scores = pd.concat([reference, asc])
    if metadata.index.has_duplicates:
        raise ValueError("Metadata sample keys are duplicated")
    cases = metadata.case_ID.astype("string")
    if cases.isna().any() or cases.str.strip().eq("").any():
        raise ValueError("Missing or blank case identifiers")
    if scores.index.has_duplicates or not scores.index.isin(metadata.index).all():
        raise ValueError("Score sample keys are duplicated or missing")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Soft scores must be finite and within [0, 1]")
    frame = metadata.loc[scores.index].copy()
    frame["soft_score"] = scores
    subtype = frame.loc[frame.sample_type.eq("T")].copy()
    if subtype.subtype.value_counts().to_dict() != {
        "LUAD": 89,
        "LUSC": 82,
        "ASC": 29,
    }:
        raise ValueError("Figure 5C requires 89/29/82 T samples")
    asc_frame = frame.loc[frame.subtype.eq("ASC")].copy()
    if not asc_frame.sample_type.isin(["A", "T", "S"]).all():
        raise ValueError("Unexpected ASC component type")
    if asc_frame.duplicated(["case_ID", "sample_type"]).any():
        raise ValueError("Duplicate ASC case/component")
    pivot = (
        asc_frame.pivot(index="case_ID", columns="sample_type", values="soft_score")
        .reindex(columns=["A", "T", "S"])
        .dropna()
        .sort_index()
    )
    if len(pivot) != 11:
        raise ValueError("Figure 5D requires 11 complete A/T/S cases")
    paired = asc_frame.loc[asc_frame.case_ID.isin(pivot.index)].copy()
    flags = as_sample_mask(asc_frame.AS_sample)
    if len(paired) != 33:
        raise ValueError("Figure 5D requires exactly 33 paired samples")
    if set(asc_frame.index[flags]) != set(paired.index):
        raise ValueError("AS_sample and complete-triplet membership differ")
    paired = paired.sort_values(["case_ID", "sample_type"])
    for part in ("A", "T", "S"):
        ids = paired.loc[paired.sample_type.eq(part), "case_ID"].tolist()
        if ids != pivot.index.tolist():
            raise ValueError("Component vectors have different patient order")
    shared = paired.index[paired.sample_type.eq("T")]
    pd.testing.assert_series_equal(
        subtype.loc[shared, "soft_score"], paired.loc[shared, "soft_score"]
    )
    return subtype, paired
