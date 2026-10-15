"""Deterministic patient-disjoint classifier partitions; no identifier exports."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class ReferenceSplit:
    # IDs are retained only in memory for matrix alignment.
    metadata: pd.DataFrame
    labels: np.ndarray
    development: np.ndarray
    holdout: np.ndarray


def reference_split(metadata: pd.DataFrame) -> ReferenceSplit:
    """Sort reference sample IDs before the historical stratified seed-42 split.

    Reject repeat patients rather than silently splitting their samples. ASC is
    excluded before splitting. Indices address the returned reference metadata.
    """
    required = {"case_ID", "subtype", "sample_type"}
    if not required.issubset(metadata.columns) or metadata.index.has_duplicates:
        raise ValueError("Invalid classifier sample metadata schema")
    reference = metadata.loc[metadata.subtype.isin(["LUAD", "LUSC"])].sort_index().copy()
    if reference.empty or reference[list(required)].isna().any().any():
        raise ValueError("Reference metadata is empty or incomplete")
    if reference.index.isna().any() or (reference.index.astype(str).str.strip() == "").any():
        raise ValueError("Reference sample keys must be nonempty")
    if (reference.case_ID.astype(str).str.strip() == "").any():
        raise ValueError("Reference patient keys must be nonempty")
    if reference.case_ID.duplicated().any():
        raise ValueError("Multiple reference samples per patient require an explicit grouping policy")
    if not reference.sample_type.eq("T").all():
        raise ValueError("Classifier reference contains non-T samples")
    asc_cases = set(metadata.loc[metadata.subtype.eq("ASC"), "case_ID"])
    if asc_cases.intersection(reference.case_ID):
        raise ValueError("Patient overlaps reference and ASC cohorts")
    y = reference.subtype.eq("LUSC").to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Both reference classes are required")
    development, holdout = train_test_split(
        np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
    return ReferenceSplit(reference, y, np.sort(development), np.sort(holdout))


def outer_folds(split: ReferenceSplit, seed: int):
    """Yield development-local positions, never full-cohort positions."""
    y = split.labels[split.development]
    yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def stacking_folds(development_labels: np.ndarray, outer_train: np.ndarray):
    """Return development-local train/validation indices within one outer train."""
    outer_train = np.asarray(outer_train)
    if outer_train.ndim != 1 or len(np.unique(outer_train)) != len(outer_train):
        raise ValueError("Outer training positions must be unique and one-dimensional")
    if not np.issubdtype(outer_train.dtype, np.integer):
        raise ValueError("Outer training positions must be integers")
    if np.any(outer_train < 0) or np.any(outer_train >= len(development_labels)):
        raise ValueError("Outer training positions are outside development data")
    y = np.asarray(development_labels)[outer_train]
    for train, validation in StratifiedKFold(5, shuffle=False).split(np.zeros(len(y)), y):
        yield outer_train[train], outer_train[validation]
