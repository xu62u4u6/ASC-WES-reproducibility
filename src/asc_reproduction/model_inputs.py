"""Rebuild classifier matrices from processed inputs, never prior experiment caches."""

from pathlib import Path

import numpy as np
import pandas as pd

from .cgc_oncoplot import group_mutation_order, mutation_presence
from .dataio import read_pivot_sqlite
from .model_splits import reference_split
from .paired_screen import cluster_mean_matrix


def prepare_model_inputs(snv, gene, arm, metadata, mapping, *, expected_clusters=114):
    """Return sample-major arrays; retain all SNV genes for fold-local selection.

    Identifiers in this in-memory result are alignment keys, not public outputs.
    Frozen CNV clustering retains the documented full-reference limitation.
    """
    split = reference_split(metadata)
    for matrix in (snv, gene, arm):
        if matrix.index.has_duplicates or matrix.columns.has_duplicates:
            raise ValueError("Duplicate classifier matrix keys")
        if set(matrix.columns) != set(metadata.index):
            raise ValueError("Classifier matrix and metadata sample sets differ")
    binary = mutation_presence(snv)
    order = group_mutation_order(binary, metadata.loc[snv.columns])
    asc_ids = metadata.loc[order].query("subtype == 'ASC'").index
    if len(asc_ids) != int(metadata.subtype.eq("ASC").sum()):
        raise ValueError("ASC ordering omitted samples")
    clusters = cluster_mean_matrix(gene, mapping, expected_cluster_count=expected_clusters)
    reference_ids = split.metadata.index
    data = {"patients": reference_ids.to_numpy(dtype=str), "y": split.labels.copy(),
            "asc_samples": asc_ids.to_numpy(dtype=str)}
    for name, matrix in (("snv", binary), ("gene", gene), ("arm", arm), ("cnv", clusters)):
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite classifier input")
        data[name+"_features"] = matrix.index.to_numpy(dtype=str)
        data[name] = matrix.loc[:, reference_ids].T.to_numpy(dtype=float)
        data["asc_"+name] = matrix.loc[:, asc_ids].T.to_numpy(dtype=float)
    return data, split


def load_model_inputs(data_root):
    root = Path(data_root)
    snv = read_pivot_sqlite(root/"data/WES/all_case_table.db")
    gene = read_pivot_sqlite(root/"data/CNV_0407/gene_cnv.db")
    arm = read_pivot_sqlite(root/"data/CNV_0407/arm_cnv.db")
    mapping = pd.read_csv(root/"data/clustering/best_k_cluster_labels.csv")
    return prepare_model_inputs(snv.values, gene.values, arm.values,
                                snv.sample_metadata, mapping)
