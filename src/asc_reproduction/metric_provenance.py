"""Audit processed TMB/MSI/HRD against supplied summaries without exporting IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import read_pivot_sqlite

CAPTURE_MB = {"LUAD": 66.0, "ASC": 41.6, "LUSC": 50.0}
HRD_COLUMNS = ["HRD", "Telomeric AI", "LST", "HRD-sum"]
NONSYNONYMOUS = (
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
    "Translation_Start_Site", "Nonstop_Mutation",
)
TRANSITIONS = ('A→G', 'C→T', 'G→A', 'T→C')
BASE_CHANGES = tuple(a+'→'+b for a in 'ACGT' for b in 'ACGT' if a != b)


def audit_titv(metadata, maf):
    """Count all SNP records, including synonymous; no multi-base splitting."""
    if metadata.empty or metadata.index.has_duplicates or maf.sample_ID.isna().any():
        raise ValueError('Invalid Ti/Tv sample keys')
    if set(maf.sample_ID) != set(metadata.index):
        raise ValueError('Ti/Tv MAF sample set differs from processed cohort')
    if maf.Variant_Type.isna().any():
        raise ValueError('Missing variant type in Ti/Tv input')
    snps = maf.loc[maf.Variant_Type.eq('SNP')].copy()
    if (not snps.Reference_Allele.isin(list('ACGT')).all()
            or not snps.Tumor_Seq_Allele2.isin(list('ACGT')).all()
            or snps.Reference_Allele.eq(snps.Tumor_Seq_Allele2).any()):
        raise ValueError('SNP record has invalid single-base substitution')
    snps['change'] = snps.Reference_Allele+'→'+snps.Tumor_Seq_Allele2
    counts = pd.crosstab(snps.sample_ID, snps.change).reindex(
        index=metadata.index, columns=list(BASE_CHANGES), fill_value=0)
    ti = counts[list(TRANSITIONS)].sum(axis=1)
    tv = counts.drop(columns=list(TRANSITIONS)).sum(axis=1)
    if tv.le(0).any():
        raise ValueError('Ti/Tv undefined: no transversions')
    saved = numeric(metadata, [*BASE_CHANGES, 'ti/tv'])
    count_error = float((counts-saved[list(BASE_CHANGES)]).abs().to_numpy().max())
    ratio_error = float((ti/tv-saved['ti/tv']).abs().max())
    if count_error != 0 or ratio_error > 1e-12:
        raise ValueError('MAF Ti/Tv recount disagrees with processed fields')
    return {'sample_count': len(metadata), 'SNP_records': len(snps),
            'transition_records': int(ti.sum()), 'transversion_records': int(tv.sum()),
            'max_base_change_count_error': count_error, 'max_ratio_error': ratio_error,
            'scope': 'All SNP-class records; no nonsynonymous filter; no multi-base splitting'}


def audit_maf_counts(metadata, maf):
    """Count records, not unique genes or unique variants; preserve legacy rule."""
    if metadata.index.has_duplicates or metadata.empty:
        raise ValueError("Expected unique nonempty processed sample keys")
    if maf[['sample_ID', 'Variant_Classification']].isna().any().any():
        raise ValueError("MAF count keys/classifications contain missing values")
    if set(maf.sample_ID) != set(metadata.index):
        raise ValueError("MAF sample set differs from processed cohort")
    selected = maf.loc[maf.Variant_Classification.isin(NONSYNONYMOUS)]
    counts = selected.groupby('sample_ID').size().reindex(metadata.index, fill_value=0)
    saved = numeric(metadata, ['mutations_count', 'capture_size', 'TMB'])
    if saved.capture_size.le(0).any():
        raise ValueError("Capture size must be positive")
    count_error = float((counts-saved.mutations_count).abs().max())
    tmb_error = float((counts/saved.capture_size-saved.TMB).abs().max())
    if count_error != 0 or tmb_error > 1e-12:
        raise ValueError("Recounted nonsynonymous MAF records disagree with processed TMB")
    return {'sample_count': len(metadata), 'all_MAF_rows': len(maf),
            'nonsynonymous_MAF_rows': len(selected), 'classes': list(NONSYNONYMOUS),
            'count_unit': 'MAF records; no deduplication or mutated-gene collapsing',
            'max_count_error': count_error, 'max_TMB_error': tmb_error}


def numeric(frame, columns):
    array = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(array.to_numpy()).all():
        raise ValueError("Metric inputs contain missing/nonfinite values")
    return array


def audit(metadata, msi, hrd):
    if any(frame.index.has_duplicates for frame in (metadata, msi, hrd)):
        raise ValueError("Duplicate metric sample keys")
    if metadata.empty or not metadata.index.isin(msi.index).all() or not metadata.index.isin(hrd.index).all():
        raise ValueError("Upstream tables do not cover all processed samples")
    expected = metadata.subtype.map(CAPTURE_MB)
    if expected.isna().any():
        raise ValueError("Unknown capture-size subtype")
    values = numeric(metadata, ["TMB", "mutations_count", "capture_size", "MSI", *HRD_COLUMNS])
    if not np.array_equal(values.capture_size, expected):
        raise ValueError("Capture sizes differ from located source definition")
    if (values.mutations_count.lt(0) | values.mutations_count.mod(1).ne(0)).any():
        raise ValueError("Mutation counts must be nonnegative integers")
    msi_values = numeric(msi.reindex(metadata.index), ["%"])
    hrd_values = numeric(hrd.reindex(metadata.index), HRD_COLUMNS)
    errors = {
        "TMB_count_divided_by_capture": float((values.TMB-values.mutations_count/values.capture_size).abs().max()),
        "MSI_upstream_percent": float((values.MSI-msi_values['%']).abs().max()),
        "HRD_sum_components": float((values['HRD-sum']-values[HRD_COLUMNS[:3]].sum(axis=1)).abs().max()),
    }
    errors.update({"upstream_"+c: float((values[c]-hrd_values[c]).abs().max()) for c in HRD_COLUMNS})
    if any(error > 1e-12 for error in errors.values()):
        raise ValueError("Processed genomic metrics disagree with supplied sources")
    return {"scope": "processed-field arithmetic and supplied-table consistency, not upstream refit",
            "sample_count": len(metadata), "capture_mb_by_subtype": CAPTURE_MB,
            "max_absolute_errors": errors, "patient_identifiers_written": False}


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": Path(path).stat().st_size}


def run(database, msi_csv, hrd_csv, output, *, maf=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Metric provenance output must be new")
    paths = {"processed_database": database, "MSI_summary": msi_csv, "HRD_summary": hrd_csv}
    if maf is not None:
        paths['MAF_records'] = maf
    before = {k: fingerprint(p) for k, p in paths.items()}
    metadata = read_pivot_sqlite(database).sample_metadata
    msi = pd.read_csv(msi_csv, index_col=0).set_index('new_sample_ID')
    hrd = pd.read_csv(hrd_csv, index_col='sample_ID')
    result = audit(metadata, msi, hrd)
    if maf is not None:
        records = pd.read_csv(maf, sep='\t', usecols=['sample_ID', 'Variant_Classification',
            'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2'])
        result['MAF_recount'] = audit_maf_counts(metadata, records)
        result['Ti_Tv_recount'] = audit_titv(metadata, records)
    else:
        result['MAF_recount'] = {'status': 'not performed; no MAF supplied'}
        result['Ti_Tv_recount'] = {'status': 'not performed; no MAF supplied'}
    if before != {k: fingerprint(p) for k, p in paths.items()}:
        raise ValueError("Input changed during metric audit")
    result['inputs'] = before
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--msi-csv', type=Path, required=True)
    parser.add_argument('--hrd-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maf', type=Path, help='Optional all-cohort MAF to recount nonsynonymous records and SNP Ti/Tv')
    args = parser.parse_args()
    run(args.database, args.msi_csv, args.hrd_csv, args.output, maf=args.maf)


if __name__ == '__main__':
    main()
