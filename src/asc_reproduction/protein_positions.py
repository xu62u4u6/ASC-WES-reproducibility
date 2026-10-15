"""Deterministic S12 protein-position counts; no patient-level exports."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .driver_oncoplot import TYPES
from .dataio import read_pivot_sqlite


def protein_counts(maf,gene):
    selected=maf.loc[maf.Hugo_Symbol.eq(gene)&maf.Variant_Classification.isin(TYPES)].copy()
    if selected.empty: raise ValueError('No selected nonsynonymous variants')
    def start(value):
        if pd.isna(value): return None
        try: return int(str(value).split('/')[0].split('-')[0])
        except ValueError: return None
    positions=selected.Protein_position.map(start)
    raw=selected.Protein_position.dropna()
    try: length=int(str(raw.iloc[0]).split('/')[-1])
    except (IndexError,ValueError): raise ValueError('Protein length unavailable') from None
    valid=positions.notna()
    if (positions.loc[valid]<1).any() or (positions.loc[valid]>length).any(): raise ValueError('Protein position outside length')
    selected['position']=positions
    counts=(selected.loc[valid].groupby(['position','Variant_Classification']).size()
            .reset_index(name='count').rename(columns={'Variant_Classification':'type'}))
    counts['position']=counts.position.astype(int)
    stats={'gene':gene,'protein_length':length,'nonsynonymous_rows':len(selected),
           'plotted_rows':int(valid.sum()),'omitted_position_rows':int((~valid).sum()),
           'count_unit':'MAF rows grouped by range-start amino-acid position and classification'}
    return counts,stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args();meta=read_pivot_sqlite(args.data_root/'data/WES/all_case_table.db').sample_metadata
    reports=[]
    for cohort in ['LUAD','ASC','LUSC']:
        maf=pd.read_csv(args.data_root/f'data/WES/{cohort}.maf',sep='\t')
        expected=set(meta.index[meta.subtype.eq(cohort)&meta.sample_type.eq('T')])
        if set(maf.sample_ID)!=expected: raise ValueError('MAF cohort differs from T-sample cohort')
        for gene in ['EGFR','TP53']:
            counts,stats=protein_counts(maf,gene);stats['cohort']=cohort;stats['samples']=len(expected)
            stats['position_class_groups']=len(counts);reports.append(stats)
    print(json.dumps({'panels':'S12A/B','audit':reports,'scope':'Position/count audit only; domain provenance and rendering pending'},indent=2))


if __name__=='__main__':main()
