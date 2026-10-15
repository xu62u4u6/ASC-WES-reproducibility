"""Quantify S5B alternative definitions without selecting a publication method."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pandas as pd
from .c47_profile import _as_sample_mask


def compare_definitions(case_sets):
    if not case_sets: raise ValueError('No complete cases')
    counters={name:Counter() for name in ['AST_shared','A_without_S','S_without_A','A_only','S_only','AS_without_T']}
    for sets in case_sets:
        a,t,s=(sets[p] for p in ['A','T','S'])
        groups={'AST_shared':a&t&s,'A_without_S':a-s,'S_without_A':s-a,
                'A_only':a-(t|s),'S_only':s-(a|t),'AS_without_T':(a&s)-t}
        for name,group in groups.items(): counters[name].update(group)
    n=len(case_sets)
    summaries={name:{'case_variant_occurrences':sum(c.values()),'unique_variants':len(c),
                     'variants_at_ge_10pct_cases':sum(count/n>=.1 for count in c.values())}
               for name,c in counters.items()}
    changes={}
    for legacy,strict in [('A_without_S','A_only'),('S_without_A','S_only')]:
        old={v for v,count in counters[legacy].items() if count/n>=.1}
        new={v for v,count in counters[strict].items() if count/n>=.1}
        changes[legacy+'__to__'+strict]={'retained_rows':len(old&new),'removed_rows':len(old-new),
                                        'retained_rows_with_changed_frequency':sum(counters[legacy][v]!=counters[strict][v] for v in old&new)}
    return {'cases':n,'definitions':summaries,'thresholded_row_changes':changes,
            'selection':'pending; this is a comparison, not a chosen publication definition'},counters


def audit_maf(maf,cases):
    if not {'sample_ID','Variant_Classification'}.issubset(maf.columns): raise ValueError('Missing MAF columns')
    if maf.index.isna().any() or maf[['sample_ID','Variant_Classification']].isna().any().any():
        raise ValueError('Missing mutation identifiers/classifications')
    if cases.case_ID.duplicated().any(): raise ValueError('Duplicate case identifiers')
    selected=cases.loc[_as_sample_mask(cases.AS_sample),'case_ID'].tolist()
    if len(selected)!=11 or pd.isna(selected).any(): raise ValueError('Expected 11 complete cases')
    # Match legacy pivot_table(aggfunc=first); annotations are part of set keys.
    pivot=maf.pivot_table(index=maf.index,columns='sample_ID',values='Variant_Classification',aggfunc='first').fillna(False)
    case_sets=[]
    for case in selected:
        parts={}
        for part in ['A','T','S']:
            key=f'{case}_{part}'
            if key not in pivot: raise ValueError('Missing component sample')
            col=pivot[key];present=col.loc[col.ne(False)]
            parts[part]=set(present.index.astype(str)+'|'+present.astype(str))
        case_sets.append(parts)
    report,counters=compare_definitions(case_sets)
    report['variant_filter']='All supplied MAF classifications; no nonsynonymous filter in legacy S5B'
    report['key']='Existing mutation key plus Variant_Classification; duplicate key/sample uses first annotation'
    report['gene_claims']={}
    for gene,name in [('EGFR','AST_shared'),('RPS3A','A_without_S'),('HLA-DQB1','S_without_A')]:
        report['gene_claims'][gene+'__'+name]=sorted(count for key,count in counters[name].items()
            if key.split('|')[0]==gene and key.split('|')[-1]=='Missense_Mutation' and count>=2)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    maf=pd.read_csv(args.data_root/'data/WES/all_case_maf_AST.csv',sep='\t',index_col=0)
    cases=pd.read_csv(args.data_root/'data/ASC_all_case_metadata_0312.csv')
    print(json.dumps(audit_maf(maf,cases),indent=2))


if __name__=='__main__':main()
