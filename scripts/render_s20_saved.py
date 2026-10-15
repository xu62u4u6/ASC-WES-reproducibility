"""Re-render verified saved S20 matrices only; no fitting or data-root access."""
import argparse
import json
from pathlib import Path

import pandas as pd
from standalone_importance_full import render, MODALITIES
from asc_reproduction.provenance import file_hash, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    record=json.loads((args.source/'COMPLETE.json').read_text())
    if record['status']!='succeeded' or record['completed_folds']!=500:
        raise ValueError('Source must be complete 500-fold results')
    matrices={}; summaries={}; hashes={}
    for n in MODALITIES:
        for suffix in ('_all_seed_means.csv.gz','_rankings.csv'):
            name=n+suffix; digest=file_hash(args.source/name)
            if digest!=record['artifacts'][name]: raise ValueError('Source artifact changed')
            hashes[name]=digest
        matrix=pd.read_csv(args.source/f'{n}_all_seed_means.csv.gz',index_col=0)
        matrix.index=matrix.index.astype(str)
        matrix.columns=matrix.columns.astype(int)
        summary=pd.read_csv(args.source/f'{n}_rankings.csv',index_col=0)
        summary.index=summary.index.astype(str)
        if matrix.columns.tolist()!=list(range(100)): raise ValueError('Seed order differs')
        matrices[n]=matrix; summaries[n]=summary
    args.output.mkdir(parents=True,exist_ok=False)
    render(args.output,matrices,summaries)
    write_json(args.output/'COMPLETE.json',{'status':'succeeded','layout':'1 row × 4 columns, shared seed x axis',
        'training_performed':False,'source':str(args.source.resolve()),'source_hashes':hashes,
        'artifacts':{f.name:file_hash(f) for f in args.output.iterdir() if f.is_file()}})


if __name__=='__main__': main()
