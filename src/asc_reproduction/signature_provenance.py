"""Read-only audit of supplied signature fractions against source activities.

This verifies deterministic normalization, not de novo signature extraction.
Only aggregate evidence and file fingerprints are returned; no sample keys.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import read_pivot_sqlite


def compare_activities(activities, fractions):
    """Both matrices are signature-by-sample; align keys before comparison."""
    for table in (activities, fractions):
        if table.index.has_duplicates or table.columns.has_duplicates or table.empty:
            raise ValueError('Empty matrix or duplicate keys')
        x=table.to_numpy(dtype=float)
        if not np.isfinite(x).all() or (x<0).any():
            raise ValueError('Nonfinite or negative signature values')
    if set(activities.index)!=set(fractions.index) or set(activities.columns)!=set(fractions.columns):
        raise ValueError('Activity/fraction sample or signature keys differ')
    totals=activities.sum(axis=0)
    # Legacy divides each sample by its total then fills zero-total NaNs with 0.
    normalized=activities.div(totals.replace(0,np.nan),axis=1).fillna(0)
    normalized=normalized.loc[fractions.index,fractions.columns]
    maximum=float(np.max(np.abs(normalized.to_numpy()-fractions.to_numpy())))
    if maximum>1e-12:
        raise ValueError('Supplied fractions do not reproduce from activities')
    return {'signatures':len(fractions),'samples':fractions.shape[1],
            'zero_total_samples':int(totals.eq(0).sum()),'maximum_absolute_difference':maximum,
            'normalization':'Per-sample activity / sum of activities; zero totals become zero',
            'scope':'Deterministic activity-to-fraction verification only; not extraction rerun'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--activities',type=Path,required=True)
    p.add_argument('--fractions-db',type=Path,required=True)
    args=p.parse_args()
    paths={'activities':args.activities,'fractions_db':args.fractions_db}
    fingerprints={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    activities=pd.read_csv(args.activities,sep='\t',index_col=0).T
    fractions=read_pivot_sqlite(args.fractions_db).values
    report=compare_activities(activities,fractions)
    after={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    if fingerprints!=after: raise ValueError('Inputs changed during verification')
    report['sha256']=fingerprints
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
