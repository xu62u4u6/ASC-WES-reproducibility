"""Full-CV completeness gate and development-only ranking; no partial winners."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .model_checkpoints import CANDIDATES, validate_metrics, verify_checkpoint
from .model_runner import model_lock, identity
from .model_snapshot import verify as verify_snapshot
from .provenance import write_json


def summarize_complete_cv(metrics):
    expected={(name,seed,fold) for name in CANDIDATES for seed in range(100) for fold in range(5)}
    if len(metrics)!=4000 or metrics.duplicated(['model','seed','fold']).any():
        raise ValueError('Full CV requires 4000 unique candidate/seed/fold rows')
    if set(metrics[['model','seed','fold']].itertuples(index=False,name=None))!=expected:
        raise ValueError('Full CV candidate/seed/fold coverage differs')
    for (seed,fold),group in metrics.groupby(['seed','fold']): validate_metrics(group,int(seed),int(fold))
    summary=metrics.groupby('model')[['acc','f1','auc']].agg(['mean','std'])
    ranking=metrics.groupby('model')[['auc','f1','acc']].mean().reset_index().sort_values(
        ['auc','f1','acc','model'],ascending=[False,False,False,True]).reset_index(drop=True)
    return summary,ranking


def snapshot_identity(snapshot,data_root):
    record=verify_snapshot(snapshot)
    expected=identity(data_root)
    if record.get('importance_schema') != 1:
        expected.pop('importance_schema', None)
    expected['runtime']['source_sha256']=record['files']
    return expected


def collect_complete_cv(root,expected):
    frames=[]
    for seed in range(100):
        for fold in range(5):
            pointer=root/f'seed_{seed:03}_fold_{fold}.latest.json'
            name=json.loads(pointer.read_text())['attempt']
            if Path(name).name!=name or not name.startswith(f's{seed:03}f{fold}-'):
                raise ValueError('Invalid checkpoint pointer')
            frames.append(verify_checkpoint(root/name,identity=expected,seed=seed,fold=fold))
    metrics=pd.concat(frames,ignore_index=True).sort_values(['seed','fold','model']).reset_index(drop=True)
    summary,ranking=summarize_complete_cv(metrics)
    return metrics,summary,ranking


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with model_lock(args.run):
        expected=snapshot_identity(args.snapshot,args.data_root)
        metrics,summary,ranking=collect_complete_cv(args.run,expected)
        if snapshot_identity(args.snapshot,args.data_root)!=expected:
            raise ValueError('Snapshot/data/runtime changed during collection')
        args.output.mkdir(parents=True,exist_ok=False)
        metrics.to_csv(args.output/'development_metrics.csv',index=False)
        summary.to_csv(args.output/'development_summary.csv')
        ranking.to_csv(args.output/'development_ranking.csv',index=False)
        write_json(args.output/'selection_decision.json',{
            'selected_model':ranking.iloc[0]['model'],'selected_using':'development_CV_only',
            'complete_folds':500,'candidate_rows':4000,'ranking_order':['mean_auc','mean_macro_f1','mean_accuracy','model_name'],
            'dispersion':'sample standard deviation across 500 correlated folds; not a confidence interval',
            'status':'development selection only; final model and holdout scoring not performed',
            'identity':expected})


if __name__=='__main__':main()
