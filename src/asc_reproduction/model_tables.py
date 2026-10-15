"""Supplementary Tables 5/6 from complete verified development CV only."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .model_evaluation import MODALITIES
from .model_learners import COMBINATIONS
from .model_summary import summarize_complete_cv, collect_complete_cv, snapshot_identity
from .model_runner import model_lock
from .provenance import runtime_identity, write_json, file_hash

BASELINE_STACK = 'SNV_TOP-CNV_ALL'
TABLE5 = [(name,name) for name in MODALITIES] + [('STACK', BASELINE_STACK)]
TABLE6 = [(name,name) for name in COMBINATIONS]


def display_from_numeric(table, precision):
    """Format a numeric summary table without recomputing model results."""
    rows = []
    for _, source in table.iterrows():
        row = {"model": source["model"]}
        metrics = [
            ("acc", "Accuracy"),
            ("f1", "Macro-F1"),
            ("auc", "AUROC"),
        ]
        for metric, heading in metrics:
            mean = float(source[f"{metric}_mean"])
            sd = float(source[f"{metric}_sd"])
            row[heading] = f"{mean:.{precision}f} ± {sd:.{precision}f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(metrics):
    summary,ranking = summarize_complete_cv(metrics)
    numeric={};display={}
    for number,mapping in [(5,TABLE5),(6,TABLE6)]:
        rows=[]
        for label,source in mapping:
            row={'model':label,'candidate':source,'n_folds':500}
            for metric,heading in [('acc','Accuracy'),('f1','Macro-F1'),('auc','AUROC')]:
                mean=float(summary.loc[source,(metric,'mean')])
                sd=float(summary.loc[source,(metric,'std')])
                row[f'{metric}_mean']=mean;row[f'{metric}_sd']=sd
            rows.append(row)
        numeric[number]=pd.DataFrame(rows)
        precision = 4 if number == 6 else 3
        display[number]=display_from_numeric(numeric[number],precision)
    return numeric,display,ranking


def format_saved_table6(numeric_path, output):
    """Format verified saved Table 6 values to four decimals without refitting."""
    numeric_path=Path(numeric_path)
    source_dir=numeric_path.parent
    manifest_path=source_dir/'COMPLETE.json'
    ranking_path=source_dir/'development_candidate_ranking.csv'
    manifest=json.loads(manifest_path.read_text())
    required={numeric_path.name,ranking_path.name}
    artifacts=manifest.get('artifacts',{})
    if manifest.get('status')!='succeeded' or not required.issubset(artifacts):
        raise ValueError('Saved Table 6 artifacts are not complete')
    for path in (numeric_path,ranking_path):
        expected=artifacts[path.name]
        if path.stat().st_size!=expected['bytes'] or file_hash(path)!=expected['sha256']:
            raise ValueError(f'Saved table artifact changed: {path.name}')
    ranking=pd.read_csv(ranking_path)
    if ranking.empty or ranking.iloc[0]['model']!='SNV_RFE-CNV_RFE':
        raise ValueError('Saved AUROC-first ranking does not select SNV_RFE-CNV_RFE')
    numeric=pd.read_csv(numeric_path)
    if len(numeric)!=4 or not numeric['n_folds'].eq(500).all():
        raise ValueError('Saved Table 6 must contain four 500-fold model summaries')
    display=display_from_numeric(numeric,4)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    display_path=output/'Supplementary_Table_6_display.csv'
    display.to_csv(display_path,index=False)
    write_json(output/'COMPLETE.json',{
        'status':'succeeded',
        'scope':'formatting-only derivative of verified step09a numeric results',
        'model_refit':False,
        'display_decimals':4,
        'ranking_rule':'Mean AUROC was compared before display rounding.',
        'source_numeric_sha256':file_hash(numeric_path),
        'source_ranking_sha256':file_hash(ranking_path),
        'artifacts':{display_path.name:{'bytes':display_path.stat().st_size,
            'sha256':file_hash(display_path)}}})


def export_tables(cv_run,snapshot,data_root,output):
    with model_lock(cv_run):
        expected=snapshot_identity(snapshot,data_root)
        execution=runtime_identity()
        metrics,_,_=collect_complete_cv(cv_run,expected)
        numeric,display,ranking=build_tables(metrics)
        if snapshot_identity(snapshot,data_root)!=expected or runtime_identity()!=execution:
            raise ValueError('Inputs or source changed during table collection')
        output.mkdir(parents=True,exist_ok=False)
        try:
            files=[]
            for number in (5,6):
                for kind,table in [('numeric',numeric[number]),('display',display[number])]:
                    path=output/f'Supplementary_Table_{number}_{kind}.csv'
                    table.to_csv(path,index=False);files.append(path)
            path=output/'development_candidate_ranking.csv'
            ranking.to_csv(path,index=False);files.append(path)
            if snapshot_identity(snapshot,data_root)!=expected or runtime_identity()!=execution:
                raise ValueError('Inputs or source changed during export')
            write_json(output/'COMPLETE.json',{
                'status':'succeeded','scope':'development CV only, 136 patients; no holdout or ASC scores',
                'complete_folds':500,'candidate_rows':4000,'baseline_STACK_candidate':BASELINE_STACK,
                'stack_is_not_posthoc_winner':True,
                'dispersion':'sample SD across 500 correlated fold scores; not SE or CI',
                'f1_definition':'macro-F1',
                'units':'0–1; Table 5 uses 3 decimals and Table 6 uses 4 decimals',
                'display_precision':{'table_5':3,'table_6':4},
                'ranking_rule':['mean AUROC','mean macro-F1','mean accuracy','model name'],
                'table_6_ranking_note':'Stacking configurations are ranked using unrounded mean AUROC.',
                'limitations':['retrospective internal split, not external validation',
                    'frozen CNV clusters were derived using full reference cohort, including later holdout'],
                'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in files},
                'cv_identity':expected,'export_runtime':execution})
        except BaseException as exc:
            write_json(output/'FAILED.json',{'error_type':type(exc).__name__})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-run',type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--saved-table6-numeric',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.saved_table6_numeric:
        if args.cv_run or args.snapshot or args.data_root:
            parser.error('Use either saved Table 6 formatting or full CV export')
        format_saved_table6(args.saved_table6_numeric,args.output)
        return
    if not all((args.cv_run,args.snapshot,args.data_root)):
        parser.error('Full CV export requires --cv-run, --snapshot, and --data-root')
    export_tables(args.cv_run,args.snapshot,args.data_root,args.output)


if __name__=='__main__':
    main()
