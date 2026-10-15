"""Gate final development fit on complete CV; freeze before holdout/ASC inference."""

import argparse
from pathlib import Path
import resource
import time
import tomllib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from .model_evaluation import MODALITIES
from .model_checkpoints import validate_curve
from .model_final import fit_final_candidate
from .model_inputs import load_model_inputs
from .model_runner import model_lock
from .model_summary import snapshot_identity, collect_complete_cv
from .provenance import runtime_identity, write_json, file_hash, utc_now

CORE_SOURCES = ('dataio.py','model_inputs.py','model_splits.py','model_learners.py',
                'model_stacking.py','model_evaluation.py','cgc_oncoplot.py','paired_screen.py')


def same_project_except_scripts(current_text,frozen_text):
    current=tomllib.loads(current_text);frozen=tomllib.loads(frozen_text)
    for project in (current,frozen): project.get('project',{}).pop('scripts',None)
    return current==frozen


def verify_fit_compatibility(expected,snapshot=None):
    current=runtime_identity()
    frozen=expected['runtime']['source_sha256']
    for name in (*('src/asc_reproduction/'+n for n in CORE_SOURCES),'analysis_spec.json','uv.lock'):
        if current['source_sha256'][name]!=frozen[name]:
            raise ValueError('Final-fit core differs from verified CV snapshot')
    if current['source_sha256']['pyproject.toml']!=frozen['pyproject.toml']:
        live=Path(__file__).resolve().parents[2]/'pyproject.toml'
        if snapshot is None or not same_project_except_scripts(live.read_text(),(Path(snapshot)/'pyproject.toml').read_text()):
            raise ValueError('Final-fit project settings differ beyond CLI entry points')
    return current


def named_matrices(data,rows=None,*,asc=False):
    return {mod:pd.DataFrame(data['asc_'+mod] if asc else data[mod][rows],columns=data[mod+'_features'])
            for mod in MODALITIES.values()}


def fit_freeze_score(data,split,winner,output):
    """Caller has already passed complete-CV gate; new output directory required."""
    if len(split.development)!=136 or len(split.holdout)!=35 or len(data['asc_samples'])!=51:
        raise ValueError('Unexpected final analysis cohort sizes')
    if set(split.development).intersection(split.holdout): raise ValueError('Holdout overlaps development')
    if not np.array_equal(data['patients'],split.metadata.index.to_numpy(dtype=str)) or not np.array_equal(data['y'],split.labels):
        raise ValueError('Final model data and split differ')
    development=named_matrices(data,split.development)
    labels=pd.Series(data['y'][split.development])
    diagnostics={}
    model=fit_final_candidate(development,labels,winner,diagnostics=diagnostics)
    save_final_diagnostics(diagnostics,output)
    model_path=output/'model.joblib'
    temporary=output/'model.tmp'
    joblib.dump(model,temporary);temporary.replace(model_path)
    digest=file_hash(model_path)
    write_json(output/'FROZEN_MODEL.json',{'sha256':digest,'training_count':136,'holdout_count':35,
                                          'frozen_utc':utc_now(),'refit_on_holdout':False})
    # Only reload this freshly-created trusted artifact, never an external pickle.
    frozen=joblib.load(model_path)
    probability,predicted=frozen.predict(named_matrices(data,split.holdout))
    truth=data['y'][split.holdout]
    holdout={'n':35,'acc':float(accuracy_score(truth,predicted)),
             'f1':float(f1_score(truth,predicted,average='macro')),'auc':float(roc_auc_score(truth,probability)),
             'confusion_matrix_LUAD_LUSC':confusion_matrix(truth,predicted,labels=[0,1]).tolist(),
             'limitation':'Retrospective internal holdout; historical CNV clusters used full reference data; not external or fully leakage-free.'}
    write_json(output/'holdout_metrics.json',holdout)
    asc_probability,asc_predicted=frozen.predict(named_matrices(data,asc=True))
    write_json(output/'ASC_score_summary.json',{'n':len(asc_probability),'mean':float(np.mean(asc_probability)),
        'min':float(np.min(asc_probability)),'max':float(np.max(asc_probability)),
        'predicted_reference_class_counts':np.bincount(asc_predicted.astype(int),minlength=2).tolist(),
        'interpretation':'Descriptive reference-model scores, not validated ASC diagnostic probabilities.'})
    features={b.name:list(b.features) for b in frozen.branches}
    write_json(output/'selected_features.json',features)
    rows=[{'branch':b.name,'feature':feature,'importance':float(value)} for b in frozen.branches
          for feature,value in zip(b.features,b.estimator.feature_importances_)]
    pd.DataFrame(rows).to_csv(output/'feature_importance.csv',index=False)
    if file_hash(model_path)!=digest: raise ValueError('Frozen model changed during inference')
    return digest


def save_final_diagnostics(diagnostics,output):
    """Export only full-development RFECV curves, never inner OOF or patient rows."""
    if diagnostics.get('training_count')!=136:
        raise ValueError('Final diagnostics require 136 development patients')
    curves=diagnostics['curves'];counts=diagnostics['selected_counts']
    if set(curves) not in (set(), {'snv','cnv'}):
        raise ValueError('Incomplete final RFECV diagnostics')
    report={'training_count':136,'selected_branches':diagnostics['selected_branches'],
        'status':'available' if curves else 'not_applicable_to_standalone_candidate',
        'scope':'RFECV fit on full development cohort, not outer-fold evaluation or holdout performance',
        'score':'RFECV internal five-fold accuracy; std is population SD (ddof=0), not CI' if curves else None,
        'snv_preprocessing':'TOP frequency filter fitted on full development before RFECV' if curves else None,
        'curves':{}}
    for modality,curve in curves.items():
        validate_curve(curve)
        branch=modality+'_rfe';selected=counts[branch]
        if selected not in set(curve.n_features):
            raise ValueError('Selected feature count missing from final RFECV curve')
        report['curves'][modality]={'selected_feature_count':selected,
            'used_in_selected_model':branch in diagnostics['selected_branches']}
    for modality,curve in curves.items():
        path=output/f'{modality}_rfecv_curve.csv'
        curve.to_csv(path,index=False)
        report['curves'][modality].update(file=path.name,sha256=file_hash(path),bytes=path.stat().st_size)
    write_json(output/'final_selection_diagnostics.json',report)


def finalize(cv_run,snapshot,data_root,output):
    started=time.monotonic()
    with model_lock(cv_run):
        expected=snapshot_identity(snapshot,data_root)
        execution=verify_fit_compatibility(expected,snapshot)
        metrics,summary,ranking=collect_complete_cv(cv_run,expected)
        winner=ranking.iloc[0]['model']
        # No output directory or fit is created before every CV checkpoint passes.
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'selection_decision.json',{'model':winner,'selected_using':'complete_development_CV_only',
                   'complete_folds':500,'candidate_rows':len(metrics),'model_seed':42,
                   'cv_identity':expected,'final_execution':execution})
        try:
            data,split=load_model_inputs(data_root)
            digest=fit_freeze_score(data,split,winner,output)
            if snapshot_identity(snapshot,data_root)!=expected or runtime_identity()!=execution:
                raise ValueError('Data/source/runtime changed during final fit')
            ranking.to_csv(output/'development_ranking.csv',index=False)
            write_json(output/'COMPLETE.json',{'status':'succeeded','model':winner,'model_sha256':digest,
                       'refit_on_holdout':False,'elapsed_seconds':time.monotonic()-started,
                       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'completed_utc':utc_now(),
                       'artifacts':{p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size}
                                    for p in sorted(output.iterdir()) if p.is_file()}})
        except BaseException as exc:
            write_json(output/'FAILED.json',{'status':'failed','error_type':type(exc).__name__})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-run',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();finalize(args.cv_run,args.snapshot,args.data_root,args.output)


if __name__=='__main__':main()
