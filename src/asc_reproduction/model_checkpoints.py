"""Validate immutable aggregate fold checkpoints; never cache by file existence alone."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_evaluation import MODALITIES
from .model_learners import BRANCHES, COMBINATIONS
from .provenance import file_hash, write_json

FILES = ('metrics.csv', 'selected_features.json', 'snv_rfecv_curve.csv', 'cnv_rfecv_curve.csv')
IMPORTANCE_FILE = 'feature_importance.csv.gz'
CANDIDATES = set(MODALITIES) | set(COMBINATIONS)


def validate_curve(frame):
    required = {'n_features', 'mean_test_score', 'std_test_score',
                *(f'split{i}_test_score' for i in range(5))}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError('Incomplete RFECV curve')
    counts=frame.n_features.to_numpy(dtype=float)
    if not np.isfinite(counts).all() or np.any(counts<10) or np.any(counts!=np.floor(counts)) or np.any(np.diff(counts)<=0):
        raise ValueError('Invalid RFECV feature counts')
    scores=frame[[f'split{i}_test_score' for i in range(5)]].to_numpy(dtype=float)
    means=frame.mean_test_score.to_numpy(dtype=float)
    stds=frame.std_test_score.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):
        raise ValueError('Invalid RFECV scores')
    if not np.allclose(means,scores.mean(axis=1),atol=1e-12,rtol=0) or not np.allclose(stds,scores.std(axis=1),atol=1e-12,rtol=0):
        raise ValueError('RFECV summaries differ from fold scores')


def validate_metrics(metrics, seed, fold):
    required = {'model', 'seed', 'fold', 'acc', 'f1', 'auc'}
    if set(metrics.columns) != required or len(metrics) != 8 or set(metrics.model) != CANDIDATES:
        raise ValueError('Fold metrics must contain exactly eight unique candidates')
    if not metrics.seed.eq(seed).all() or not metrics.fold.eq(fold).all():
        raise ValueError('Metric partition differs from checkpoint')
    values=metrics[['acc','f1','auc']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
        raise ValueError('Invalid metric values')


def validate_selections(selections):
    if set(selections) != {'standalone','full','inner'} or len(selections['inner']) != 5:
        raise ValueError('Incomplete fold feature selections')
    groups=[(selections['standalone'],set(MODALITIES)),(selections['full'],set(BRANCHES))]
    groups.extend((group,set(BRANCHES)) for group in selections['inner'])
    for group,names in groups:
        if set(group)!=names:
            raise ValueError('Selection branches differ')
        for features in group.values():
            if not isinstance(features,list) or not features or not all(isinstance(v,str) and v for v in features):
                raise ValueError('Selected features must be nonempty string lists')
            if len(features)!=len(set(features)):
                raise ValueError('Duplicate selected features')


def save_checkpoint(folder, *, metrics, selections, curves, identity, seed, fold, resources, importance=None):
    """New directory only; completion manifest written last after validation."""
    validate_metrics(metrics,seed,fold)
    validate_selections(selections)
    if set(curves)!={'snv','cnv'} or any(frame.empty for frame in curves.values()):
        raise ValueError('Missing full-training RFECV curves')
    for frame in curves.values(): validate_curve(frame)
    if importance is not None:
        from .model_importance import validate_importance
        validate_importance(importance, selections, seed, fold)
    elif identity.get('importance_schema') == 1:
        raise ValueError('New runs require per-fold importance')
    if not isinstance(identity,dict) or not identity:
        raise ValueError('Missing execution identity')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    metrics.to_csv(folder/'metrics.csv',index=False)
    write_json(folder/'selected_features.json',selections)
    for modality,frame in curves.items(): frame.to_csv(folder/f'{modality}_rfecv_curve.csv',index=False)
    files = list(FILES)
    if importance is not None:
        importance.to_csv(folder/IMPORTANCE_FILE,index=False,compression={'method':'gzip','mtime':0})
        files.append(IMPORTANCE_FILE)
    inventory={name:{'sha256':file_hash(folder/name),'bytes':(folder/name).stat().st_size} for name in files}
    write_json(folder/'COMPLETE.json',{'status':'succeeded','seed':seed,'fold':fold,
                                     'checkpoint_schema':2 if importance is not None else 1,
                                     'identity':identity,'artifacts':inventory,'resources':resources})


def verify_checkpoint(folder, *, identity, seed, fold):
    """Return verified metrics or raise; callers rerun in a new attempt on failure."""
    folder=Path(folder)
    if folder.is_symlink(): raise ValueError('Checkpoint cannot be a symlink')
    record=json.loads((folder/'COMPLETE.json').read_text())
    if identity.get('importance_schema') == 1 and record.get('checkpoint_schema',1) != 2:
        raise RuntimeError('Legacy checkpoint lacks importance; use a new output directory after explicit rerun approval')
    if record.get('status')!='succeeded' or record.get('identity')!=identity or record.get('seed')!=seed or record.get('fold')!=fold:
        raise ValueError('Checkpoint identity differs')
    schema = record.get('checkpoint_schema',1)
    if schema not in (1,2): raise ValueError('Unknown checkpoint schema')
    files = set(FILES) | ({IMPORTANCE_FILE} if schema == 2 else set())
    if set(record.get('artifacts',{}))!=files: raise ValueError('Incomplete checkpoint inventory')
    for name,expected in record['artifacts'].items():
        path=folder/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=expected['bytes'] or file_hash(path)!=expected['sha256']:
            raise ValueError('Checkpoint artifact changed or missing')
    metrics=pd.read_csv(folder/'metrics.csv')
    validate_metrics(metrics,seed,fold)
    selections = json.loads((folder/'selected_features.json').read_text())
    validate_selections(selections)
    if schema == 2:
        from .model_importance import validate_importance
        frame = pd.read_csv(folder/IMPORTANCE_FILE, dtype={'feature':str}, keep_default_na=False)
        validate_importance(frame, selections, seed, fold)
    for modality in ('snv','cnv'):
        validate_curve(pd.read_csv(folder/f'{modality}_rfecv_curve.csv'))
    return metrics
