import json

import numpy as np
import pandas as pd
import pytest

from asc_reproduction.model_importance import COLUMNS, validate_importance
from asc_reproduction.model_checkpoints import save_checkpoint, verify_checkpoint, IMPORTANCE_FILE
from asc_reproduction.model_runner import guard_legacy_run
from test_model_checkpoints import payload


def archive(args):
    rows = []
    groups = [('standalone', -1, args['selections']['standalone']),
              ('stack_full', -1, args['selections']['full'])]
    groups += [('stack_inner', i, g) for i, g in enumerate(args['selections']['inner'])]
    for scope, inner, group in groups:
        for model, features in group.items():
            rows.extend((0,0,scope,inner,model,f,1/len(features)) for f in features)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_importance_roundtrip_and_tamper(tmp_path):
    args=payload(); args['identity']['importance_schema']=1
    args['importance']=archive(args)
    save_checkpoint(tmp_path/'fold', **args)
    verify_checkpoint(tmp_path/'fold', identity=args['identity'],seed=0,fold=0)
    actual=pd.read_csv(tmp_path/'fold'/IMPORTANCE_FILE)
    pd.testing.assert_frame_equal(actual,args['importance'])
    assert json.loads((tmp_path/'fold'/'COMPLETE.json').read_text())['checkpoint_schema']==2
    (tmp_path/'fold'/IMPORTANCE_FILE).unlink()
    with pytest.raises(ValueError,match='artifact'):
        verify_checkpoint(tmp_path/'fold',identity=args['identity'],seed=0,fold=0)


@pytest.mark.parametrize('fault',['missing','duplicate','nan','negative','wrong_feature','wrong_seed','wrong_sum','inner'])
def test_invalid_importance_blocks_completion(tmp_path,fault):
    args=payload(); frame=archive(args)
    if fault=='missing': frame=frame.iloc[1:]
    if fault=='duplicate': frame=pd.concat([frame,frame.iloc[:1]])
    if fault=='nan': frame.loc[0,'importance']=np.nan
    if fault=='negative': frame.loc[0,'importance']=-1
    if fault=='wrong_feature': frame.loc[0,'feature']='other'
    if fault=='wrong_seed': frame.loc[0,'seed']=1
    if fault=='wrong_sum': frame.loc[0,'importance']=.5
    if fault=='inner': frame.loc[frame.scope.eq('stack_inner'),'inner_fold']=99
    args['importance']=frame
    with pytest.raises(ValueError): save_checkpoint(tmp_path/'bad',**args)
    assert not (tmp_path/'bad').exists()


def test_new_run_cannot_complete_without_importance(tmp_path):
    args=payload(); args['identity']['importance_schema']=1
    with pytest.raises(ValueError,match='require'): save_checkpoint(tmp_path/'bad',**args)


def test_legacy_readable_but_not_resumed_as_complete(tmp_path):
    args=payload(); save_checkpoint(tmp_path/'old',**args)
    verify_checkpoint(tmp_path/'old',identity=args['identity'],seed=0,fold=0)
    with pytest.raises(RuntimeError,match='Legacy'):
        verify_checkpoint(tmp_path/'old',identity={**args['identity'],'importance_schema':1},seed=0,fold=0)
    (tmp_path/'seed_000_fold_0.latest.json').write_text(json.dumps({'attempt':'old'}))
    with pytest.raises(RuntimeError,match='Legacy'):
        guard_legacy_run(tmp_path,{'importance_schema':1})


def test_new_archive_resume_skips_worker(tmp_path,monkeypatch):
    from asc_reproduction import model_runner as runner
    root=tmp_path/'run'
    args=payload(); args['identity']['importance_schema']=1; args['importance']=archive(args)
    with runner.model_lock(root):
        save_checkpoint(root/'s000f0-test',**args)
        (root/'seed_000_fold_0.latest.json').write_text(json.dumps({'attempt':'s000f0-test'}))
    monkeypatch.setattr(runner,'identity',lambda path:args['identity'])
    def forbidden(*a): raise AssertionError('Valid cache must not retrain')
    monkeypatch.setattr(runner,'execute',forbidden)
    report=runner.run_folds(tmp_path,root,seeds=[0],folds=[0],resume=True)
    assert report['folds'][0]['status']=='verified_cached'


def test_legacy_guard_runs_before_summary_overwrite_or_worker(tmp_path,monkeypatch):
    from asc_reproduction import model_runner as runner
    root=tmp_path/'run'; args=payload()
    with runner.model_lock(root):
        save_checkpoint(root/'s000f0-test',**args)
        (root/'seed_000_fold_0.latest.json').write_text(json.dumps({'attempt':'s000f0-test'}))
        (root/'run_summary.json').write_text('preserve original')
    monkeypatch.setattr(runner,'identity',lambda path:{'importance_schema':1})
    def forbidden(*a): raise AssertionError('Legacy archive must not silently retrain')
    monkeypatch.setattr(runner,'execute',forbidden)
    with pytest.raises(RuntimeError,match='Legacy'):
        runner.run_folds(tmp_path,root,seeds=[0],folds=[0],resume=True)
    assert (root/'run_summary.json').read_text()=='preserve original'


def test_actual_small_fold_capture_is_complete_and_does_not_change_fit(tmp_path):
    from asc_reproduction.model_evaluation import evaluate_development_fold, MODALITIES
    from asc_reproduction.model_stacking import fit_stacking
    from asc_reproduction.model_splits import reference_split, outer_folds
    rng=np.random.default_rng(81)
    ids=[f'synthetic-{i:03}' for i in range(100)]
    meta=pd.DataFrame({'case_ID':ids,'subtype':['LUAD']*50+['LUSC']*50,'sample_type':'T'},index=ids)
    split=reference_split(meta)
    data={'patients':np.array(ids),'y':split.labels}
    for modality in MODALITIES.values():
        data[modality]=rng.normal(size=(100,10))
        data[modality+'_features']=np.array([f'{modality}_{i}' for i in range(10)])
    data['snv']=rng.integers(0,2,(100,10))
    result=evaluate_development_fold(data,split,seed=0,fold=0)
    def names(selected,standalone=False):
        return {n:data[(MODALITIES[n] if standalone else ('snv' if n.startswith('snv') else 'cnv'))+'_features'][i].tolist()
                for n,i in selected.items()}
    selections={'standalone':names(result['standalone'].selected,True),
                'full':names(result['stack'].base.selected),
                'inner':[names(s) for s in result['stack'].inner_selected]}
    validate_importance(result['importance'], selections, 0,0)
    assert len(result['importance'])==280  # 4 standalone + 4 full + 5*4 inner, ten features each
    tr,te=list(outer_folds(split,0))[0]
    train=split.development[tr]; test=split.development[te]
    plain=fit_stacking(data['snv'][train],data['cnv'][train],split.labels[train],seed=0)
    predictions=plain.predict(data['snv'][test],data['cnv'][test])
    for model in predictions:
        np.testing.assert_array_equal(predictions[model]['probability'],result['predictions'][model]['probability'])
    args=dict(metrics=result['metrics'],selections=selections,curves=result['stack'].base.curves,
              identity={'synthetic':True,'importance_schema':1},seed=0,fold=0,resources={},importance=result['importance'])
    save_checkpoint(tmp_path/'fold',**args)
    verify_checkpoint(tmp_path/'fold',identity=args['identity'],seed=0,fold=0)
