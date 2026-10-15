"""Fold-isolated CV runner with verified resume (no final-fit stage yet)."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from .provenance import file_hash, runtime_identity, write_json, utc_now
from .runner import process_tree_rss_kib, THREAD_VARIABLES, SNV, GENE, ARM, CLUSTERS
from .model_checkpoints import save_checkpoint, verify_checkpoint
from .model_evaluation import evaluate_development_fold, MODALITIES
from .model_inputs import load_model_inputs


def identity(data_root):
    return {'runtime':runtime_identity(), 'inputs':{p:file_hash(data_root/p) for p in (SNV,GENE,ARM,CLUSTERS)},
            'threads':1,'analysis':'development_CV_only','schema':1,'importance_schema':1}


def guard_legacy_run(root, expected):
    """Do not turn an archive-format upgrade into an unannounced full rerun."""
    if expected.get('importance_schema') != 1:
        return
    for pointer in root.glob('seed_*_fold_*.latest.json'):
        name = json.loads(pointer.read_text())['attempt']
        if Path(name).name != name:
            raise ValueError('Invalid checkpoint pointer')
        record = json.loads((root/name/'COMPLETE.json').read_text())
        if record.get('checkpoint_schema', 1) != 2:
            raise RuntimeError('Legacy run lacks importance; preserve it and explicitly approve a separate backfill run')


def validate_resume(root, expected):
    """Verify every existing attempt before scheduling any resumed training.

    A missing fold may run; an incompatible or damaged existing run must not
    silently become a new experiment. Failed attempts retain request identity.
    """
    try:
        for request in sorted(root.glob('*.request.json')):
            if json.loads(request.read_text())['identity'] != expected:
                raise ValueError('Saved request identity differs')
        for pointer in sorted(root.glob('seed_*_fold_*.latest.json')):
            parts = pointer.name.split('_')
            seed = int(parts[1])
            fold = int(parts[3].split('.')[0])
            name = json.loads(pointer.read_text())['attempt']
            if Path(name).name != name or not name.startswith(f's{seed:03}f{fold}-'):
                raise ValueError('Invalid attempt path')
            verify_checkpoint(root / name, identity=expected, seed=seed, fold=fold)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            'Resume refused: saved identity or checkpoint verification failed. '
            'No folds were launched. Use the original verified model_snapshot '
            'and environment; investigate damaged inputs/checkpoints. A new '
            'experiment requires a separate output directory.'
        ) from exc


@contextmanager
def model_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.model.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=root/'.model_runner.json'
        if not marker.exists():
            if any(p.name!='.model.lock' for p in root.iterdir()):
                raise ValueError('Refuse existing non-model-runner directory')
            write_json(marker,{'schema':1,'created_utc':utc_now()})
        yield handle.fileno()


def worker(data_root,folder,request):
    start=time.monotonic()
    report={'status':'running','started_utc':utc_now()}
    record=folder.parent/(folder.name+'.execution.json')
    write_json(record,report)
    try:
        config=json.loads(request.read_text())
        if identity(data_root)!=config['identity']: raise ValueError('Worker identity differs')
        data,split=load_model_inputs(data_root)
        result=evaluate_development_fold(data,split,seed=config['seed'],fold=config['fold'])
        def names(selected,standalone=False):
            return {n:data[(MODALITIES[n] if standalone else ('snv' if n.startswith('snv') else 'cnv'))+'_features'][ids].tolist()
                    for n,ids in selected.items()}
        selections={'standalone':names(result['standalone'].selected,True),
                    'full':names(result['stack'].base.selected),
                    'inner':[names(s) for s in result['stack'].inner_selected]}
        if identity(data_root)!=config['identity']: raise ValueError('Worker inputs/source changed')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        resources={'elapsed_seconds':time.monotonic()-start,'peak_rss_kib':usage.ru_maxrss,
                   'cpu_seconds':usage.ru_utime+usage.ru_stime}
        save_checkpoint(folder,metrics=result['metrics'],selections=selections,curves=result['stack'].base.curves,
                        importance=result['importance'],
                        identity=config['identity'],seed=config['seed'],fold=config['fold'],resources=resources)
        report.update(status='succeeded',**resources)
        code=0
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        traceback.print_exc(file=sys.stderr)
        code=1
    report['diagnostic_log'] = folder.name + '.log'
    report.update(completed_utc=utc_now(),elapsed_seconds=time.monotonic()-start)
    write_json(record,report)
    return code


def launch_worker(command, env, lock_fd, folder):
    """Capture private worker stdout/stderr in an exclusive, owner-only log."""
    log_path = folder.parent / (folder.name + '.log')
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as log:
        return subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )


def execute(data_root,folder,request,lock_fd):
    env=os.environ.copy();env.update({n:'1' for n in THREAD_VARIABLES})
    cmd=[sys.executable,'-P','-m','asc_reproduction.model_runner','__worker',str(data_root),str(folder),str(request)]
    proc = launch_worker(cmd, env, lock_fd, folder)
    start=time.monotonic();peak=0
    try:
        while proc.poll() is None:
            current=process_tree_rss_kib(proc.pid);peak=max(peak,current)
            write_json(folder.parent/'progress.json',{'attempt':folder.name,'pid':proc.pid,'status':'running',
                       'elapsed_seconds':time.monotonic()-start,'tree_rss_kib':current,'sampled_tree_peak_rss_kib':peak})
            time.sleep(.5)
    except BaseException:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    return proc.returncode


def run_folds(data_root,root,*,seeds,folds,resume=False):
    if not seeds or not folds or len(set(seeds))!=len(seeds) or len(set(folds))!=len(folds):
        raise ValueError('Nonempty unique seeds/folds required')
    if any(s not in range(100) for s in seeds) or any(f not in range(5) for f in folds):
        raise ValueError('Seed/fold out of range')
    with model_lock(root) as lock_fd:
        expected=identity(data_root)
        guard_legacy_run(root, expected)
        if resume:
            validate_resume(root, expected)
        summary={'status':'running','scope':'requested development folds only; no final model or holdout evaluation',
                 'full_cv_requested':set(seeds)==set(range(100)) and set(folds)==set(range(5)), 'folds':[]}
        write_json(root/'run_summary.json',summary)
        try:
            for seed in seeds:
                for fold in folds:
                    pointer=root/f'seed_{seed:03}_fold_{fold}.latest.json'
                    cached=None
                    if resume and pointer.exists():
                        try:
                            name=json.loads(pointer.read_text())['attempt']
                            if Path(name).name!=name or not name.startswith(f's{seed:03}f{fold}-'): raise ValueError('Invalid attempt path')
                            verify_checkpoint(root/name,identity=expected,seed=seed,fold=fold)
                            cached=name
                        except (OSError,ValueError,KeyError,TypeError) as exc:
                            raise RuntimeError('Resume checkpoint changed after preflight') from exc
                    if identity(data_root)!=expected: raise ValueError('Execution identity changed')
                    if cached is None:
                        name=f's{seed:03}f{fold}-{uuid4().hex[:12]}'
                        folder=root/name;request=root/(name+'.request.json')
                        write_json(request,{'identity':expected,'seed':seed,'fold':fold})
                        if execute(data_root,folder,request,lock_fd)!=0:
                            raise RuntimeError(f'Model worker failed; inspect local log: {name}.log')
                        execution=json.loads((root/(name+'.execution.json')).read_text())
                        if execution.get('status')!='succeeded': raise RuntimeError('Missing successful execution')
                        verify_checkpoint(folder,identity=expected,seed=seed,fold=fold)
                        if identity(data_root)!=expected: raise ValueError('Execution identity changed')
                        write_json(pointer,{'attempt':name})
                    else: name=cached
                    summary['folds'].append({'seed':seed,'fold':fold,'attempt':name,'status':'verified_cached' if cached else 'succeeded'})
                    write_json(root/'run_summary.json',summary)
            summary['status']='succeeded'
        except BaseException as exc:
            summary.update(status='failed',error_type=type(exc).__name__)
            raise
        finally:
            summary['completed_utc']=utc_now();write_json(root/'run_summary.json',summary)
            write_json(root/'progress.json',{'status':summary['status'],'completed_folds':len(summary['folds'])})
        return summary


def main():
    if len(sys.argv)>1 and sys.argv[1]=='__worker':
        return worker(*(Path(v) for v in sys.argv[2:]))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(100)))
    parser.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=1,choices=range(1,33))
    args=parser.parse_args()
    if args.workers==1:
        run_folds(args.data_root,args.output,seeds=args.seeds,folds=args.folds,resume=args.resume)
    else:
        from .model_parallel import run_parallel
        run_parallel(args.data_root,args.output,seeds=args.seeds,folds=args.folds,workers=args.workers,resume=args.resume)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
