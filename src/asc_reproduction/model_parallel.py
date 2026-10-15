"""Bounded process-level fold scheduling; each worker remains single-threaded."""

from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from .model_runner import model_lock, identity, guard_legacy_run, validate_resume, launch_worker
from .model_checkpoints import verify_checkpoint
from .provenance import write_json, utc_now
from .runner import THREAD_VARIABLES, process_tree_rss_kib


def stop_workers(active):
    for item in active.values():
        proc=item['process']
        if proc.poll() is None:
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for item in active.values():
        proc=item['process']
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()


def run_parallel(data_root,root,*,seeds,folds,workers,resume=False):
    if not isinstance(workers,int) or not 1<=workers<=32: raise ValueError('workers must be 1–32')
    if not seeds or not folds or len(set(seeds))!=len(seeds) or len(set(folds))!=len(folds):
        raise ValueError('Nonempty unique seeds/folds required')
    if any(s not in range(100) for s in seeds) or any(f not in range(5) for f in folds):
        raise ValueError('Seed/fold out of range')
    with model_lock(root) as lock_fd:
        expected=identity(data_root)
        guard_legacy_run(root, expected)
        if resume:
            validate_resume(root, expected)
        summary={'status':'running','workers':workers,'scope':'requested development folds only; no final fit or holdout',
                 'full_cv_requested':set(seeds)==set(range(100)) and set(folds)==set(range(5)),'folds':[]}
        write_json(root/'run_summary.json',summary)
        pending=deque((s,f) for s in seeds for f in folds);active={};peak=0
        env=os.environ.copy();env.update({n:'1' for n in THREAD_VARIABLES})
        try:
            while pending or active:
                while pending and len(active)<workers:
                    seed,fold=pending.popleft();pointer=root/f'seed_{seed:03}_fold_{fold}.latest.json'
                    cached=None
                    if resume and pointer.exists():
                        try:
                            name=json.loads(pointer.read_text())['attempt']
                            if Path(name).name!=name or not name.startswith(f's{seed:03}f{fold}-'): raise ValueError('Invalid attempt path')
                            verify_checkpoint(root/name,identity=expected,seed=seed,fold=fold);cached=name
                        except (OSError,ValueError,KeyError,TypeError) as exc:
                            raise RuntimeError('Resume checkpoint changed after preflight') from exc
                    if identity(data_root)!=expected: raise ValueError('Execution identity changed')
                    if cached:
                        summary['folds'].append({'seed':seed,'fold':fold,'attempt':cached,'status':'verified_cached'})
                        write_json(root/'run_summary.json',summary)
                        continue
                    name=f's{seed:03}f{fold}-{uuid4().hex[:12]}'
                    folder=root/name;request=root/(name+'.request.json')
                    write_json(request,{'seed':seed,'fold':fold,'identity':expected})
                    command=[sys.executable,'-P','-m','asc_reproduction.model_runner','__worker',str(data_root),str(folder),str(request)]
                    proc = launch_worker(command, env, lock_fd, folder)
                    active[name]={'process':proc,'seed':seed,'fold':fold,'started':time.monotonic()}
                for name,item in list(active.items()):
                    code=item['process'].poll()
                    if code is None: continue
                    if code!=0:
                        raise RuntimeError(f'Model worker failed; inspect local log: {name}.log')
                    report=json.loads((root/(name+'.execution.json')).read_text())
                    if report.get('status')!='succeeded': raise RuntimeError('Worker success report missing')
                    seed,fold=item['seed'],item['fold']
                    verify_checkpoint(root/name,identity=expected,seed=seed,fold=fold)
                    if identity(data_root)!=expected: raise ValueError('Execution identity changed')
                    write_json(root/f'seed_{seed:03}_fold_{fold}.latest.json',{'attempt':name})
                    summary['folds'].append({'seed':seed,'fold':fold,'attempt':name,'status':'succeeded'})
                    del active[name]
                    write_json(root/'run_summary.json',summary)
                running=[{'attempt':name,'pid':item['process'].pid,'seed':item['seed'],'fold':item['fold'],
                          'elapsed_seconds':time.monotonic()-item['started'],
                          'tree_rss_kib':process_tree_rss_kib(item['process'].pid)} for name,item in active.items()]
                peak=max(peak,sum(item['tree_rss_kib'] for item in running))
                write_json(root/'progress.json',{'status':'running','active':running,'completed_folds':len(summary['folds']),
                                                'pending_folds':len(pending),'sampled_workers_peak_rss_kib':peak})
                if active: time.sleep(.5)
            if len(summary['folds'])!=len(seeds)*len(folds): raise ValueError('Incomplete schedule')
            summary['status']='succeeded'
        except BaseException as exc:
            summary.update(status='failed',error_type=type(exc).__name__)
            stop_workers(active)
            raise
        finally:
            summary['folds'].sort(key=lambda row:(row['seed'],row['fold']))
            summary.update(completed_utc=utc_now(),sampled_workers_peak_rss_kib=peak)
            write_json(root/'run_summary.json',summary)
            write_json(root/'progress.json',{'status':summary['status'],'completed_folds':len(summary['folds'])})
        return summary
