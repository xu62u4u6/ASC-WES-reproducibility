"""Verified standalone-only 500-fold backfill and S20 rendering (no stacking)."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from asc_reproduction.model_inputs import load_model_inputs
from asc_reproduction.model_splits import outer_folds
from asc_reproduction.model_evaluation import fit_standalone, MODALITIES
from asc_reproduction.model_importance import collect_importance
from asc_reproduction.provenance import file_hash, write_json

DATA = SPLIT = None  # inherited read-only through fork; no patient IDs exported
FILES = ('importance.csv.gz','metrics.csv','selected_features.json')


def validate(frame, selections, seed, fold):
    if not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all():
        raise ValueError('Wrong partition')
    if not frame.scope.eq('standalone').all() or not frame.inner_fold.eq(-1).all():
        raise ValueError('Mixed importance scope')
    if frame.isna().any().any() or frame.duplicated(['model','feature']).any():
        raise ValueError('Missing or duplicate importance')
    expected={(n,f) for n,fs in selections.items() for f in fs}
    if set(frame[['model','feature']].itertuples(index=False,name=None))!=expected:
        raise ValueError('Selection/importance mismatch')
    if set(selections)!=set(MODALITIES): raise ValueError('Missing models')
    values=frame.importance.to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Invalid importance')
    if not np.allclose(frame.groupby('model').importance.sum(),1,atol=1e-10):
        raise ValueError('Importance not normalized')


def read_fold(folder, identity, seed, fold):
    record=json.loads((folder/'COMPLETE.json').read_text())
    if record['identity']!=identity or record['seed']!=seed or record['fold']!=fold:
        raise ValueError('Checkpoint identity mismatch')
    if record['status']!='succeeded' or set(record['artifacts'])!=set(FILES):
        raise ValueError('Incomplete fold')
    for name,digest in record['artifacts'].items():
        if file_hash(folder/name)!=digest: raise ValueError('Corrupt fold')
    selections=json.loads((folder/'selected_features.json').read_text())
    frame=pd.read_csv(folder/'importance.csv.gz',dtype={'feature':str},keep_default_na=False)
    validate(frame,selections,seed,fold)
    return frame,selections,record


def run_fold(seed,fold,baseline,output,identity):
    started=time.monotonic()
    folder=output/f'seed_{seed:03}_fold_{fold}'
    if (folder/'COMPLETE.json').exists():
        _,_,report=read_fold(folder,identity,seed,fold)
        return report
    folder.mkdir(exist_ok=False)  # incomplete attempts are preserved, not overwritten
    train,test=list(outer_folds(SPLIT,seed))[fold]
    tr=SPLIT.development[train]; te=SPLIT.development[test]
    fit=fit_standalone({m:DATA[m][tr] for m in MODALITIES.values()},SPLIT.labels[tr],seed=seed)
    pred=fit.predict({m:DATA[m][te] for m in MODALITIES.values()})
    frame=collect_importance(fit,DATA,seed=seed,fold=fold,scope='standalone',inner_fold=-1)
    selections={n:DATA[MODALITIES[n]+'_features'][ids].tolist() for n,ids in fit.selected.items()}
    old_selections=json.loads((baseline/'selected_features.json').read_text())['standalone']
    if selections!=old_selections: raise ValueError('Baseline selections differ')
    validate(frame,selections,seed,fold)
    rows=[dict(model=n,seed=seed,fold=fold,acc=accuracy_score(SPLIT.labels[te],p['predicted']),
        f1=f1_score(SPLIT.labels[te],p['predicted'],average='macro'),
        auc=roc_auc_score(SPLIT.labels[te],p['probability'])) for n,p in pred.items()]
    metrics=pd.DataFrame(rows)
    old=pd.read_csv(baseline/'metrics.csv').set_index('model').loc[list(MODALITIES)]
    diff=(metrics.set_index('model').loc[list(MODALITIES),['acc','f1','auc']]-old[['acc','f1','auc']]).abs().to_numpy().max()
    if diff>1e-12: raise ValueError('Baseline metrics differ')
    frame.to_csv(folder/'importance.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    metrics.to_csv(folder/'metrics.csv',index=False)
    write_json(folder/'selected_features.json',selections)
    report={'status':'succeeded','identity':identity,'seed':seed,'fold':fold,
        'training_n':len(train),'validation_n':len(test),'elapsed_seconds':time.monotonic()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'max_metric_difference':float(diff),'artifacts':{n:file_hash(folder/n) for n in FILES}}
    write_json(folder/'COMPLETE.json',report)
    return report


def render(output,matrices,summaries):
    panels=[]
    for name in MODALITIES:
        table=summaries[name]
        top=matrices[name].loc[table.index[:20]]
        top.to_csv(output/f'{name}_top20_seed_means.csv')
        panels.append((name,top))
    def draw(ax,name,top,letter):
        sns.heatmap(top,ax=ax,cmap='viridis',vmin=0,xticklabels=False,yticklabels=True,
                    cbar_kws={'label':'Mean RF impurity importance'})
        ax.set_xticks([.5,24.5,49.5,74.5,99.5],labels=['0','24','49','74','99'])
        ax.set_xlabel('Seed (mean of 5 folds)'); ax.set_ylabel('Feature')
        ax.set_title(f'{letter}  {name}',loc='left'); ax.tick_params(axis='y',labelsize=8,rotation=0)
    def save(fig,stem):
        fig.savefig(output/f'{stem}.png',dpi=180)
        path=output/f'{stem}.tiff'
        fig.savefig(path,dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        plt.close(fig)
        with Image.open(path) as im: rgb=im.convert('RGB')
        rgb.save(path,dpi=(600,600),compression='tiff_lzw')
    fig,axes=plt.subplots(1,4,figsize=(26,6),layout='constrained',sharex=True)
    for letter,(name,top),ax in zip('ABCD',panels,axes.flat): draw(ax,name,top,letter)
    fig.suptitle('Feature importance across 100 seeds × 5 folds\nUnselected features contribute zero; independent color scale per model',fontsize=13)
    save(fig,'S20_combined')
    for letter,(name,top) in zip('ABCD',panels):
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        draw(ax,name,top,letter); save(fig,f'S20{letter}_{name}')


def main():
    global DATA,SPLIT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--formal-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,9))
    args=parser.parse_args(); started=time.monotonic()
    args.output.mkdir(exist_ok=True,parents=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=[Path(__file__),*Path('src/asc_reproduction').glob('*.py')]
        hashes={str(p):file_hash(p) for p in sources}
        baseline={}; digests={}; input_hashes=None
        for seed in range(100):
            for fold in range(5):
                pointer=json.loads((args.formal_run/f'seed_{seed:03}_fold_{fold}.latest.json').read_text())
                name=pointer['attempt']
                if Path(name).name!=name: raise ValueError('Invalid baseline pointer')
                d=args.formal_run/name; m=json.loads((d/'COMPLETE.json').read_text())
                if m['status']!='succeeded' or m['seed']!=seed or m['fold']!=fold: raise ValueError('Invalid baseline')
                for f,v in m['artifacts'].items():
                    if file_hash(d/f)!=v['sha256']: raise ValueError('Baseline corrupted')
                if input_hashes is None: input_hashes=m['identity']['inputs']
                if input_hashes!=m['identity']['inputs']: raise ValueError('Mixed baseline inputs')
                baseline[seed,fold]=d; digests[f'{seed}:{fold}']=file_hash(d/'COMPLETE.json')
        for name,digest in input_hashes.items():
            if file_hash(args.data_root/name)!=digest: raise ValueError('Input differs from original run')
        identity={'source':hashes,'inputs':input_hashes,'baseline_manifests':digests,
                  'method':'standalone-only; fixed original seed/fold/parameters; impurity importance'}
        write_json(args.output/'RUN_IDENTITY.json',identity)
        DATA,SPLIT=load_model_inputs(args.data_root)
        write_json(args.output/'feature_universe.json',{n:DATA[m+'_features'].tolist() for n,m in MODALITIES.items()})
        progress={'status':'running','completed_folds':0,'total_folds':500}
        write_json(args.output/'progress.json',progress)
        reports=[]
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork')) as pool:
                futures=[pool.submit(run_fold,s,f,d,args.output,identity) for (s,f),d in baseline.items()]
                for future in as_completed(futures):
                    report=future.result(); reports.append(report)
                    progress.update(completed_folds=len(reports),elapsed_seconds=time.monotonic()-started,
                        max_worker_peak_rss_kib=max(r['peak_rss_kib'] for r in reports))
                    write_json(args.output/'progress.json',progress)
                    if len(reports)%25==0: print(json.dumps(progress),flush=True)
            matrices={n:pd.DataFrame(0.,index=DATA[m+'_features'],columns=range(100)) for n,m in MODALITIES.items()}
            counts={n:pd.Series(0,index=matrix.index) for n,matrix in matrices.items()}
            for seed in range(100):
                for fold in range(5):
                    frame,selections,_=read_fold(args.output/f'seed_{seed:03}_fold_{fold}',identity,seed,fold)
                    for n in MODALITIES:
                        values=frame.loc[frame.model.eq(n)].set_index('feature').importance
                        matrices[n].loc[values.index,seed]+=values/5
                        counts[n].loc[values.index]+=1
            summaries={}
            for n,matrix in matrices.items():
                table=pd.DataFrame({'mean_importance':matrix.mean(axis=1),'selected_folds':counts[n],
                                    'total_folds':500,'selection_frequency':counts[n]/500})
                table=table.sort_values('mean_importance',ascending=False,kind='stable')
                table.index.name='feature'; summaries[n]=table
                table.to_csv(args.output/f'{n}_rankings.csv')
                matrix.loc[table.index].to_csv(args.output/f'{n}_all_seed_means.csv.gz',compression='gzip')
            render(args.output,matrices,summaries)
            if {str(p):file_hash(p) for p in sources}!=hashes: raise ValueError('Source changed during run')
            for name,digest in input_hashes.items():
                if file_hash(args.data_root/name)!=digest: raise ValueError('Input changed during run')
            for (s,f),d in baseline.items():
                if file_hash(d/'COMPLETE.json')!=digests[f'{s}:{f}']: raise ValueError('Baseline changed during run')
            summary={'status':'succeeded','completed_folds':500,'candidate_fits':2000,
                'max_metric_difference':max(r['max_metric_difference'] for r in reports),
                'elapsed_seconds':time.monotonic()-started,'max_worker_peak_rss_kib':max(r['peak_rss_kib'] for r in reports),
                'scope':'S20 standalone importance backfill, no stacking/RFECV/holdout refit',
                'identity':identity,'fold_manifests':{p.parent.name:file_hash(p) for p in args.output.glob('seed_*/COMPLETE.json')},
                'artifacts':{p.name:file_hash(p) for p in args.output.iterdir() if p.is_file() and p.suffix in ('.csv','.gz','.tiff','.png')}}
            write_json(args.output/'COMPLETE.json',summary)
            progress.update(status='succeeded',elapsed_seconds=summary['elapsed_seconds'])
        except BaseException as exc:
            progress.update(status='failed',error_type=type(exc).__name__,error=str(exc)); raise
        finally: write_json(args.output/'progress.json',progress)


if __name__=='__main__': main()
