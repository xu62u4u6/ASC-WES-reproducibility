"""Freeze and verify source used by a long model run; data and environments stay external."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from .provenance import file_hash, write_json, utc_now

CONFIGS = ('analysis_spec.json', 'pyproject.toml', 'uv.lock')


def source_files(repository):
    paths = list((repository/'src'/'asc_reproduction').rglob('*.py'))
    paths.extend(repository/name for name in CONFIGS)
    if not (repository/'src'/'asc_reproduction'/'model_runner.py').is_file():
        raise ValueError('Missing model runner source')
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Snapshot requires regular source/config files')
    return sorted(paths)


def freeze(repository, destination):
    repository=Path(repository).resolve();destination=Path(destination).absolute()
    paths=source_files(repository)
    before={str(p.relative_to(repository)):file_hash(p) for p in paths}
    destination.mkdir(parents=True,exist_ok=False)
    for source in paths:
        target=destination/source.relative_to(repository)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    after={str(p.relative_to(repository)):file_hash(p) for p in source_files(repository)}
    if before!=after: raise ValueError('Source changed during snapshot creation')
    for name,digest in before.items():
        if file_hash(destination/name)!=digest: raise ValueError('Snapshot copy differs')
    write_json(destination/'SNAPSHOT.json',{'schema':1,'importance_schema':1,'created_utc':utc_now(),'files':before,
                'scope':'source and analysis config only; no data, results or Python environment'})
    return verify(destination)


def verify(snapshot):
    snapshot=Path(snapshot).absolute()
    if snapshot.is_symlink(): raise ValueError('Snapshot root must not be a symlink')
    record=json.loads((snapshot/'SNAPSHOT.json').read_text())
    if record.get('schema')!=1 or not isinstance(record.get('files'),dict) or not record['files']:
        raise ValueError('Invalid snapshot manifest')
    files=source_files(snapshot)
    actual={str(p.relative_to(snapshot)):file_hash(p) for p in files}
    if actual!=record['files']: raise ValueError('Snapshot source/config integrity mismatch')
    return record


def launch_command(snapshot, runner_args):
    snapshot=Path(snapshot).absolute()
    verify(snapshot)
    snapshot=snapshot.resolve()
    env=os.environ.copy()
    # Do not inherit a parent checkout on PYTHONPATH. -P excludes cwd/script paths.
    env['PYTHONPATH']=str(snapshot/'src')
    env['PYTHONNOUSERSITE']='1'
    return [sys.executable,'-P','-m','asc_reproduction.model_runner',*runner_args],env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    create=commands.add_parser('create')
    create.add_argument('--output',type=Path,required=True)
    check=commands.add_parser('verify');check.add_argument('--snapshot',type=Path,required=True)
    run=commands.add_parser('run');run.add_argument('--snapshot',type=Path,required=True)
    run.add_argument('runner_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command=='create':
        freeze(Path(__file__).resolve().parents[2],args.output)
    elif args.command=='verify': verify(args.snapshot)
    else:
        forwarded=args.runner_args
        if forwarded[:1]==['--']:forwarded=forwarded[1:]
        if not forwarded: raise ValueError('Runner arguments required')
        cmd,env=launch_command(args.snapshot,forwarded)
        os.execve(sys.executable,cmd,env)


if __name__=='__main__':
    main()
