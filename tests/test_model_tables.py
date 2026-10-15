from contextlib import nullcontext
import json

import pandas as pd
import pytest

from asc_reproduction import model_tables as tables
from asc_reproduction.model_checkpoints import CANDIDATES


def metrics():
    return pd.DataFrame([dict(model=m,seed=s,fold=f,acc=.7+f*.01,f1=.6,auc=.8 if m==tables.BASELINE_STACK else .9)
                        for m in sorted(CANDIDATES) for s in range(100) for f in range(5)])


def test_complete_table_mapping_and_non_posthoc_stack():
    n,d,r=tables.build_tables(metrics())
    assert len(n[5])==5 and len(n[6])==4
    row=n[5].set_index('model').loc['STACK']
    assert row.candidate==tables.BASELINE_STACK and row.auc_mean==pytest.approx(.8)
    assert r.iloc[0].model!=tables.BASELINE_STACK
    assert d[5].set_index('model').loc['STACK','AUROC']=='0.800 ± 0.000'
    assert row.acc_sd==pytest.approx(metrics().query('model == @tables.BASELINE_STACK').acc.std())
    with pytest.raises(ValueError): tables.build_tables(metrics().iloc[:-1])


def test_table6_display_has_four_decimals_while_table5_keeps_three():
    _, display, _ = tables.build_tables(metrics())

    assert display[5].set_index('model').loc['STACK', 'AUROC'] == '0.800 ± 0.000'
    assert display[6].iloc[0]['AUROC'] == '0.8000 ± 0.0000'


def test_format_saved_table6_is_hash_checked_and_does_not_refit(tmp_path):
    from asc_reproduction.provenance import file_hash

    source = tmp_path / 'source'
    source.mkdir()
    numeric_path = source / 'table6_numeric.csv'
    ranking_path = source / 'development_candidate_ranking.csv'
    pd.DataFrame([
        {
            'model': model,
            'n_folds': 500,
            'acc_mean': 0.9,
            'acc_sd': 0.01,
            'f1_mean': 0.9,
            'f1_sd': 0.01,
            'auc_mean': auc,
            'auc_sd': 0.01,
        }
        for model, auc in [
            ('SNV_TOP-CNV_ALL', 0.991),
            ('SNV_TOP-CNV_RFE', 0.9939),
            ('SNV_RFE-CNV_ALL', 0.992),
            ('SNV_RFE-CNV_RFE', 0.9942),
        ]
    ]).to_csv(numeric_path, index=False)
    pd.DataFrame([
        {'model': 'SNV_RFE-CNV_RFE', 'mean_auc': 0.9942},
        {'model': 'SNV_TOP-CNV_RFE', 'mean_auc': 0.9939},
    ]).to_csv(ranking_path, index=False)
    manifest = {
        'status': 'succeeded',
        'artifacts': {
            path.name: {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
            for path in (numeric_path, ranking_path)
        },
    }
    (source / 'COMPLETE.json').write_text(json.dumps(manifest))

    output = tmp_path / 'formatted'
    tables.format_saved_table6(numeric_path, output)

    display = pd.read_csv(output / 'Supplementary_Table_6_display.csv')
    report = json.loads((output / 'COMPLETE.json').read_text())
    winner = display.set_index('model').loc['SNV_RFE-CNV_RFE', 'AUROC']
    assert winner == '0.9942 ± 0.0100'
    assert report['model_refit'] is False
    assert report['display_decimals'] == 4


def test_format_saved_table6_rejects_hash_mismatch_before_output(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    numeric_path = source / 'table6_numeric.csv'
    ranking_path = source / 'development_candidate_ranking.csv'
    numeric_path.write_text('model,n_folds\nSNV_RFE-CNV_RFE,500\n')
    ranking_path.write_text('model\nSNV_RFE-CNV_RFE\n')
    (source / 'COMPLETE.json').write_text(json.dumps({
        'status': 'succeeded',
        'artifacts': {
            numeric_path.name: {
                'bytes': numeric_path.stat().st_size,
                'sha256': 'wrong',
            },
            ranking_path.name: {
                'bytes': ranking_path.stat().st_size,
                'sha256': 'wrong',
            },
        },
    }))
    output = tmp_path / 'formatted'

    with pytest.raises(ValueError, match='artifact changed'):
        tables.format_saved_table6(numeric_path, output)

    assert not output.exists()


def test_export_requires_complete_checkpoints_and_does_not_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(tables,'model_lock',lambda p:nullcontext())
    monkeypatch.setattr(tables,'snapshot_identity',lambda *a:{'verified':'identity'})
    monkeypatch.setattr(tables,'runtime_identity',lambda:{'export':'runtime'})
    def incomplete(*a): raise ValueError('Missing checkpoint')
    monkeypatch.setattr(tables,'collect_complete_cv',incomplete)
    out=tmp_path/'out'
    with pytest.raises(ValueError):tables.export_tables(tmp_path,tmp_path,tmp_path,out)
    assert not out.exists()
    monkeypatch.setattr(tables,'collect_complete_cv',lambda *a:(metrics(),None,None))
    tables.export_tables(tmp_path,tmp_path,tmp_path,out)
    report=json.loads((out/'COMPLETE.json').read_text())
    assert len(report['artifacts'])==5 and report['complete_folds']==500
    assert report['stack_is_not_posthoc_winner'] is True
    with pytest.raises(FileExistsError):tables.export_tables(tmp_path,tmp_path,tmp_path,out)


def test_export_rejects_identity_drift_before_output(tmp_path,monkeypatch):
    monkeypatch.setattr(tables,'model_lock',lambda p:nullcontext())
    identities=iter([{'v':1},{'v':2}])
    monkeypatch.setattr(tables,'snapshot_identity',lambda *a:next(identities))
    monkeypatch.setattr(tables,'runtime_identity',lambda:{})
    monkeypatch.setattr(tables,'collect_complete_cv',lambda *a:(metrics(),None,None))
    with pytest.raises(ValueError,match='changed'):
        tables.export_tables(tmp_path,tmp_path,tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
