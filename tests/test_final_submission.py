"""Final audit: raw preservation, path aliases, identity checks and safe CLI entry."""
from pathlib import Path, PureWindowsPath
import importlib
import json
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('folder',['archive(3)','archive (3)'])
def test_both_dataset_folder_spellings_and_env_paths_work(tmp_path,monkeypatch,folder):
    from src.data_loader import get_data_dir,discover_raw_files,RAW_FILENAMES
    project=tmp_path/'AirSense with spaces';raw=project/'data'/'raw'/folder
    raw.mkdir(parents=True)
    for filename in RAW_FILENAMES.values():(raw/filename).write_text('City,Date\nDelhi,2020-01-01\n')
    monkeypatch.chdir(project)
    monkeypatch.delenv('AIRSENSE_DATA_DIR',raising=False)
    assert get_data_dir()==raw.resolve()
    assert len(discover_raw_files())==5
    monkeypatch.setenv('AIRSENSE_DATA_DIR',str(raw))
    monkeypatch.chdir(tmp_path)
    assert get_data_dir()==raw.resolve()
    assert len(discover_raw_files(raw))==5


def test_windows_documented_path_joins_without_unix_assumptions():
    path=PureWindowsPath(r'C:\dev\AirSense')/'data'/'raw'/'archive(3)'/'city_day.csv'
    assert str(path)==r'C:\dev\AirSense\data\raw\archive(3)\city_day.csv'


def test_raw_files_match_original_uploaded_reference():
    from src.data_loader import discover_raw_files
    from scripts.final_audit import sha
    original={v['filename']:v for v in json.loads((ROOT/'reports/raw_data_reference.json').read_text())['files']}
    files=discover_raw_files()
    assert len(original)==len(files)==5
    for path in files.values():
        assert sha(path)==original[path.name]['sha256']
        assert path.stat().st_size==original[path.name]['size_bytes']


def test_split_audit_detects_record_overlap_after_resetting_indices():
    from src.data_split import audit_data_split
    train=pd.DataFrame({'City':['Delhi','Mumbai'],'Date':['2020-01-01','2020-01-02'],'AQI_Bucket':['Good','Moderate']})
    val=train.iloc[[0]].reset_index(drop=True)
    test=pd.DataFrame({'City':['Delhi'],'Date':['2020-01-03'],'AQI_Bucket':['Good']})
    audit=audit_data_split(train,val,test)
    assert audit['overlap_check']['checked']
    assert not audit['overlap_check']['disjoint']
    assert audit['overlap_check']['overlap_counts']['train_validation']==1
    assert audit['overlap_check']['overlap_counts']['train_test']==0


def test_legacy_runner_import_does_not_train_or_set_data_dir(monkeypatch):
    from src import experiment
    def forbidden(*a,**kw):raise AssertionError('Import must not train')
    monkeypatch.setattr(experiment,'fit_baselines',forbidden)
    monkeypatch.delenv('AIRSENSE_DATA_DIR',raising=False)
    module=importlib.import_module('scripts.run_all_models');importlib.reload(module)
    import os
    assert 'AIRSENSE_DATA_DIR' not in os.environ


def test_compatibility_results_equal_current_evaluation():
    current=json.loads((ROOT/'reports/evaluation_results.json').read_text())
    compatibility=json.loads((ROOT/'models/all_results.json').read_text())
    for key in ['decision_tree','random_forest','svm','xgboost']:
        assert compatibility[key]['test_macro_f1']==current['candidates'][key]['test_metrics']['macro_f1']
        assert compatibility[key]['test_accuracy']==current['candidates'][key]['test_metrics']['accuracy']
    assert (ROOT/'reports/archive/pre_final_audit/models__all_results.json').is_file()


def test_legacy_runner_uses_correct_stage_order(monkeypatch):
    from src import experiment
    from scripts import analyze_features,run_all_models
    calls=[]
    monkeypatch.setattr(experiment,'fit_baselines',lambda:calls.append('training'))
    monkeypatch.setattr(experiment,'tune_all',lambda:calls.append('tuning'))
    monkeypatch.setattr(experiment,'evaluate_all',lambda:calls.append('frozen evaluation'))
    monkeypatch.setattr(analyze_features,'main',lambda:calls.append('feature analysis'))
    run_all_models.main()
    assert calls==['training','tuning','frozen evaluation','feature analysis']
