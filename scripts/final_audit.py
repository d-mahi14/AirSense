"""Read-only scientific/artifact audit, with optional isolated training reproduction.

Writes final audit evidence and final_results.csv, never replaces submitted models
or experiment outputs. --verify-training fits baselines in a temporary directory.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import joblib
from src.model_io import load_pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

from src.data_loader import discover_raw_files
from src.feature_engineering import LEAKAGE_COLS
from src import experiment
from src.tuning import date_blocked_cv
from src.explainability import get_tree_importance,get_permutation_importance

R=ROOT/'reports';LOGS=R/'final_audit_logs';NAMES=experiment.NAMES;LABELS=list(range(6))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def raw_integrity():
    expected={v['filename']:v for v in read(R/'raw_data_reference.json')['files']}
    files=discover_raw_files();rows=[]
    for path in files.values():
        original=expected[path.name];current=sha(path)
        assert current==original['sha256'] and path.stat().st_size==original['size_bytes'],f'Raw data changed: {path.name}'
        columns=pd.read_csv(path,nrows=0).columns.tolist()
        weather=[c for c in columns if any(token in c.lower() for token in ['temperature','humidity','wind','rain','pressure','weather'])]
        assert not weather,(path.name,weather)
        rows.append({'file':path.name,'bytes':path.stat().st_size,'sha256':current,'unchanged_from_uploaded_original':True})
    pd.DataFrame(rows).to_csv(R/'raw_data_integrity.csv',index=False)
    return rows


def independent_metrics(y,pred):
    return {'accuracy':accuracy_score(y,pred),
            'macro_precision':precision_score(y,pred,labels=LABELS,average='macro',zero_division=0),
            'macro_recall':recall_score(y,pred,labels=LABELS,average='macro',zero_division=0),
            'macro_f1':f1_score(y,pred,labels=LABELS,average='macro',zero_division=0),
            'weighted_f1':f1_score(y,pred,labels=LABELS,average='weighted',zero_division=0)}


def check_saved_metrics(y,pred,saved):
    for key,value in independent_metrics(y,pred).items():assert round(value,6)==saved[key],key
    assert confusion_matrix(y,pred,labels=LABELS).tolist()==saved['confusion_matrix']
    report=classification_report(y,pred,labels=LABELS,target_names=saved['class_labels'],zero_division=0,output_dict=True)
    for category in saved['class_labels']:
        for metric in ['precision','recall','f1-score','support']:
            assert np.isclose(report[category][metric],saved['per_class_report'][category][metric],atol=1e-12)


def audit_training_reproduction(evidence):
    original_models=experiment.MODELS;original_reports=experiment.REPORTS
    before={p.name:sha(p) for p in original_models.glob('*.joblib')}
    output=io.StringIO()
    try:
        with tempfile.TemporaryDirectory(prefix='airsense-training-audit-') as temporary:
            experiment.MODELS=Path(temporary)/'models';experiment.REPORTS=Path(temporary)/'reports'
            with contextlib.redirect_stdout(output):results=experiment.fit_baselines()
            assert set(results)==set(NAMES)
            verification={}
            for name,result in results.items():
                assert 'error' not in result,(name,result)
                assert result['used_n_train']==16645
                assert result['validation_metrics']['macro_f1']==evidence['candidates'][name]['validation_metrics']['macro_f1']
                assert result['validation_metrics']['confusion_matrix']==evidence['candidates'][name]['validation_metrics']['confusion_matrix']
                verification[name]={'full_training_rows':result['used_n_train'],'validation_rows':result['n_val'],
                                    'validation_macro_f1':result['validation_metrics']['macro_f1'],
                                    'reproduced_fit_time_sec':result['train_time_sec'],'validation_confusion_matches_saved':True}
            return verification
    finally:
        experiment.MODELS=original_models;experiment.REPORTS=original_reports
        LOGS.mkdir(exist_ok=True);(LOGS/'training_reproduction.log').write_text(output.getvalue(),encoding='utf-8')
        assert before=={p.name:sha(p) for p in original_models.glob('*.joblib')},'Submitted model changed during isolated reproduction'


def run(verify_training=False,deep_features=False):
    LOGS.mkdir(exist_ok=True)
    result={'audit_time_ist':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),'raw_files':raw_integrity(),
            'execution_platform':'Linux / Python local runtime; Windows commands supplied but not executed on Windows'}
    evidence=read(R/'evaluation_results.json');selection=read(ROOT/'models/selection_manifest.json')
    assert sha(ROOT/'models/selection_manifest.json')==evidence['selection_manifest_sha256']
    current_environment=experiment.identity()
    assert {k:v for k,v in current_environment.items() if k!='python'}=={k:v for k,v in evidence['experiment'].items() if k!='python'}
    result['runtime_environment']=current_environment
    result['training_environment']=evidence['experiment']
    partitions={key:experiment.data_for(key) for key in ['train','validation','test']}
    identities={key:set(zip(values[0].City,values[0].Date.astype(str))) for key,values in partitions.items()}
    assert not identities['train']&identities['validation'] and not identities['train']&identities['test'] and not identities['validation']&identities['test']
    assert partitions['train'][0].Date.max()<partitions['validation'][0].Date.min()<partitions['test'][0].Date.min()
    result['partitions']={key:{'n':len(v[0]),'start':str(v[0].Date.min().date()),'end':str(v[0].Date.max().date())} for key,v in partitions.items()}
    assert sum(len(v[0]) for v in partitions.values())==24850
    for tr,va in date_blocked_cv(partitions['train'][0].Date):
        dates=partitions['train'][0].Date
        assert dates.iloc[tr].max()<dates.iloc[va].min()-pd.Timedelta(days=7)
    result['partitions_disjoint']=True;result['temporal_cv_gap_verified']=True
    predictions=pd.read_csv(R/'test_predictions.csv');test_frame,Xtest,ytest,_=partitions['test'];_,Xval,yval,_=partitions['validation']
    rows=[];validation_scores={};models={}
    for candidate,record in evidence['candidates'].items():
        assert 'error' not in record,(candidate,record)
        path=ROOT/'models'/f'{candidate}.joblib';assert sha(path)==record['model_sha256']
        assert selection['candidates'][candidate]['model_sha256']==record['model_sha256']
        model=load_pipeline(path);models[candidate]=model
        columns=list(model.feature_names_in_)
        assert columns==Xtest.columns.tolist()
        assert not set(columns)&set(LEAKAGE_COLS)
        assert set(columns)==set(partitions['train'][1].columns)
        assert not any(any(token in c.lower() for token in ['weather','temperature','humidity','wind']) for c in columns)
        config=read(ROOT/'models'/f'{candidate}_config.json')
        assert config['used_n_train']==len(partitions['train'][1])
        actual_params=model.get_params(deep=True)
        if candidate.startswith('tuned_'):
            for parameter,value in config['best_params'].items():assert actual_params[parameter]==value,(candidate,parameter)
            cv=pd.read_csv(R/'tuning'/f"{candidate.removeprefix('tuned_')}_cv_results.csv")
            assert np.isclose(cv.mean_test_score.max(),config['cv_best_f1'],atol=1e-12)
            matching=[row for _,row in cv.iterrows() if ast.literal_eval(row['params'])==config['best_params']]
            assert matching and any(np.isclose(row.mean_test_score,config['cv_best_f1'],atol=1e-12) for row in matching)
        else:
            clf_params=model.named_steps['classifier'].get_params()
            for parameter,value in config['hyperparameters'].items():
                actual=clf_params[parameter]
                if value is None and isinstance(actual,float) and np.isnan(actual):continue
                assert actual==value,(candidate,parameter)
        # Independently reconstruct the fitted numeric preprocessing from training rows only.
        pre=model.named_steps['preprocessor'];numeric_columns=pre.transformers_[0][2]
        values=partitions['train'][1][numeric_columns].to_numpy(dtype=float)
        lower=np.nanpercentile(values,1,axis=0);upper=np.nanpercentile(values,99,axis=0)
        clipped=np.clip(values,lower,upper);medians=np.nanmedian(clipped,axis=0)
        filled=np.where(np.isnan(clipped),medians,clipped)
        numeric=pre.named_transformers_['numerical']
        np.testing.assert_allclose(numeric.named_steps['winsorize'].lo_,lower,atol=1e-12)
        np.testing.assert_allclose(numeric.named_steps['winsorize'].hi_,upper,atol=1e-12)
        np.testing.assert_allclose(numeric.named_steps['imputer'].statistics_,medians,atol=1e-12)
        np.testing.assert_allclose(numeric.named_steps['scaler'].mean_,filled.mean(axis=0),atol=1e-10)
        np.testing.assert_allclose(numeric.named_steps['scaler'].var_,filled.var(axis=0),atol=1e-10)
        pred=model.predict(Xtest);saved=predictions[predictions.candidate==candidate].reset_index(drop=True)
        assert len(saved)==len(Xtest)
        assert list(zip(saved.City,saved.Date))==list(zip(test_frame.City,test_frame.Date.dt.strftime('%Y-%m-%d')))
        assert np.array_equal(saved.y_true.to_numpy(),np.asarray(ytest)) and np.array_equal(saved.y_pred.to_numpy(),pred)
        check_saved_metrics(ytest,pred,record['test_metrics'])
        vp=model.predict(Xval);check_saved_metrics(yval,vp,record['validation_metrics']);validation_scores[candidate]=round(f1_score(yval,vp,labels=LABELS,average='macro',zero_division=0),6)
        row={'candidate':candidate,'model':NAMES[candidate.removeprefix('tuned_')],'variant':'tuned' if candidate.startswith('tuned_') else 'baseline',
             'selected_on_validation':candidate==selection['selected_candidate'],'status':'success','validation_macro_f1':validation_scores[candidate],
             **{k:record['test_metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']},
             'train_refit_time_sec':record['train_time_sec'],'inference_time_sec':record['test_metrics']['inference_time_sec'],
             'search_time_sec':record.get('search_time_sec'), 'test_rows':len(Xtest)}
        rows.append(row);print(candidate,'current predictions match saved test and validation outputs',flush=True)
    winner=min(validation_scores,key=lambda key:(-validation_scores[key],key))
    assert winner==selection['selected_candidate']==evidence['selected_candidate']
    result['selected_candidate']=winner;result['all_eight_current_predictions_reproduced']=True;result['training_only_preprocessing_reconstructed']=True;result['saved_hyperparameters_and_cv_results_verified']=True
    result['no_target_or_weather_features']=True;result['all_model_hashes_verified']=True
    # Existing comparison CSVs must exactly reflect the same experiment, including its original timings.
    for path,display in [('model_comparison.csv',True),('all_candidate_comparison.csv',False)]:
        table=pd.read_csv(R/path)
        for _,row in table.iterrows():
            candidate=next((k for k,v in NAMES.items() if v==row.model),None) if display else row.model
            record=evidence['candidates'][candidate]
            for metric in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','inference_time_sec']:
                assert np.isclose(row[metric],record['test_metrics'][metric],atol=1e-12)
            assert np.isclose(row.train_time_sec,record['train_time_sec'],atol=1e-12)
    result['comparison_csvs_match_saved_outputs']=True
    feature_manifest=read(R/'feature_analysis/sampling_manifest.json');feature_checks={}
    for name,kind in [('random_forest','mdi'),('xgboost','gain')]:
        model=models[name];assert feature_manifest['models'][name]['model_sha256']==sha(ROOT/'models'/f'{name}.joblib')
        names=model[:-1].get_feature_names_out().tolist()
        fresh=get_tree_importance(model,names,kind).set_index('feature')
        saved=pd.read_csv(R/'feature_analysis'/f'{name}_{kind}.csv').set_index('feature')
        np.testing.assert_allclose(fresh.loc[saved.index,'importance'],saved.importance,atol=1e-12)
        feature_checks[name]={'built_in_importance_matches_saved':True}
        if deep_features:
            idx=feature_manifest['permutation_validation_indices']
            new=get_permutation_importance(model,Xval.iloc[idx].reset_index(drop=True),yval.iloc[idx].reset_index(drop=True),
                    Xval.columns.tolist(),n_repeats=feature_manifest['permutation_repeats'],random_state=42,n_jobs=1).set_index('feature')
            old=pd.read_csv(R/'feature_analysis'/f'{name}_permutation.csv').set_index('feature')
            np.testing.assert_allclose(new.loc[old.index,['importance_mean','importance_std']],old[['importance_mean','importance_std']],atol=1e-10)
            feature_checks[name]['permutation_recomputed_and_matches']=True
    shap_values=np.load(R/'feature_analysis/xgboost_shap_values.npz')['values']
    shap=pd.read_csv(R/'feature_analysis/xgboost_shap.csv').set_index('feature')
    names=[str(n).split('__')[-1] for n in models['xgboost'][:-1].get_feature_names_out()]
    means=pd.Series(np.abs(shap_values).mean(axis=(0,2)),index=names)
    np.testing.assert_allclose(means.loc[shap.index],shap.importance,atol=1e-12)
    assert shap_values.shape==(feature_manifest['shap_validation_rows'],18,6)
    result['features']={'models':feature_checks,'saved_shap_values_match_report':True,'shap_shape':list(shap_values.shape),
                        'shap_explainer_not_rerun':'Saved per-sample SHAP attributions verified; full TreeSHAP computation was not repeated.'}
    if verify_training:
        result['isolated_training_reproduction']=audit_training_reproduction(evidence)
        result['submitted_models_preserved_during_training']=True
    result['existing_research_pdf_sha256']=sha(R/'research_paper.pdf')
    from pypdf import PdfReader
    pdf=PdfReader(R/'research_paper.pdf');assert len(pdf.pages)==read(R/'paper_verification.json')['pdf_pages']
    result['research_pdf_pages']=len(pdf.pages)
    final=pd.DataFrame(rows).sort_values(['variant','macro_f1'],ascending=[True,False])
    final.to_csv(R/'final_results.csv',index=False)
    result['status']='passed';(R/'final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Final artifact audit passed; original evaluation metrics/timings were preserved.',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-training',action='store_true',help='Fit the four baselines in temporary storage; never replace submitted models')
    parser.add_argument('--deep-features',action='store_true',help='Recompute validation permutation importance')
    args=parser.parse_args()
    try:run(args.verify_training,args.deep_features)
    except Exception as exc:
        (R/'final_audit_failure.json').write_text(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'},indent=2))
        raise


if __name__=='__main__':main()
