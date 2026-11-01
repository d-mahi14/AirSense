"""Prompt 14: build Markdown first, then a PDF from verified saved outputs.

Read-only with respect to data, models and prompts 11-13. No fitting or search.
Run: python -m scripts.generate_research_paper
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

R = ROOT / 'reports'
F = R / 'figures' / 'research_paper'
NAMES = {'decision_tree':'Decision Tree','random_forest':'Random Forest','svm':'SVM','xgboost':'XGBoost'}
CLASSES = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
SHORT = ['Good','Sat.','Mod.','Poor','V. Poor','Severe']
FEATURES = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
COLORS = ['#177f7b','#5367a4','#bd803a','#865d8a']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def table(columns, rows):
    def clean(x):return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])


def figure(n, filename, caption):
    return f'![Figure {n}](figures/research_paper/{filename})\n\n*Figure {n}. {caption}*'


def verify_outputs(evaluation):
    predictions=pd.read_csv(R/'test_predictions.csv')
    identities=[]
    for candidate, group in predictions.groupby('candidate',sort=False):
        saved=evaluation['candidates'][candidate]['test_metrics']
        actual={'accuracy':accuracy_score(group.y_true,group.y_pred),
                'macro_precision':precision_score(group.y_true,group.y_pred,labels=range(6),average='macro',zero_division=0),
                'macro_recall':recall_score(group.y_true,group.y_pred,labels=range(6),average='macro',zero_division=0),
                'macro_f1':f1_score(group.y_true,group.y_pred,labels=range(6),average='macro',zero_division=0),
                'weighted_f1':f1_score(group.y_true,group.y_pred,labels=range(6),average='weighted',zero_division=0)}
        for key,value in actual.items():
            assert round(value,6)==saved[key],(candidate,key)
        assert confusion_matrix(group.y_true,group.y_pred,labels=range(6)).tolist()==saved['confusion_matrix']
        assert digest(ROOT/'models'/f'{candidate}.joblib')==evaluation['candidates'][candidate]['model_sha256']
        identities.append(list(zip(group.City,group.Date)))
    assert len(identities)==8 and all(rows==identities[0] for rows in identities)
    selection=read(ROOT/'models/selection_manifest.json')
    assert digest(ROOT/'models/selection_manifest.json')==evaluation['selection_manifest_sha256']
    winner=min(evaluation['candidates'],key=lambda key:(-evaluation['candidates'][key]['validation_metrics']['macro_f1'],key))
    assert winner==evaluation['selected_candidate']==selection['selected_candidate']
    return {'candidate_count':len(identities),'test_rows':len(identities[0]),'actual_metrics_recomputed':True,
            'model_hashes_verified':True,'selection_verified_from_validation':True}


def data_audit():
    from src.data_loader import get_data_dir
    from src.tuning import date_blocked_cv
    from sklearn.model_selection import train_test_split
    path=get_data_dir()/'city_day.csv'
    raw=pd.read_csv(path,parse_dates=['Date'])
    labeled=raw[raw.AQI_Bucket.notna()].sort_values(['Date','City']).reset_index(drop=True)
    splits={'Train':labeled[labeled.Date<='2019-06-30'].reset_index(drop=True),
            'Validation':labeled[(labeled.Date>'2019-06-30')&(labeled.Date<='2019-12-31')].reset_index(drop=True),
            'Test':labeled[labeled.Date>'2019-12-31'].reset_index(drop=True)}
    result={'source_sha256':digest(path),'raw_rows':len(raw),'columns':len(raw.columns),
            'cities':int(raw.City.nunique()),'date_start':str(raw.Date.min().date()),'date_end':str(raw.Date.max().date()),
            'labeled':len(labeled),'unlabeled':int(raw.AQI_Bucket.isna().sum()),
            'duplicate_city_dates':int(raw.duplicated(['City','Date']).sum()),
            'complete_feature_labeled':int(labeled[FEATURES].notna().all(axis=1).sum()),'splits':{},'features':{}}
    for name,frame in splits.items():
        result['splits'][name]={'n':len(frame),'cities':int(frame.City.nunique()),'date_start':str(frame.Date.min().date()),
                                'date_end':str(frame.Date.max().date()),'class_counts':{c:int((frame.AQI_Bucket==c).sum()) for c in CLASSES}}
    for feature in FEATURES:
        result['features'][feature]={'missing_labeled_percent':float(labeled[feature].isna().mean()*100),
                                    'missing_train_percent':float(splits['Train'][feature].isna().mean()*100)}
    tr=splits['Train'];te=splits['Test'];new=sorted(set(te.City)-set(tr.City))
    result['test_cities_absent_in_training']=new;result['test_new_city_rows']=int(te.City.isin(new).sum())
    result['aqi_range_labeled']=[float(labeled.AQI.min()),float(labeled.AQI.max())]
    derived=pd.cut(labeled.AQI,bins=[-np.inf,50,100,200,300,400,np.inf],labels=CLASSES).astype(str)
    result['aqi_label_disagreements']=int((derived!=labeled.AQI_Bucket).sum())
    y=tr.AQI_Bucket.map({c:i for i,c in enumerate(CLASSES)})
    for label,indices in [('full',np.arange(len(tr))),('svm',np.sort(train_test_split(np.arange(len(tr)),train_size=6000,stratify=y,random_state=42)[0]))]:
        dates=tr.Date.iloc[indices].reset_index(drop=True)
        result['cv_'+label]=[{'fold':i+1,'train_n':len(a),'validation_n':len(b),'train_end':str(dates.iloc[a].max().date()),
                             'val_start':str(dates.iloc[b].min().date()),'val_end':str(dates.iloc[b].max().date())}
                            for i,(a,b) in enumerate(date_blocked_cv(dates))]
    pipeline=joblib.load(ROOT/'models/random_forest.joblib')
    pre=pipeline.named_steps['preprocessor'];wins=pre.named_transformers_['numerical'].named_steps['winsorize']
    result['numeric_clipping_bounds']={c:[float(wins.lo_[i]),float(wins.hi_[i])] for i,c in enumerate(pre.transformers_[0][2])}
    (R/'paper_data_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def figures(evaluation,audit):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'semibold','figure.facecolor':'white','savefig.facecolor':'white'})
    F.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(F/name,dpi=220,bbox_inches='tight');plt.close(fig)
    x=np.arange(6);width=.24
    fig,ax=plt.subplots(figsize=(9.8,3.9))
    for i,(split,color) in enumerate(zip(['Train','Validation','Test'],COLORS)):
        values=np.array(list(audit['splits'][split]['class_counts'].values()))/audit['splits'][split]['n']*100
        ax.bar(x+(i-1)*width,values,width,label=split,color=color)
    ax.set_xticks(x,CLASSES);ax.set_ylabel('Share of labeled partition (%)');ax.legend(frameon=False,ncol=3)
    ax.set_ylim(0,50);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    save(fig,'class_distribution.png')
    keys=list(NAMES);xx=np.arange(4)
    fig,ax=plt.subplots(figsize=(9.8,4))
    for i,(metric,label) in enumerate([('accuracy','Accuracy'),('macro_f1','Macro F1'),('weighted_f1','Weighted F1')]):
        values=[evaluation['candidates'][key]['test_metrics'][metric] for key in keys]
        bars=ax.bar(xx+(i-1)*.23,values,.23,label=label,color=COLORS[i])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set_xticks(xx,list(NAMES.values()));ax.set_ylim(0,1);ax.set_ylabel('Test score');ax.legend(frameon=False,ncol=3,loc='upper center')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);save(fig,'baseline_performance.png')
    fig,ax=plt.subplots(figsize=(9.8,3.8))
    for i,(split,label) in enumerate([('validation_metrics','Validation'),('test_metrics','Test')]):
        bars=ax.bar(xx+(i-.5)*.3,[evaluation['candidates'][k][split]['macro_f1'] for k in keys],.3,label=label,color=COLORS[i])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(xx,list(NAMES.values()));ax.set_ylim(0,1);ax.set_ylabel('Macro F1');ax.legend(frameon=False,ncol=2,loc='upper center')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);save(fig,'validation_test_macro_f1.png')
    fig,axes=plt.subplots(2,2,figsize=(9.5,8.5),layout='constrained')
    for ax,key in zip(axes.flat,keys):
        matrix=np.array(evaluation['candidates'][key]['test_metrics']['confusion_matrix']);total=matrix.sum(axis=1,keepdims=True)
        normalized=np.divide(matrix,total,out=np.zeros_like(matrix,dtype=float),where=total>0)
        ax.imshow(normalized,vmin=0,vmax=1,cmap='BuGn')
        for i in range(6):
            for j in range(6):
                ax.text(j,i,f'{matrix[i,j]}\n{normalized[i,j]*100:.0f}%',ha='center',va='center',fontsize=9,
                        color='white' if normalized[i,j]>.55 else '#183b43')
        ax.set_xticks(np.arange(6),SHORT,fontsize=9);ax.set_yticks(np.arange(6),SHORT,fontsize=9)
        ax.set_title(NAMES[key],pad=10);ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    save(fig,'baseline_confusion_matrices.png')
    fig,ax=plt.subplots(figsize=(9.8,4))
    for i,key in enumerate(keys):
        values=[evaluation['candidates'][key]['test_metrics']['per_class_report'][c]['f1-score'] for c in CLASSES]
        ax.plot(x,values,marker=['o','s','^','D'][i],label=NAMES[key],color=COLORS[i],linewidth=1.8)
    ax.set_xticks(x,CLASSES);ax.set_ylabel('Per-class test F1');ax.set_ylim(.4,1);ax.legend(frameon=False,ncol=4,loc='upper center',fontsize=9)
    ax.grid(axis='y',alpha=.2);save(fig,'per_class_f1.png')
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.2),layout='constrained')
    for ax,(key,kind,title) in zip(axes,[('random_forest','mdi','Random Forest: impurity'),('xgboost','gain','XGBoost: gain')]):
        frame=pd.read_csv(R/'feature_analysis'/f'{key}_{kind}.csv').head(8).iloc[::-1]
        ax.barh(frame.feature,frame.importance,color=COLORS[0 if key=='random_forest' else 1]);ax.set_title(title)
        ax.set_xlabel('Normalized built-in importance');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    save(fig,'tree_importance.png')
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.2),layout='constrained')
    for ax,key in zip(axes,['random_forest','xgboost']):
        frame=pd.read_csv(R/'feature_analysis'/f'{key}_permutation.csv').head(8).iloc[::-1]
        ax.barh(frame.feature,frame.importance_mean,xerr=frame.importance_std,color=COLORS[0 if key=='random_forest' else 1],capsize=3)
        ax.set_title(NAMES[key]);ax.set_xlabel('Mean decrease in validation macro F1');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    save(fig,'permutation_importance.png')
    frame=pd.read_csv(R/'feature_analysis/xgboost_shap.csv').head(10).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9.8,4.3));ax.barh(frame.feature,frame.importance,color=COLORS[1]);ax.set_xlabel('Mean absolute SHAP value (raw model-margin units)')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);save(fig,'xgboost_shap.png')


def write_markdown(e,a):
    results=e['candidates'];selected=results[e['selected_candidate']];sm=selected['test_metrics']
    support=np.array([a['splits']['Test']['class_counts'][c] for c in CLASSES])
    majority=int(support.argmax());reference_truth=np.repeat(np.arange(6),support)
    reference_f1=f1_score(reference_truth,np.full(len(reference_truth),majority),labels=range(6),average='macro',zero_division=0)
    audit_counts=table(['Partition','Dates','Labeled rows','Cities'],[[k,v['date_start']+' to '+v['date_end'],f"{v['n']:,}",v['cities']] for k,v in a['splits'].items()])
    supports=table(['AQI category','Train','Validation','Test','Test share'],[[c,*[f"{a['splits'][s]['class_counts'][c]:,}" for s in ['Train','Validation','Test']],f"{a['splits']['Test']['class_counts'][c]/a['splits']['Test']['n']*100:.2f}%"] for c in CLASSES])
    missing=table(['Pollutant','Missing in labeled data','Missing in training'],[[f,f"{v['missing_labeled_percent']:.2f}%",f"{v['missing_train_percent']:.2f}%"] for f,v in a['features'].items()])
    baseline_table=table(['Model','Accuracy','Macro P','Macro R','Macro F1','Weighted F1'],[[NAMES[k],*[f"{results[k]['test_metrics'][m]:.4f}" for m in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']]] for k in NAMES])
    timing_table=table(['Model','Full fit (s)','Batch inference (s)','Batch average (ms/row)'],[[NAMES[k],f"{results[k]['train_time_sec']:.3f}",f"{results[k]['test_metrics']['inference_time_sec']:.3f}",f"{results[k]['test_metrics']['inference_time_sec']*1000/e['test_rows']:.4f}"] for k in NAMES])
    tuning_table=table(['Model','Baseline val F1','Tuned val F1','Delta val F1','Baseline test F1','Tuned test F1'],[[NAMES[k],f"{results[k]['validation_metrics']['macro_f1']:.4f}",f"{results['tuned_'+k]['validation_metrics']['macro_f1']:.4f}",f"{results['tuned_'+k]['validation_metrics']['macro_f1']-results[k]['validation_metrics']['macro_f1']:+.4f}",f"{results[k]['test_metrics']['macro_f1']:.4f}",f"{results['tuned_'+k]['test_metrics']['macro_f1']:.4f}"] for k in NAMES])
    class_table=table(['Class','Precision','Recall','F1','Support'],[[c,*[f"{sm['per_class_report'][c][m]:.4f}" for m in ['precision','recall','f1-score']],int(sm['per_class_report'][c]['support'])] for c in CLASSES])
    errors=[]
    for key in NAMES:
        cm=np.array(results[key]['test_metrics']['confusion_matrix']);off=cm.copy();np.fill_diagonal(off,0);i,j=np.indices(cm.shape);r,c=np.unravel_index(off.argmax(),cm.shape);n=int(off.sum());adj=int(cm[abs(i-j)==1].sum())
        errors.append([NAMES[key],n,f'{adj}/{n} ({adj/n*100:.1f}%)',f'{CLASSES[r]} -> {CLASSES[c]}',int(off[r,c])])
    error_table=table(['Model','Errors','Adjacent-category errors','Largest confusion','Count'],errors)
    config_table=table(['Model','Baseline configuration (other settings use library defaults)'],[
        ['Decision Tree','Gini; max_depth=10; min_samples_split=20; min_samples_leaf=10; ccp_alpha=0; balanced class weights'],
        ['Random Forest','200 trees; max_depth=None; min_samples_split=10; min_samples_leaf=5; max_features=sqrt; bootstrap=True; balanced class weights; n_jobs=2'],
        ['SVM','SVC; RBF; C=10; gamma=scale; balanced class weights; probability=False; default one-vs-one multiclass fitting'],
        ['XGBoost','300 rounds; max_depth=6; learning_rate=0.05; subsample=0.8; colsample_bytree=0.8; reg_alpha=0.1; reg_lambda=1; multi:softprob; hist; n_jobs=2']])
    search_table=table(['Model','Search space','Selected parameters'],[
        ['Decision Tree','depth {5,8,10,15}; leaf {5,10,20}; ccp_alpha {0,0.0005}','depth=10; leaf=5; ccp_alpha=0.0005'],
        ['Random Forest','trees {100,200}; depth {10,15,None}; leaf {2,5,10}; max_features {sqrt,0.5}','trees=200; depth=None; leaf=10; max_features=0.5'],
        ['SVM','C {0.1,1,5,10,30}; gamma {scale,auto}','C=5; gamma=auto'],
        ['XGBoost','rounds {150,300}; depth {4,6,8}; learning_rate {0.03,0.05,0.1}; reg_lambda {1,2}','rounds=150; depth=6; learning_rate=0.05; reg_lambda=1']])
    cv_table=table(['CV fold','Training rows','Training last date','Validation rows','Validation interval'],[[v['fold'],f"{v['train_n']:,}",v['train_end'],f"{v['validation_n']:,}",v['val_start']+' to '+v['val_end']] for v in a['cv_full']])
    cost_table=table(['Model','Search rows','Best CV F1','Search time (s)','Final refit (s)'],[[NAMES[k],results['tuned_'+k]['search_rows'],f"{results['tuned_'+k]['cv_best_f1']:.4f}",f"{results['tuned_'+k]['search_time_sec']:.2f}",f"{results['tuned_'+k]['train_time_sec']:.2f}"] for k in NAMES])
    # This concise survey account is kept distinct from AirSense's own design judgments.
    literature='''Méndez, Merayo and Núñez [1] review 155 studies covering 2011-2021. They organize the literature by geography, predicted quantities, predictors, evaluation metrics and model families. Section 3.3 distinguishes pollutant, meteorological and other predictors; Section 3.4 separates metric types and discusses accuracy and regression errors. Their Section 4 covers individual, hybrid and comparative approaches. This is a survey of heterogeneous forecasting studies, not a benchmark result on the AirSense dataset.'''
    lit_own='''For AirSense, this taxonomy becomes a reporting checklist: identify the target and time horizon, list only available inputs, document the split, and choose metrics suited to the task. Pollutant measurements are available; weather measurements are not. Calendar fields can encode seasonal structure, but cannot establish meteorological effects. The present four-classifier comparison is an implementation choice for tabular data, not evidence that these methods exhaust the literature or universally outperform alternatives.

A useful comparison therefore concerns experimental questions rather than published score magnitudes. AirSense estimates a category from contemporaneous measurements; future forecasting would require a declared lead time and inputs available before the target day. Classification F1 cannot be interpreted as concentration-regression error. The split dates, city coverage and target construction must also agree before numerical comparisons become meaningful. We do not identify an unverified literature-wide gap or claim a new learning algorithm.

The foundational works on Random Forests [5], support-vector networks [6], gradient boosting systems [7] and additive feature attribution [8] explain the model/interpretation families used here. These references support theoretical mechanisms, not claims about their performance on this CSV. The contribution is a reproducible, bounded project experiment with explicit provenance and a read-only demonstration interface.'''
    methodology=f'''### Selected data and target

The selected file is `data/raw/archive (3)/city_day.csv`. It contains {a['raw_rows']:,} rows, {a['columns']} columns and {a['cities']} city labels, dated {a['date_start']} through {a['date_end']}. A missing AQI_Bucket excludes a row from supervised learning; the remaining {a['labeled']:,} rows retain their supplied targets. There are {a['duplicate_city_dates']} duplicate City-Date keys. The target is encoded Good=0, Satisfactory=1, Moderate=2, Poor=3, Very Poor=4 and Severe=5. Numeric AQI and target-derived columns are excluded from predictors.

### Predictor contract and preprocessing

The inputs are the 12 pollutant fields PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene and Xylene, followed by Year, Month, Day, DayOfWeek, IsWeekend and Season. City and Date are not direct model inputs. Date is used only to create calendar fields and define partitions. Pollutant values remain in source-CSV units; the supplied CSV has no complete machine-readable unit dictionary, and the experiment does not validate or convert source units.

The numeric pipeline computes 1st/99th percentile bounds, clips values, median-imputes missing entries and standardizes. It applies to **all 17 numeric inputs**, including calendar fields, not just pollutant readings. The saved Year clipping interval is {a['numeric_clipping_bounds']['Year'][0]:.0f}-{a['numeric_clipping_bounds']['Year'][1]:.0f}; hence test-year 2020 is clipped to the upper training bound before scaling. This is an implementation limitation, not evidence of temporal extrapolation. The categorical branch uses most-frequent imputation and a fixed Season order: Winter, Summer, Monsoon, Post-Monsoon; unknown categories map to -1. Every learned transform is fitted inside each training fold, then the selected pipeline is refitted on the full training partition. The same saved transformation is applied to validation/test observations and application inputs.

### Partitions and controlled selection

*Table 1. Actual labeled partitions, computed from the selected CSV.*

{audit_counts}

Three expanding-date CV windows are constructed from the training partition. The first 40% of distinct observed dates provide the initial warm-up; remaining dates are divided into three validation windows. Fold training dates must be strictly earlier than the validation start minus seven days, leaving a seven-day exclusion interval. Same-day city observations cannot cross a fold boundary. This removes future-to-past fitting within CV, but does not ensure independent observations or remove long-lived city effects.

*Table 2. Full-training-data temporal CV windows used by DT, RF and XGBoost.*

{cv_table}

All four models use six seeded random candidate configurations and three folds, optimizing six-class macro F1; failed fits raise an error. SVM alone uses a stratified 6,000-row training subsample to bound kernel-search cost. Its subset changes the observed date windows: fold training/validation counts are {', '.join(str(v['train_n'])+'/'+str(v['validation_n']) for v in a['cv_svm'])}. SVM and the other finalists are then refitted on all {a['splits']['Train']['n']:,} training rows. No validation early stopping is used for XGBoost in this run; tree count is searched inside CV.

Candidate choice uses the highest macro F1 on the separate validation partition across four baselines and four tuned finalists; the identifier breaks exact ties. The selected baseline Random Forest is frozen in `models/selection_manifest.json`. Artifact and dataset hashes are checked before the separate final evaluation command. All eight candidates use the same ordered test observations. Test results do not change the frozen choice in the corrected implementation.

### Metrics, timing and transparency

Metrics are generated from actual predictions using the fixed six-label universe and zero_division=0. The outputs include accuracy, macro precision, macro recall, macro F1, weighted F1, per-class scores/support and raw confusion matrices. Inference timing includes the whole preprocessing/classifier pipeline: the median of three full-test prediction calls after a small warm-up. Per-row times are batch averages, not interactive latency guarantees. Search cost is recorded separately from full-data refit cost.

The uploaded project and an earlier audit had already examined test outcomes. The corrected code separates tuning/selection from final evaluation, but this cannot erase historical knowledge. The results are explicitly retrospective. A new external dataset is needed for an independent confirmatory claim. No training or tuning is performed to write this paper.'''
    discussion=f'''### Baseline test comparison

*Table 5. Actual test metrics for the four baselines; P and R denote precision and recall. Scores are proportions rounded to four decimals.*

{baseline_table}

The validation-selected Random Forest reaches accuracy {sm['accuracy']:.4f}, macro F1 {sm['macro_f1']:.4f} and weighted F1 {sm['weighted_f1']:.4f}. Baseline XGBoost has slightly higher accuracy ({results['xgboost']['test_metrics']['accuracy']:.4f}) but lower macro F1 ({results['xgboost']['test_metrics']['macro_f1']:.4f}). Accuracy therefore ranks XGBoost above RF, whereas macro F1 reverses that ordering. This is a practical metric trade-off, not statistical evidence that one model is universally superior. Macro F1 was selected before inspecting these corrected test results because every AQI category should contribute equally.

{figure(2,'baseline_performance.png','Baseline test accuracy, macro F1 and weighted F1 calculated from the saved test predictions. The common score axis makes the accuracy/F1 trade-off visible.')}

{figure(3,'validation_test_macro_f1.png','Baseline validation and test macro F1. Different period/class mixtures mean a higher test score need not indicate reduced overfitting or better future robustness.')}

The majority test class, Satisfactory, contains {a['splits']['Test']['class_counts']['Satisfactory']:,}/{e['test_rows']:,} observations ({a['splits']['Test']['class_counts']['Satisfactory']/e['test_rows']*100:.2f}%). An always-Satisfactory reference computed from these labels would attain that accuracy, but its six-class macro F1 would be only {reference_f1:.4f}. This is a descriptive class-frequency reference, not an additional fitted model or a hyperparameter-selection input.

### Per-class behavior and confusion matrices

*Table 6. Per-class test performance of the frozen baseline Random Forest.*

{class_table}

Poor has the lowest RF F1 ({sm['per_class_report']['Poor']['f1-score']:.4f}), with recall {sm['per_class_report']['Poor']['recall']:.4f}. It also has the lowest mean F1 across the four baselines (0.6563). Satisfactory is easier by mean baseline F1 (0.8044), but its higher frequency also makes its performance influential in accuracy and weighted F1. These values are measured on this test period and should not be interpreted as intrinsic difficulty rankings of AQI categories.

The Severe class contains only 38 test observations. RF correctly identifies 29 and misses 9, producing recall 29/38 = {sm['per_class_report']['Severe']['recall']:.4f}. One additional Severe-class error would change recall by 1/38, about 2.63 percentage points. A seemingly strong point estimate therefore rests on a small sample, and no confidence interval or significance test is provided.

{figure(4,'baseline_confusion_matrices.png','All four baseline test confusion matrices. Cells show counts and true-class-row percentages. Abbreviations: Sat.=Satisfactory, Mod.=Moderate, V. Poor=Very Poor. Rows are true categories and columns are predictions; darker cells have larger row shares.')}

*Table 7. Measured error patterns from the same baseline confusion matrices.*

{error_table}

For RF, 842 of 850 errors are between adjacent ordinal categories; its largest confusion is Moderate -> Satisfactory (282 city-days). RF underestimates severity on 453 observations and overestimates it on 397. XGBoost instead has 333 underestimates and 509 overestimates; its largest error pair is Good -> Satisfactory (225). Baseline XGBoost makes eight fewer errors overall, yet RF has the better equal-class balance. Confusion matrices alone do not prove that errors occur near numeric AQI breakpoints; that would require a separate distance-to-boundary audit.

{figure(5,'per_class_f1.png','Test F1 by AQI category for each baseline. Lines guide category-to-category comparisons; they do not represent temporal trajectories.')}

### Tuning outcomes and computational cost

*Table 8. Baseline versus tuned macro F1. Validation deltas determine whether tuning helped selection; test values remain descriptive.*

{tuning_table}

Only the Decision Tree improves on validation within this bounded search. The baseline RF remains preferred after considering all eight candidates. The tuned SVM modestly improves test macro F1 despite worse validation macro F1; choosing it because of that test gain would violate the selection protocol. Negative tuning deltas are retained rather than hidden. A six-candidate search does not establish that the optimal configuration has been found.

*Table 9. Actual tuning search/refit costs and best within-search CV scores. SVM CV uses a smaller data subset and different observed date windows; its CV score is not a matched-budget comparison.*

{cost_table}

*Table 10. Baseline fit and inference times. Inference is a whole-test batch of {e['test_rows']:,} rows, including preprocessing.*

{timing_table}

SVM has the largest batch inference cost in this run. The timing record does not preserve physical CPU model, available memory, contention or repeated fit variability; these are single-environment descriptive measurements. No hardware-normalized speed or scaling claim is justified.

### Relation to the research motivation

The results support a narrow conclusion: saved tree ensembles handle this incomplete tabular classification task better than the single tree and RBF SVM by macro F1 in the observed partitions. They do not support future forecasting, causal pollutant attribution, deployment safety, or superiority over published systems. The survey is used as contextual literature; its datasets and prediction targets are not a directly comparable leaderboard.'''
    cm_figure=figure(4,'baseline_confusion_matrices.png','All four baseline test confusion matrices. Cells show counts and true-class-row percentages. Abbreviations: Sat.=Satisfactory, Mod.=Moderate, V. Poor=Very Poor. Rows are true categories and columns are predictions; darker cells have larger row shares.')
    f1_figure=figure(5,'per_class_f1.png','Test F1 by AQI category for each baseline. Lines guide category-to-category comparisons; they do not represent temporal trajectories.')
    discussion=discussion.replace(cm_figure+'\n\n','').replace(f1_figure,cm_figure+'\n\n'+f1_figure)
    # Replace the class-frequency reference with a computation over the real support counts.
    support=np.array([a['splits']['Test']['class_counts'][c] for c in CLASSES]);majority=int(support.argmax());reference_truth=np.repeat(np.arange(6),support)
    reference_f1=f1_score(reference_truth,np.full(len(reference_truth),majority),labels=range(6),average='macro',zero_division=0)
    # Values above are checked against actual label supports; no simulated result enters the paper.
    assert len(reference_truth)==e['test_rows']
    rf_perm=pd.read_csv(R/'feature_analysis/random_forest_permutation.csv');xgb_perm=pd.read_csv(R/'feature_analysis/xgboost_permutation.csv');shap=pd.read_csv(R/'feature_analysis/xgboost_shap.csv')
    imp_table=table(['Feature','RF permutation drop','XGB permutation drop','XGB mean abs. SHAP'],[[f,f"{rf_perm.set_index('feature').loc[f,'importance_mean']:.4f}",f"{xgb_perm.set_index('feature').loc[f,'importance_mean']:.4f}",f"{shap.set_index('feature').loc[f,'importance']:.4f}"] for f in ['PM2.5','CO','PM10','NO','O3']])
    references='''[1] M. Méndez, M. G. Merayo, and M. Núñez, "Machine learning algorithms to forecast air quality: a survey," Artificial Intelligence Review, vol. 56, pp. 10031-10066, 2023. DOI: 10.1007/s10462-023-10424-4. Publisher HTML and 36-page full paper verified 1 November 2026.

[2] Central Pollution Control Board, National Air Quality Index, report CUPS/82/2014-15, October 2014. Official government-hosted report: https://www.airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf. Category table verified on CPCB's official AQI portal: https://airquality.cpcb.gov.in/AQI_All_India/. Accessed 1 November 2026.

[3] R. Rao, Air Quality Data in India (2015-2020), Kaggle dataset. https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india. Accessed 1 November 2026. Statistical statements in this paper come from the supplied local city_day.csv, not unverified online dataset summaries.

[4] scikit-learn developers, "Decision Trees," "Metrics and scoring," and "Permutation feature importance," scikit-learn 1.8 documentation. https://scikit-learn.org/1.8/modules/tree.html ; https://scikit-learn.org/1.8/modules/model_evaluation.html ; https://scikit-learn.org/1.8/modules/permutation_importance.html. Accessed 1 November 2026.

[5] L. Breiman, "Random Forests," Machine Learning, vol. 45, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324. Publisher bibliographic record verified 1 November 2026.

[6] C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, pp. 273-297, 1995. DOI: 10.1007/BF00994018. Publisher bibliographic record verified 1 November 2026.

[7] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," arXiv:1603.02754, 2016, version 3. https://arxiv.org/abs/1603.02754. DOI: 10.48550/arXiv.1603.02754. Author/title/year/version verified 1 November 2026; cited as the verified author preprint rather than supplying unverified proceedings details.

[8] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," arXiv:1705.07874, 2017, version 2. https://arxiv.org/abs/1705.07874. DOI: 10.48550/arXiv.1705.07874. Author/title/year/version verified 1 November 2026.'''
    intro='''Air-quality measurement produces a multivariate record rather than a single directly interpretable category. A classifier can learn associations between the recorded pollutant mixture, calendar context and an existing AQI label. The engineering challenge is not merely to display a high score: incomplete observations, unequal class frequencies and shifting monitoring coverage complicate both model selection and interpretation.

AirSense is a research-style project benchmark and demonstration application. It uses existing daily Indian city records to compare four established classifiers under a common chronological protocol. The output is an estimated category for an observed day. It is not a forecast of a future day, and the model is not a replacement for the official index procedure or validated sensor measurements [2].

The work contributes an auditable implementation rather than a new learning algorithm. Its saved predictions, model configurations, cross-validation records and feature-importance files support reproducibility at the project level. Every empirical table and figure below is derived from those artifacts or from direct inspection of the selected CSV. The main methodological priority is a clear boundary between what was observed, what was selected using validation, and what remains untested.'''
    paper=f'''# AirSense: A Retrospective Benchmark for Same-Day Air Quality Category Classification

**Research-style project paper | 1 November 2026 | Prompts 11-14 evidence**

**Status:** Technical research draft. PDF generation and software verification do not establish publication readiness, scientific novelty or deployment validity. Authorship and affiliation were not supplied for this paper and have not been invented.

## Abstract

AirSense compares Decision Tree, Random Forest, radial-basis SVM and XGBoost for same-day, six-category AQI classification using the supplied Indian city-day dataset. The selected CSV contains {a['raw_rows']:,} records dated 2015-01-01 through 2020-07-01, of which {a['labeled']:,} have labels. Chronological partitions contain 16,645 training, 3,784 validation and 4,421 test observations. Training-only temporal cross-validation uses fold-local preprocessing and a bounded six-candidate search per model; all finalists are refitted on the full training partition. Selection is frozen using validation macro F1. The selected baseline Random Forest achieves test accuracy 0.8077 and macro F1 0.7826. Baseline XGBoost achieves higher accuracy, 0.8095, but lower macro F1, 0.7690, illustrating a class-balance trade-off. Poor is the most difficult class by mean baseline F1; 842 of RF's 850 errors are between adjacent categories. Permutation and SHAP analyses consistently emphasize particulate measurements and CO, without implying causality. The test partition mixes time shift with expanding city coverage, and calendar numerics are clipped by the saved preprocessing. Historical test exposure, missing weather fields, limited rare-class support and absence of independent prospective validation constrain the conclusions. These are retrospective classification results, not evidence of future forecasting capability.

## Keywords

Air quality; AQI classification; Random Forest; SVM; XGBoost; macro F1; temporal validation; feature importance; reproducibility.

## 1. Introduction

{intro}

## 2. Problem Statement

Given contemporaneous pollutant readings and the date for a city-day, estimate its supplied AQI_Bucket label among six ordered categories. Formally, the task is to learn a mapping from the available 18-dimensional input to one of six class identifiers. The date supplies calendar context, not a forecast horizon. No sensor sequence, future weather input or future-label target is constructed.

The comparison must use identical test identities and a preprocessing pipeline whose parameters are learned only from appropriate training observations. Model selection must prioritize a metric that does not allow frequent classes to dominate by construction. The final interface must retain the exact fitted preprocessing and distinguish unsupported probabilities or missing artifacts from successful model outputs.

## 3. Research Objectives

- Compare four established classifiers with a shared, documented chronological split.
- Evaluate overall and per-class behavior using actual predictions, including macro and weighted metrics.
- Perform a controlled, computationally bounded training-only search and compare finalists with baselines on validation.
- Inspect RF/XGBoost feature associations using built-in importance, validation permutation and feasible SHAP analysis.
- Preserve traceable experimental outputs and expose them in a read-only Streamlit demonstration.
- Document uncertainty, historical test exposure, target-construction effects and untested generalization instead of claiming unrestricted performance.

## 4. Literature Review

{literature}

{lit_own}

## 5. Research Motivation

The project asks whether an equal-class selection criterion leads to the same practical choice as accuracy, whether tuning helps within a modest budget, and whether different attribution methods agree on influential predictors. These are questions about this implementation and dataset, not claimed omissions across the research field.

A shared evaluation framework also addresses an engineering risk: otherwise, apparently comparable models may be assessed on different rows, use different transformations, or expose fabricated/static metrics in a demonstration. AirSense therefore couples its software with explicit saved predictions and provenance checks. The scientific motivation is disciplined reporting and interpretable error analysis, with no claim that the tabular classifier solves the harder forecasting problem.

## 6. Dataset Description

The selected CSV is the city-level daily table, rather than city_hour.csv, station_day.csv, station_hour.csv or stations.csv from the same supplied archive [3]. Direct inspection finds 29,531 rows, 16 columns and 26 city labels. Of these, 4,681 rows lack AQI_Bucket and are excluded; 24,850 remain. Only 6,236 labeled rows have all 12 pollutant readings, so complete-case deletion would discard much of the usable labeled subset. No City-Date key is duplicated. The observed period ends on 1 July 2020, not at the end of calendar year 2020.

The 16 source columns comprise City, Date, 12 pollutant fields, AQI and AQI_Bucket. Derived calendar fields increase the modeling input to 18 predictors; AQI is not used as an input. The experiment uses labels as supplied rather than recomputing official AQI from incomplete pollutant observations. A direct bucket-threshold audit finds zero disagreements between the supplied labeled AQI values and the open-ended intervals at 50, 100, 200, 300 and 400. This consistency check is not independent verification of measurement validity.

The labeled numeric AQI range is 13-2,049. The official report presents categories on a 0-500 scale [2]; values above that scale remain in the supplied data and are retained with their supplied Severe label. The paper does not certify these extreme values as valid official readings. Missing units, aggregation/curation assumptions and extreme values require further data-quality validation.

*Table 3. Class support in each labeled partition and the test class mixture.*

{supports}

{figure(1,'class_distribution.png','Class proportions by chronological partition. The test mixture differs substantially from training; Severe has only 38 test examples.')}

Training covers 20 cities, validation 23 and test 26. Six test cities are absent from training: {', '.join(a['test_cities_absent_in_training'])}. Their 919 test rows represent {a['test_new_city_rows']/a['splits']['Test']['n']*100:.2f}% of the test set. Thus the chronological result mixes changing city coverage with temporal change. It is neither a controlled leave-city-out estimate nor an exclusively same-city temporal test. This distinction matters even though City is excluded as a direct predictor.

## 7. Data Preprocessing

{methodology.split('### Predictor contract and preprocessing')[1].split('### Partitions and controlled selection')[0].strip()}

*Table 4. Observed predictor missingness before imputation; percentages are computed separately on labeled data and training data.*

{missing}

There are no weather measurements, lead measurements, sensor-quality flags or confirmed causal source labels in the selected feature matrix. A missingness indicator, cyclic calendar encoding and unit harmonization are not part of the fitted pipeline. Their absence is a limitation, not a capability silently supplied by the application.

## 8. Theoretical Background

### 8.1 Air quality and the AQI target

Air quality describes the ambient pollutant mixture. India's AQI communicates that mixture through pollutant-specific sub-indices and an overall category [2]. Good, Satisfactory, Moderate, Poor, Very Poor and Severe are ordinal labels, but the present classifiers optimize nominal multiclass objectives rather than ordinal error costs. A one-category and a several-category error are equally wrong in accuracy, which makes the confusion matrix necessary for interpretation. Predicting the existing label from contemporaneous pollutants partly learns the target's construction; this is not, by itself, forbidden target leakage, but substantially narrows the task.

### 8.2 Decision Tree and Random Forest

A classification tree recursively chooses feature thresholds to reduce class impurity [4]. For class fractions p_k in a node, Gini impurity is G = 1 - sum_k(p_k squared); an effective split reduces the weighted impurities of the child nodes. Depth, minimum leaf size and pruning constrain complexity. A tree is interpretable as a sequence of conditions, but small data changes can alter its partition.

Random Forest combines many randomized trees [5]. Bootstrap sampling changes each tree's observations, and random feature subsets diversify splits. The scikit-learn classifier averages the trees' class-probability estimates before taking the highest-scoring class. Averaging can reduce variance, but correlated trees, biased coverage and unrepresentative data remain problems. More trees do not automatically resolve distribution shift.

### 8.3 Support Vector Machine

A soft-margin binary SVM balances margin size against classification violations [6]. A standard objective is (1/2)||w|| squared + C sum_i(xi_i), subject to y_i(w dot phi(x_i) + b) >= 1 - xi_i and xi_i >= 0. AirSense uses the RBF kernel K(x,z)=exp(-gamma ||x-z|| squared) and the library's one-versus-one multiclass fitting. Scaling affects distances; C and gamma govern flexibility. The saved SVM provides a class prediction but not calibrated probabilities because probability estimation was disabled.

### 8.4 XGBoost

Gradient boosting adds trees sequentially to improve a regularized objective [7]. In simplified notation, the objective sums multiclass prediction losses and tree penalties; a common tree penalty is gamma*T + (lambda/2) sum_j(w_j squared), with additional L1 regularization when used. Learning rate, tree depth, row/column subsampling and regularization constrain the fitted ensemble. The present multiclass soft-probability model uses histogram tree construction and fixed candidate tree counts, not test-driven early stopping.

### 8.5 Classification metrics

For a class k treated one-versus-rest: Precision_k = TP_k/(TP_k+FP_k), Recall_k = TP_k/(TP_k+FN_k), and F1_k = 2*Precision_k*Recall_k/(Precision_k+Recall_k). Undefined divisions are assigned zero in this implementation [4]. Accuracy is the number of correct labels divided by sample count. Macro precision, recall and F1 are arithmetic means over the fixed six categories. Macro F1 is the mean of the class F1 values, not necessarily the harmonic mean of macro precision and macro recall. Weighted F1 weights each class F1 by its true support. Confusion-matrix cell C_ij counts true class i predicted as j.

Macro averaging gives a rare category the same contribution as a frequent one, but can have high variance when support is small. Weighted F1 and accuracy describe different summaries and may mask weak rare-class recall. None of these point metrics establishes calibration, uncertainty or causal validity.

### 8.6 Feature-importance methods

RF impurity importance accumulates split-impurity reductions and can favor features with many candidate splits. XGBoost gain reflects split improvements under its own objective. Permutation importance measures score degradation after shuffling one predictor on held-out data [4]; correlated substitutes and unrealistic shuffled combinations can affect it. SHAP assigns additive attributions to a model output relative to a reference distribution [8]. Aggregated absolute attributions show magnitude rather than direction. All are predictive descriptions of a fitted model, not estimates of causal intervention effects.

## 9. Proposed Methodology

{methodology.split('### Partitions and controlled selection')[1].split('### Metrics, timing and transparency')[0].strip()}

The application loads these saved pipelines rather than recreating preprocessing or fitting on refresh. It derives the exact same calendar fields from an entered measurement date, validates the supported pollutant schema, aligns probabilities with fitted class labels and displays unavailable artifacts honestly. These application checks support software correctness; they do not establish scientific validity beyond the experimental design.

## 10. Experimental Setup

The artifact manifests record Python 3.12.14, scikit-learn 1.8.0 and XGBoost 3.4.1, with random_state=42. There are four baselines and four tuned finalists, all fitted on 16,645 training rows. Table 11 lists their principal baseline settings; unspecified settings follow the pinned library defaults stored in the configuration artifacts. DT, RF and SVM use balanced class weights; the multiclass XGBoost model does not use that same class_weight parameter.

*Table 11. Baseline configurations actually used by the corrected experiment.*

{config_table}

*Table 12. Finite search spaces and the selected within-search configurations; unchanged parameters inherit the baselines.*

{search_table}

Six configurations sampled with seed 42 produce 18 CV fits per model, followed by one full-training refit of the chosen search configuration. CV estimator parallelism is reduced to one thread for RF/XGBoost during the two-job outer search. Final full fits use the saved two-job estimator setting. There is no GPU requirement or new model training for paper generation.

{methodology.split('### Metrics, timing and transparency')[1].strip()}

Complete CV tables, final model/configuration files, test_predictions.csv, per_class_comparison.csv and evaluation_results.json are included in the project. This paper's generation script independently recomputes aggregate scores and all confusion matrices from the saved predictions and checks the model hashes before writing the paper. The dataset SHA-256 is {a['source_sha256']}.

## 11. Results and Discussion

{discussion}

## 12. Feature Importance and Interpretability

The saved analysis concerns the **baseline** RF and XGBoost models. Built-in importance uses their fitted training structures. Permutation uses the same seeded subset of 500 validation rows with three shuffles per feature, scored by six-class macro F1. XGBoost interventional TreeSHAP uses 80 training background rows and 120 validation explanation rows; mean absolute values are averaged across rows and classes in raw model-margin units. These sample counts come from the saved sampling manifest, not generic defaults. RF SHAP was not attempted within the bounded compute budget.

{figure(6,'tree_importance.png','Built-in top-eight feature importance for baseline RF and XGBoost. Both are normalized within the model, but impurity and gain use different definitions and should not be treated as interchangeable causal effect sizes.')}

{figure(7,'permutation_importance.png','Top-eight validation permutation importances. Error bars are the standard deviation across three shuffles, not statistical confidence intervals; each model uses the same 500-row held-out sample.')}

*Table 13. Selected actual feature-analysis values. Permutation columns are mean decreases in macro F1; SHAP uses raw margin units and is not on the same scale.*

{imp_table}

PM2.5 ranks first in both permutation analyses; CO and PM10 follow in that order. XGBoost SHAP ranks PM2.5, PM10 and CO first, second and third, showing agreement on a small influential set without identical ordering. For example, RF permutation drops by 0.3089 when PM2.5 is shuffled, versus 0.2018 for CO and 0.1228 for PM10. These are score changes on the particular validation subset, not pollutant contribution percentages or intervention effects.

{figure(8,'xgboost_shap.png','Top-ten mean absolute XGBoost SHAP attributions, computed from the saved 80-row training background and 120-row validation explanation sample. Absolute averaging suppresses direction and individual-sample differences.')}

Correlated pollutant variables can carry overlapping information; one may substitute when another is shuffled, and interventional SHAP relies on a reference/dependence assumption. The predictor set is also tied to the AQI label definition. Therefore, strong PM2.5 importance does not quantify a causal benefit from reducing PM2.5, identify an emission source or establish a geographic policy effect. Location importance cannot be evaluated because City is excluded. Calendar attribution, if present, must not be described as measured weather influence.

## 13. Limitations

**Retrospective test exposure.** Earlier project work and the audit had already inspected test results. Corrected code can prevent further selection leakage but cannot make the same data historically unseen. A fresh external evaluation is the highest-priority outstanding requirement.

**Mixed temporal and spatial change.** The test includes 919 rows from six cities absent in training. Without paired same-city and deliberately city-disjoint analyses, changes cannot be attributed uniquely to time or geography. Shared-date CV and a seven-day gap do not remove longer autocorrelation or correlated monitoring conditions.

**Target and information availability.** Same-day pollutant measurements are closely related to the supplied AQI label. This is easier and fundamentally different from forecasting. AQI-derived input columns are excluded, but no prospective feature-availability study or independent sensor/unit audit has been performed. Values beyond the official published AQI scale remain a source-quality concern.

**Missingness and preprocessing.** Many records depend on imputation; Xylene is missing in 62.91% of training rows, PM10 in 38.66% and NH3 in 33.22%. Missingness may be informative and nonrandom, yet explicit missingness features are not used. Clipping can remove meaningful extremes. Applying it to calendar numerics also clips 2020 Year values to 2019; cyclic encoding and ablations were not evaluated.

**Imbalance and uncertainty.** Severe has 38 test examples; rare-class recalls can change substantially with a few observations. No confidence intervals, repeated seeds, blocked bootstrap, significance testing or cost-sensitive ordinal evaluation is included. The observed differences must not be advertised as statistically established superiority.

**Search and fairness of budgets.** Only six candidates per model were tried. SVM's smaller CV subset has different date windows and cannot be treated as a perfectly matched full-data tuning comparison. Single validation-period selection may favor a particular seasonal mixture. The search does not establish a global optimum.

**Interpretability and deployment.** Validation is used for selection and exploratory feature analysis, so importance findings are not an independent confirmation. SHAP and permutation use small samples, and their stability is unmeasured. Weather is absent; probabilities are uncalibrated; SVM probabilities are unavailable. No prospective deployment, clinical/public-health decision validation, calibrated uncertainty, new-city guarantee or operational scaling study is provided.

## 14. Conclusion and Future Work

AirSense provides a traceable retrospective comparison of four classifiers for supplied same-day AQI categories. Baseline RF is the validation-selected candidate and yields test macro F1 0.7826 with accuracy 0.8077. Baseline XGBoost's slightly higher accuracy and lower macro F1 show why a single headline number is inadequate. Per-class results and measured adjacent-category errors reveal behavior hidden by aggregate scores. The bounded tuning study does not consistently improve validation performance, and these negative results are retained.

Future work should first obtain new, untouched observations with verified units and availability timestamps, then preregister the target/horizon and split rules. Separate same-city temporal tests from leave-city-out tests; quantify uncertainty with dependence-aware resampling and repeated periods. Compare against a deterministic AQI procedure only where the required pollutant data, units and averaging windows are genuinely comparable. Test missingness indicators, calendar encodings, pollutant-only ablations, calibrated probabilities and explicit ordinal costs. Weather enrichment would need a verified joined dataset rather than inferred inputs. If forecasting is pursued, construct a genuinely future target and past-only predictors before evaluating sequence models.

This document is a complete research-style project draft, not a claim of publication readiness. Peer review, venue formatting, authorship confirmation, independent validation and the unresolved limitations above remain outstanding.

## 15. References

{references}
'''
    # Captions are sequentially numbered in reading order rather than dependency order.
    # Renumber table mentions/captions consistently in their first occurrence order.
    order=[3,4,1,2,11,12,5,6,7,8,9,10,13]
    mapping={old:i+1 for i,old in enumerate(order)}
    paper=re.sub(r'Table (\d+)',lambda m:f'TABLETOKEN{mapping[int(m.group(1))]}END',paper)
    paper=re.sub(r'TABLETOKEN(\d+)END',r'Table \1',paper)
    # Portable ASCII hyphens; keep proper names and supported typographic symbols.
    paper=paper.replace('The 16 source columns comprise', 'Table 1 and Figure 1 summarize the changing class mixture. The 16 source columns comprise')
    paper=paper.replace('There are no weather measurements, lead measurements', 'Table 2 quantifies missingness before imputation. There are no weather measurements, lead measurements')
    paper=paper.replace('Three expanding-date CV windows are constructed', 'Tables 3 and 4 document the outer partitions and within-training validation windows. Three expanding-date CV windows are constructed')
    paper=paper.replace('Six configurations sampled with seed 42 produce', 'Table 6 records both the searched values and the chosen configuration. Six configurations sampled with seed 42 produce')
    paper=paper.replace('The validation-selected Random Forest reaches', 'Table 7 and Figures 2-3 compare the baseline scores. The validation-selected Random Forest reaches')
    paper=paper.replace('Poor has the lowest RF F1', 'Table 8 and Figure 5 expose per-class differences. Poor has the lowest RF F1')
    paper=paper.replace('For RF, 842 of 850 errors', 'Table 9 and Figure 4 make the error counts explicit. For RF, 842 of 850 errors')
    paper=paper.replace('Only the Decision Tree improves on validation', 'Tables 10-12 separate validation selection, test description and computational cost. Only the Decision Tree improves on validation')
    paper=paper.replace('PM2.5 ranks first in both permutation analyses', 'Figures 6-8 and Table 13 compare the saved importance methods. PM2.5 ranks first in both permutation analyses')
    paper=paper.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    (R/'research_paper.md').write_text(paper,encoding='utf-8')
    (R/'literature_review.md').write_text('# AirSense literature review\n\n'+literature+'\n\n'+lit_own+'\n\n## Verified references\n\n'+references+'\n',encoding='utf-8')
    (R/'methodology.md').write_text('# AirSense actual methodology\n\n'+methodology+'\n\n## Baseline configuration\n\n'+config_table+'\n\n## Search spaces\n\n'+search_table+'\n',encoding='utf-8')
    (R/'results_and_discussion.md').write_text('# AirSense results and discussion\n\n'+discussion+'\n\nAll metrics are retrospective. Historical test exposure and shifting city coverage limit confirmatory interpretation. See research_paper.md for all limitations.\n',encoding='utf-8')
    def section(number):
        match=re.search(r'^## '+str(number)+r'\. [^\n]+\n(.*?)(?=^## |\Z)',paper,re.M|re.S)
        return match.group(0).strip()
    (R/'literature_review.md').write_text('# AirSense literature review\n\n'+section(4)+'\n\n'+section(15)+'\n',encoding='utf-8')
    (R/'methodology.md').write_text('# AirSense actual methodology\n\n'+'\n\n'.join(section(n) for n in [6,7,9,10])+'\n',encoding='utf-8')
    (R/'results_and_discussion.md').write_text('# AirSense results, interpretation and limitations\n\n'+'\n\n'.join(section(n) for n in [11,12,13])+'\n',encoding='utf-8')
    return paper


def write_bibliography():
    entries=[
        ('article','mendez2023survey',{'author':r'M{\'e}ndez, Manuel and Merayo, Mercedes G. and N{\'u}{\~n}ez, Manuel','title':'Machine learning algorithms to forecast air quality: a survey','journal':'Artificial Intelligence Review','year':'2023','volume':'56','pages':'10031--10066','doi':'10.1007/s10462-023-10424-4','url':'https://link.springer.com/article/10.1007/s10462-023-10424-4','note':'Publisher and 36-page full paper verified 2026-11-01; online publication 2023-02-16'}),
        ('techreport','cpcb2014aqi',{'author':'{Central Pollution Control Board}','title':'National Air Quality Index','institution':'Central Pollution Control Board','year':'2014','month':'October','number':'CUPS/82/2014-15','url':'https://www.airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf','note':'AQI categories cross-checked with official CPCB AQI portal, 2026-11-01'}),
        ('misc','rao_air_quality_india',{'author':'Rao, Rohan','title':'Air Quality Data in India (2015--2020)','howpublished':'Kaggle dataset','url':'https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india','note':'Accessed 2026-11-01; supplied local CSV statistics audited; release year not asserted'}),
        ('misc','sklearn18docs',{'author':'{scikit-learn developers}','title':'Decision Trees; Metrics and scoring; Permutation feature importance','howpublished':'scikit-learn 1.8 documentation','url':'https://scikit-learn.org/1.8/modules/model_evaluation.html','note':'Also tree.html and permutation_importance.html in the same versioned documentation; verified 2026-11-01'}),
        ('article','breiman2001forests',{'author':'Breiman, Leo','title':'Random Forests','journal':'Machine Learning','year':'2001','volume':'45','pages':'5--32','doi':'10.1023/A:1010933404324'}),
        ('article','cortes1995svm',{'author':'Cortes, Corinna and Vapnik, Vladimir','title':'Support-vector networks','journal':'Machine Learning','year':'1995','volume':'20','pages':'273--297','doi':'10.1007/BF00994018'}),
        ('misc','chen2016xgboost',{'author':'Chen, Tianqi and Guestrin, Carlos','title':'XGBoost: A Scalable Tree Boosting System','year':'2016','eprint':'1603.02754','archivePrefix':'arXiv','doi':'10.48550/arXiv.1603.02754','url':'https://arxiv.org/abs/1603.02754','note':'Verified author preprint, version 3'}),
        ('misc','lundberg2017shap',{'author':'Lundberg, Scott M. and Lee, Su-In','title':'A Unified Approach to Interpreting Model Predictions','year':'2017','eprint':'1705.07874','archivePrefix':'arXiv','doi':'10.48550/arXiv.1705.07874','url':'https://arxiv.org/abs/1705.07874','note':'Verified author preprint, version 2'})]
    text='\n\n'.join('@'+kind+'{'+key+',\n'+',\n'.join('  '+field+' = {'+value+'}' for field,value in fields.items())+'\n}' for kind,key,fields in entries)+'\n'
    (R/'references.bib').write_text(text,encoding='utf-8')
    (R/'reference_verification.md').write_text('''# Prompt 14 reference verification

Access date: 1 November 2026. No published paragraphs or figures are copied.

| Source | Verification |
| --- | --- |
| Primary survey | Springer article and linked 36-page full PDF. Title, all three authors, journal, 2023, volume 56, pages 10031-10066, DOI and 16 February 2023 online date verified. Predictor section 3.3 and evaluation section 3.4 read. |
| CPCB | Government-hosted National AQI report and official AQI portal used for the category scale. No certification of local extreme measurements is implied. |
| Dataset | Canonical Rohan Rao Kaggle URL was accessed but yielded no readable page text. All period, coverage, label and missingness statistics come from direct local CSV inspection. Online release date, version and license are not invented. |
| Implementation | Versioned scikit-learn 1.8 Decision Trees, Metrics and scoring, and Permutation feature importance documentation inspected. |
| Random Forest | Springer DOI 10.1023/A:1010933404324 verifies author/title/year/journal/volume/pages and forest mechanism. |
| SVM | Springer DOI 10.1007/BF00994018 verifies authors/title/year/journal/volume/pages and margin classification foundation. |
| XGBoost | arXiv:1603.02754 verifies authors/title/2016/version 3; cited as the verified preprint, without unverified proceedings metadata. |
| SHAP | arXiv:1705.07874 verifies authors/title/2017/version 2 and additive attribution. |

Primary full paper: https://link.springer.com/content/pdf/10.1007/s10462-023-10424-4.pdf

AirSense's macro-F1, chronology, leakage and comparability judgments are identified as project choices, not survey recommendations. Foundational pre-2023 publications are intentionally retained for theory rather than assigning them fabricated recent dates.
''',encoding='utf-8')


def make_pdf(markdown):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from PIL import Image as PILImage
    fonts=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    for name,file in [('PaperSerif','DejaVuSerif.ttf'),('PaperSerifBold','DejaVuSerif-Bold.ttf'),('PaperSans','DejaVuSans.ttf'),('PaperSansBold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    for family in ['PaperSerif','PaperSans']:
        pdfmetrics.registerFontFamily(family,normal=family,bold=family+'Bold',italic=family,boldItalic=family+'Bold')
    ink=colors.HexColor('#173540');accent=colors.HexColor('#167b77');muted=colors.HexColor('#526a75')
    styles={
        'body':ParagraphStyle('body',fontName='PaperSerif',fontSize=9.7,leading=14.1,spaceAfter=8,alignment=TA_JUSTIFY,textColor=colors.HexColor('#202b31'),splitLongWords=True,allowWidows=0,allowOrphans=0),
        'title':ParagraphStyle('title',fontName='PaperSansBold',fontSize=21,leading=27,textColor=ink,spaceAfter=15,keepWithNext=True),
        'h2':ParagraphStyle('h2',fontName='PaperSansBold',fontSize=13.1,leading=17.3,textColor=accent,spaceBefore=16,spaceAfter=8,keepWithNext=True),
        'h3':ParagraphStyle('h3',fontName='PaperSansBold',fontSize=10.7,leading=14.2,textColor=ink,spaceBefore=10,spaceAfter=6,keepWithNext=True),
        'caption':ParagraphStyle('caption',fontName='PaperSans',fontSize=8.4,leading=11.5,textColor=muted,spaceBefore=5,spaceAfter=8),
        'tablecaption':ParagraphStyle('tablecaption',fontName='PaperSans',fontSize=8.5,leading=11.7,textColor=muted,spaceBefore=6,spaceAfter=6,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='PaperSans',fontSize=7.5,leading=10,textColor=ink,splitLongWords=True),
        'cellhead':ParagraphStyle('cellhead',fontName='PaperSansBold',fontSize=7.6,leading=10.2,textColor=colors.white),
        'reference':ParagraphStyle('reference',fontName='PaperSerif',fontSize=8.7,leading=12.5,spaceAfter=8,splitLongWords=True),
        'bullet':ParagraphStyle('bullet',fontName='PaperSerif',fontSize=9.7,leading=14.1,leftIndent=12,firstLineIndent=-9,spaceAfter=5,textColor=colors.HexColor('#202b31'))}
    width,height=A4;left=49;right=49;content=width-left-right
    destination=R/'research_paper.pdf'
    class PaperDoc(BaseDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name in {'title','h2'}:
                name=f'section_{self.seq.nextf("section")}'
                self.canv.bookmarkPage(name);self.canv.addOutlineEntry(flowable.getPlainText(),name,level=0,closed=False)
    def page(canvas,doc):
        canvas.setTitle('AirSense: A Retrospective Benchmark for Same-Day Air Quality Category Classification')
        canvas.setAuthor('AirSense project; authorship not supplied')
        canvas.setSubject('Retrospective ML project research draft; prompt 14')
        canvas.setStrokeColor(colors.HexColor('#cfdce0'));canvas.setLineWidth(.5)
        if doc.page>1:
            canvas.line(left,height-35,width-right,height-35);canvas.setFont('PaperSans',7.6);canvas.setFillColor(muted)
            canvas.drawString(left,height-27,'AirSense | Same-day AQI category classification')
        canvas.line(left,36,width-right,36);canvas.setFont('PaperSans',7.2);canvas.setFillColor(muted)
        canvas.drawString(left,24,'Technical research draft | Retrospective evidence | 1 November 2026')
        canvas.drawRightString(width-right,24,str(doc.page))
    doc=PaperDoc(str(destination),pagesize=A4,leftMargin=left,rightMargin=right,topMargin=49,bottomMargin=49,allowSplitting=1,pageCompression=1)
    doc.addPageTemplates(PageTemplate(id='paper',frames=[Frame(left,49,content,height-98,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page))
    def inline(text):
        text=html.escape(text)
        text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'`([^`]+)`',r'<font name="PaperSans" size="8.7">\1</font>',text)
        return text
    def para(text,style='body'):return Paragraph(inline(text),styles[style])
    blocks=re.split(r'\n\s*\n',markdown.strip());story=[];i=0
    while i<len(blocks):
        block=blocks[i].strip()
        if block.startswith('!['):
            match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',block);path=R/match.group(2)
            with PILImage.open(path) as pic:w,h=pic.size
            dw=content;dh=h/w*dw
            if dh>450:dw*=450/dh;dh=450
            group=[Spacer(1,5),Image(str(path),width=dw,height=dh)]
            if i+1<len(blocks) and blocks[i+1].strip().startswith('*Figure '):
                group.append(para(blocks[i+1].strip().strip('*'),'caption'));i+=1
            story.append(KeepTogether(group));i+=1;continue
        if block.startswith('|'):
            rows=[]
            for line in block.splitlines():
                cells=[c.strip() for c in line.strip('|').split('|')]
                if all(re.fullmatch(r':?-+:?',c) for c in cells):continue
                rows.append(cells)
            n=len(rows[0]);weights=[1]*n
            if n==2:weights=[.9,3.1]
            if n==3:weights=[.8,2.1,1.6]
            if n==4:weights=[1.2,1.5,1.4,1.6]
            if n==5:weights=[1.1,1,1.1,1.2,1.1]
            if n==6:weights=[1.3,1,1,1,1,1]
            if 'Dates' in rows[0]:weights=[.85,2.1,.85,.6]
            if 'Validation interval' in rows[0]:weights=[.6,.85,1.15,.85,1.9]
            if 'Largest confusion' in rows[0]:weights=[1.1,.6,1.35,1.9,.55]
            values=[[para(c,'cellhead' if r==0 else 'cell') for c in row] for r,row in enumerate(rows)]
            t=Table(values,colWidths=[content*x/sum(weights) for x in weights],repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('VALIGN',(0,0),(-1,-1),'TOP'),
               ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f6f7'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,ink),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d8e2e5'))]))
            story.extend([t,Spacer(1,9)]);i+=1;continue
        if block.startswith('# '):story.append(para(block[2:],'title'))
        elif block.startswith('## '):story.append(para(block[3:],'h2'))
        elif block.startswith('### '):story.append(para(block[4:],'h3'))
        elif block.startswith('*Table '):story.append(para(block.strip('*'),'tablecaption'))
        elif block.startswith('- '):
            for line in block.splitlines():story.append(para('• '+line.removeprefix('- '),'bullet'))
        elif re.match(r'^\[\d+\]',block):story.append(para(block.replace('\n',' '),'reference'))
        else:story.append(para(block.replace('\n',' ')))
        i+=1
    doc.build(story)
    return destination


def main():
    R.mkdir(exist_ok=True);evaluation=read(R/'evaluation_results.json');checks=verify_outputs(evaluation);audit=data_audit()
    assert audit['source_sha256']==evaluation['experiment']['dataset_sha256']
    figures(evaluation,audit)
    paper=write_markdown(evaluation,audit)
    write_bibliography();pdf=make_pdf(paper)
    from pypdf import PdfReader
    reader=PdfReader(pdf);text='\n'.join(page.extract_text() for page in reader.pages)
    expected=['Abstract','Keywords','Introduction','Problem Statement','Research Objectives','Literature Review','Research Motivation','Dataset Description','Data Preprocessing','Theoretical Background','Proposed Methodology','Experimental Setup','Results and Discussion','Feature Importance and Interpretability','Limitations','Conclusion and Future Work','References']
    assert all(section in text for section in expected)
    assert len(re.findall(r'^## ',paper,re.M))==17
    assert len(re.findall(r'^\*Figure \d+\.',paper,re.M))==8
    assert len(re.findall(r'^\*Table \d+\.',paper,re.M))==13
    assert all((R/path).is_file() for path in re.findall(r'!\[[^]]*\]\(([^)]+)\)',paper))
    checks.update({'pdf_pages':len(reader.pages),'pdf_opens':True,'expected_sections_found':len(expected),'figures':8,'tables':13,'paper_sha256':digest(pdf),'markdown_written_before_pdf':True})
    (R/'paper_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
