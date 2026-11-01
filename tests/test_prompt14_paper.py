"""Prompt 14: document completeness and saved-output consistency."""
from pathlib import Path
import json
import re
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'


def test_requested_sources_sections_captions_and_figures_exist():
    for name in ['research_paper.md','literature_review.md','methodology.md','results_and_discussion.md','references.bib']:
        assert (REPORTS/name).is_file()
    paper=(REPORTS/'research_paper.md').read_text()
    assert len(re.findall(r'^# ',paper,re.M))==1
    assert len(re.findall(r'^## ',paper,re.M))==17
    assert re.findall(r'^\*Table (\d+)\.',paper,re.M)==[str(n) for n in range(1,14)]
    assert re.findall(r'^\*Figure (\d+)\.',paper,re.M)==[str(n) for n in range(1,9)]
    paths=re.findall(r'!\[[^]]*\]\(([^)]+)\)',paper)
    assert len(paths)==8 and all((REPORTS/path).is_file() for path in paths)
    assert 'not a claim of publication readiness' in paper
    assert 'retrospective' in paper.lower() and 'not a forecast' in paper.lower()


def test_paper_table_contains_actual_metrics_and_frozen_selection():
    evidence=json.loads((REPORTS/'evaluation_results.json').read_text())
    paper=(REPORTS/'research_paper.md').read_text()
    names={'decision_tree':'Decision Tree','random_forest':'Random Forest','svm':'SVM','xgboost':'XGBoost'}
    for key,name in names.items():
        m=evidence['candidates'][key]['test_metrics']
        expected='| '+name+' | '+' | '.join(f'{m[x]:.4f}' for x in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1'])+' |'
        assert expected in paper
    winner=min(evidence['candidates'],key=lambda k:(-evidence['candidates'][k]['validation_metrics']['macro_f1'],k))
    assert winner==evidence['selected_candidate']=='random_forest'


def test_data_audit_is_computed_from_current_csv():
    from src.data_loader import get_data_dir
    frame=pd.read_csv(get_data_dir()/'city_day.csv',parse_dates=['Date'])
    labeled=frame[frame.AQI_Bucket.notna()]
    audit=json.loads((REPORTS/'paper_data_audit.json').read_text())
    assert audit['raw_rows']==len(frame)
    assert audit['labeled']==len(labeled)
    assert sum(v['n'] for v in audit['splits'].values())==len(labeled)
    train=labeled[labeled.Date<='2019-06-30'];test=labeled[labeled.Date>'2019-12-31']
    unseen=sorted(set(test.City)-set(train.City))
    assert audit['test_cities_absent_in_training']==unseen
    assert audit['test_new_city_rows']==int(test.City.isin(unseen).sum())


def test_pdf_opens_and_contains_all_expected_sections():
    pypdf=pytest.importorskip('pypdf')
    reader=pypdf.PdfReader(REPORTS/'research_paper.pdf')
    assert not reader.is_encrypted and len(reader.pages)>1
    assert len(reader.outline)==18
    text='\n'.join(page.extract_text() for page in reader.pages)
    for section in ['Abstract','Keywords','Introduction','Problem Statement','Research Objectives','Literature Review',
                    'Research Motivation','Dataset Description','Data Preprocessing','Theoretical Background','Proposed Methodology',
                    'Experimental Setup','Results and Discussion','Feature Importance and Interpretability','Limitations','Conclusion and Future Work','References']:
        assert section in text
    assert '\ufffd' not in text
    verify=json.loads((REPORTS/'paper_verification.json').read_text())
    assert verify['actual_metrics_recomputed'] and verify['selection_verified_from_validation']


def test_bibliography_has_verified_entries_and_no_fake_release_year():
    bib=(REPORTS/'references.bib').read_text()
    assert len(re.findall(r'^@',bib,re.M))==8
    for doi in ['10.1007/s10462-023-10424-4','10.1023/A:1010933404324','10.1007/BF00994018',
                '10.48550/arXiv.1603.02754','10.48550/arXiv.1705.07874']:
        assert doi in bib
    assert '10031--10066' in bib
    assert 'release year not asserted' in bib
    assert (REPORTS/'reference_verification.md').is_file()
