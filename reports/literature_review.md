# AirSense literature review

## 4. Literature Review

Méndez, Merayo and Núñez [1] review 155 studies covering 2011-2021. They organize the literature by geography, predicted quantities, predictors, evaluation metrics and model families. Section 3.3 distinguishes pollutant, meteorological and other predictors; Section 3.4 separates metric types and discusses accuracy and regression errors. Their Section 4 covers individual, hybrid and comparative approaches. This is a survey of heterogeneous forecasting studies, not a benchmark result on the AirSense dataset.

For AirSense, this taxonomy becomes a reporting checklist: identify the target and time horizon, list only available inputs, document the split, and choose metrics suited to the task. Pollutant measurements are available; weather measurements are not. Calendar fields can encode seasonal structure, but cannot establish meteorological effects. The present four-classifier comparison is an implementation choice for tabular data, not evidence that these methods exhaust the literature or universally outperform alternatives.

A useful comparison therefore concerns experimental questions rather than published score magnitudes. AirSense estimates a category from contemporaneous measurements; future forecasting would require a declared lead time and inputs available before the target day. Classification F1 cannot be interpreted as concentration-regression error. The split dates, city coverage and target construction must also agree before numerical comparisons become meaningful. We do not identify an unverified literature-wide gap or claim a new learning algorithm.

The foundational works on Random Forests [5], support-vector networks [6], gradient boosting systems [7] and additive feature attribution [8] explain the model/interpretation families used here. These references support theoretical mechanisms, not claims about their performance on this CSV. The contribution is a reproducible, bounded project experiment with explicit provenance and a read-only demonstration interface.

## 15. References

[1] M. Méndez, M. G. Merayo, and M. Núñez, "Machine learning algorithms to forecast air quality: a survey," Artificial Intelligence Review, vol. 56, pp. 10031-10066, 2023. DOI: 10.1007/s10462-023-10424-4. Publisher HTML and 36-page full paper verified 1 November 2026.

[2] Central Pollution Control Board, National Air Quality Index, report CUPS/82/2014-15, October 2014. Official government-hosted report: https://www.airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf. Category table verified on CPCB's official AQI portal: https://airquality.cpcb.gov.in/AQI_All_India/. Accessed 1 November 2026.

[3] R. Rao, Air Quality Data in India (2015-2020), Kaggle dataset. https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india. Accessed 1 November 2026. Statistical statements in this paper come from the supplied local city_day.csv, not unverified online dataset summaries.

[4] scikit-learn developers, "Decision Trees," "Metrics and scoring," and "Permutation feature importance," scikit-learn 1.8 documentation. https://scikit-learn.org/1.8/modules/tree.html ; https://scikit-learn.org/1.8/modules/model_evaluation.html ; https://scikit-learn.org/1.8/modules/permutation_importance.html. Accessed 1 November 2026.

[5] L. Breiman, "Random Forests," Machine Learning, vol. 45, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324. Publisher bibliographic record verified 1 November 2026.

[6] C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, pp. 273-297, 1995. DOI: 10.1007/BF00994018. Publisher bibliographic record verified 1 November 2026.

[7] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," arXiv:1603.02754, 2016, version 3. https://arxiv.org/abs/1603.02754. DOI: 10.48550/arXiv.1603.02754. Author/title/year/version verified 1 November 2026; cited as the verified author preprint rather than supplying unverified proceedings details.

[8] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," arXiv:1705.07874, 2017, version 2. https://arxiv.org/abs/1705.07874. DOI: 10.48550/arXiv.1705.07874. Author/title/year/version verified 1 November 2026.
