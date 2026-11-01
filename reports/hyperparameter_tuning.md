# AirSense — Hyperparameter tuning

The uploaded project and the earlier audit had already inspected test outcomes. This corrected rerun is a retrospective benchmark, not a pristine prospective holdout. No test scores are used by the corrected selection/tuning code. New external data is required for an independent final claim.

## Protocol

Training: dates through 2019-06-30. Validation: 2019-07-01 through 2019-12-31. Test is evaluated only by the separate final evaluation command. Search uses training rows only, three expanding date windows with a seven-day gap, and fold-local preprocessing. All cities on one date stay on the same side of a fold. This targets future dates in observed cities, not new-city transfer. Six seeded random candidates per model limit computation. XGBoost uses CV with fixed tree-count candidates; no validation early stopping. SVM search is limited to 6,000 stratified training rows; its finalist is refitted on every training row.

## Baseline versus tuned — validation only

| model | baseline_macro_f1 | tuned_macro_f1 | delta_f1 | baseline_accuracy | tuned_accuracy |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | 0.706760 | 0.716606 | 0.009846 | 0.719609 | 0.725687 |
| Random Forest | 0.774868 | 0.773660 | -0.001208 | 0.798362 | 0.789112 |
| SVM | 0.666401 | 0.660938 | -0.005463 | 0.669662 | 0.662526 |
| XGBoost | 0.766091 | 0.761736 | -0.004355 | 0.807347 | 0.803118 |

## tuned_decision_tree

```json
{
  "model": "Decision Tree",
  "variant": "tuned",
  "best_params": {
    "classifier__min_samples_leaf": 5,
    "classifier__max_depth": 10,
    "classifier__ccp_alpha": 0.0005
  },
  "search_space": {
    "classifier__max_depth": [
      5,
      8,
      10,
      15
    ],
    "classifier__min_samples_leaf": [
      5,
      10,
      20
    ],
    "classifier__ccp_alpha": [
      0.0,
      0.0005
    ]
  },
  "cv_best_f1": 0.5991855187572432,
  "cv_method": "3 expanding date windows, 7-day gap, preprocessing within folds",
  "search_rows": 16645,
  "n_train": 16645,
  "used_n_train": 16645,
  "search_time_sec": 3.237085855999794,
  "train_time_sec": 0.45630703200004064,
  "val_macro_f1": 0.716606,
  "val_accuracy": 0.725687,
  "validation_metrics": {
    "split": "Validation",
    "n_samples": 3784,
    "accuracy": 0.725687,
    "macro_precision": 0.70216,
    "macro_recall": 0.75691,
    "macro_f1": 0.716606,
    "weighted_f1": 0.73292,
    "per_class_report": {
      "Good": {
        "precision": 0.4484304932735426,
        "recall": 0.8645533141210374,
        "f1-score": 0.5905511811023622,
        "support": 347.0
      },
      "Satisfactory": {
        "precision": 0.8219645293315143,
        "recall": 0.7228554289142172,
        "f1-score": 0.7692307692307693,
        "support": 1667.0
      },
      "Moderate": {
        "precision": 0.8058690744920993,
        "recall": 0.6761363636363636,
        "f1-score": 0.7353244078269825,
        "support": 1056.0
      },
      "Poor": {
        "precision": 0.6160714285714286,
        "recall": 0.6509433962264151,
        "f1-score": 0.6330275229357798,
        "support": 318.0
      },
      "Very Poor": {
        "precision": 0.6947791164658634,
        "recall": 0.7723214285714286,
        "f1-score": 0.7315010570824524,
        "support": 224.0
      },
      "Severe": {
        "precision": 0.8258426966292135,
        "recall": 0.8546511627906976,
        "f1-score": 0.84,
        "support": 172.0
      },
      "accuracy": 0.7256871035940803,
      "macro avg": {
        "precision": 0.7021595564606103,
        "recall": 0.7569101823766933,
        "f1-score": 0.7166058230297243,
        "support": 3784.0
      },
      "weighted avg": {
        "precision": 0.7585634710410948,
        "recall": 0.7256871035940803,
        "f1-score": 0.7329203266108756,
        "support": 3784.0
      }
    },
    "confusion_matrix": [
      [
        300,
        45,
        1,
        1,
        0,
        0
      ],
      [
        343,
        1205,
        117,
        1,
        1,
        0
      ],
      [
        25,
        213,
        714,
        102,
        1,
        1
      ],
      [
        1,
        3,
        45,
        207,
        57,
        5
      ],
      [
        0,
        0,
        3,
        23,
        173,
        25
      ],
      [
        0,
        0,
        6,
        2,
        17,
        147
      ]
    ],
    "class_labels": [
      "Good",
      "Satisfactory",
      "Moderate",
      "Poor",
      "Very Poor",
      "Severe"
    ],
    "inference_time_sec": 0.006874,
    "inference_runs_sec": [
      0.009435503999611683,
      0.006873834999623796,
      0.006355637000069692
    ],
    "inference_ms_per_row": 0.0018165961945031711
  },
  "model_sha256": "9b98b823fc135456f67b0cc620006d8b6a33981e76c45a924a0bcfd218265f3c"
}
```

## tuned_random_forest

```json
{
  "model": "Random Forest",
  "variant": "tuned",
  "best_params": {
    "classifier__n_estimators": 200,
    "classifier__min_samples_leaf": 10,
    "classifier__max_features": 0.5,
    "classifier__max_depth": null
  },
  "search_space": {
    "classifier__n_estimators": [
      100,
      200
    ],
    "classifier__max_depth": [
      10,
      15,
      null
    ],
    "classifier__min_samples_leaf": [
      2,
      5,
      10
    ],
    "classifier__max_features": [
      "sqrt",
      0.5
    ]
  },
  "cv_best_f1": 0.6855692282957421,
  "cv_method": "3 expanding date windows, 7-day gap, preprocessing within folds",
  "search_rows": 16645,
  "n_train": 16645,
  "used_n_train": 16645,
  "search_time_sec": 54.444899208999686,
  "train_time_sec": 13.845521810000264,
  "val_macro_f1": 0.77366,
  "val_accuracy": 0.789112,
  "validation_metrics": {
    "split": "Validation",
    "n_samples": 3784,
    "accuracy": 0.789112,
    "macro_precision": 0.763016,
    "macro_recall": 0.793902,
    "macro_f1": 0.77366,
    "weighted_f1": 0.791331,
    "per_class_report": {
      "Good": {
        "precision": 0.594704684317719,
        "recall": 0.8414985590778098,
        "f1-score": 0.6968973747016707,
        "support": 347.0
      },
      "Satisfactory": {
        "precision": 0.8409785932721713,
        "recall": 0.8248350329934013,
        "f1-score": 0.8328285887341006,
        "support": 1667.0
      },
      "Moderate": {
        "precision": 0.8430939226519337,
        "recall": 0.7225378787878788,
        "f1-score": 0.7781744008159103,
        "support": 1056.0
      },
      "Poor": {
        "precision": 0.6565096952908587,
        "recall": 0.7452830188679245,
        "f1-score": 0.6980854197349042,
        "support": 318.0
      },
      "Very Poor": {
        "precision": 0.759090909090909,
        "recall": 0.7455357142857143,
        "f1-score": 0.7522522522522522,
        "support": 224.0
      },
      "Severe": {
        "precision": 0.8837209302325582,
        "recall": 0.8837209302325582,
        "f1-score": 0.8837209302325582,
        "support": 172.0
      },
      "accuracy": 0.7891120507399577,
      "macro avg": {
        "precision": 0.7630164558093582,
        "recall": 0.7939018557075478,
        "f1-score": 0.7736598277452327,
        "support": 3784.0
      },
      "weighted avg": {
        "precision": 0.8005780839065161,
        "recall": 0.7891120507399577,
        "f1-score": 0.7913309941022808,
        "support": 3784.0
      }
    },
    "confusion_matrix": [
      [
        292,
        53,
        1,
        1,
        0,
        0
      ],
      [
        194,
        1375,
        96,
        1,
        1,
        0
      ],
      [
        4,
        202,
        763,
        86,
        1,
        0
      ],
      [
        1,
        4,
        39,
        237,
        36,
        1
      ],
      [
        0,
        0,
        3,
        35,
        167,
        19
      ],
      [
        0,
        1,
        3,
        1,
        15,
        152
      ]
    ],
    "class_labels": [
      "Good",
      "Satisfactory",
      "Moderate",
      "Poor",
      "Very Poor",
      "Severe"
    ],
    "inference_time_sec": 0.116474,
    "inference_runs_sec": [
      0.12297542299984343,
      0.09463234799977727,
      0.1164743230001477
    ],
    "inference_ms_per_row": 0.030780655391120505
  },
  "model_sha256": "b232c7fad9eb8861de62b03fa5a5a8a7317078366f2f607582109342e1ac8471"
}
```

## tuned_svm

```json
{
  "model": "SVM",
  "variant": "tuned",
  "best_params": {
    "classifier__gamma": "auto",
    "classifier__C": 5.0
  },
  "search_space": {
    "classifier__C": [
      0.1,
      1.0,
      5.0,
      10.0,
      30.0
    ],
    "classifier__gamma": [
      "scale",
      "auto"
    ]
  },
  "cv_best_f1": 0.5907816439428499,
  "cv_method": "3 expanding date windows, 7-day gap, preprocessing within folds",
  "search_rows": 6000,
  "n_train": 16645,
  "used_n_train": 16645,
  "search_time_sec": 6.236036968999997,
  "train_time_sec": 7.094371433000106,
  "val_macro_f1": 0.660938,
  "val_accuracy": 0.662526,
  "validation_metrics": {
    "split": "Validation",
    "n_samples": 3784,
    "accuracy": 0.662526,
    "macro_precision": 0.654705,
    "macro_recall": 0.697982,
    "macro_f1": 0.660938,
    "weighted_f1": 0.673996,
    "per_class_report": {
      "Good": {
        "precision": 0.3507853403141361,
        "recall": 0.7723342939481268,
        "f1-score": 0.48244824482448245,
        "support": 347.0
      },
      "Satisfactory": {
        "precision": 0.78003003003003,
        "recall": 0.6232753449310138,
        "f1-score": 0.6928976325441814,
        "support": 1667.0
      },
      "Moderate": {
        "precision": 0.7445255474452555,
        "recall": 0.6761363636363636,
        "f1-score": 0.7086848635235732,
        "support": 1056.0
      },
      "Poor": {
        "precision": 0.5957446808510638,
        "recall": 0.6163522012578616,
        "f1-score": 0.6058732612055642,
        "support": 318.0
      },
      "Very Poor": {
        "precision": 0.6571428571428571,
        "recall": 0.6160714285714286,
        "f1-score": 0.6359447004608295,
        "support": 224.0
      },
      "Severe": {
        "precision": 0.8,
        "recall": 0.8837209302325582,
        "f1-score": 0.8397790055248618,
        "support": 172.0
      },
      "accuracy": 0.6625264270613108,
      "macro avg": {
        "precision": 0.6547047426305571,
        "recall": 0.6979817604295587,
        "f1-score": 0.6609379513472488,
        "support": 3784.0
      },
      "weighted avg": {
        "precision": 0.7089054861950034,
        "recall": 0.6625264270613108,
        "f1-score": 0.6739964083517469,
        "support": 3784.0
      }
    },
    "confusion_matrix": [
      [
        268,
        76,
        2,
        1,
        0,
        0
      ],
      [
        455,
        1039,
        171,
        2,
        0,
        0
      ],
      [
        41,
        214,
        714,
        83,
        3,
        1
      ],
      [
        0,
        3,
        64,
        196,
        51,
        4
      ],
      [
        0,
        0,
        7,
        46,
        138,
        33
      ],
      [
        0,
        0,
        1,
        1,
        18,
        152
      ]
    ],
    "class_labels": [
      "Good",
      "Satisfactory",
      "Moderate",
      "Poor",
      "Very Poor",
      "Severe"
    ],
    "inference_time_sec": 2.589673,
    "inference_runs_sec": [
      2.6154425059999085,
      2.5875304519995552,
      2.589673049999874
    ],
    "inference_ms_per_row": 0.6843744714587737
  },
  "model_sha256": "dc1540062acd459e5e0118123375c3f46f43189736c4c7bfef6f6c0b8182fed0"
}
```

## tuned_xgboost

```json
{
  "model": "XGBoost",
  "variant": "tuned",
  "best_params": {
    "classifier__reg_lambda": 1.0,
    "classifier__n_estimators": 150,
    "classifier__max_depth": 6,
    "classifier__learning_rate": 0.05
  },
  "search_space": {
    "classifier__n_estimators": [
      150,
      300
    ],
    "classifier__max_depth": [
      4,
      6,
      8
    ],
    "classifier__learning_rate": [
      0.03,
      0.05,
      0.1
    ],
    "classifier__reg_lambda": [
      1.0,
      2.0
    ]
  },
  "cv_best_f1": 0.6537593796503263,
  "cv_method": "3 expanding date windows, 7-day gap, preprocessing within folds",
  "search_rows": 16645,
  "n_train": 16645,
  "used_n_train": 16645,
  "search_time_sec": 43.47302762299978,
  "train_time_sec": 4.04461746700008,
  "val_macro_f1": 0.761736,
  "val_accuracy": 0.803118,
  "validation_metrics": {
    "split": "Validation",
    "n_samples": 3784,
    "accuracy": 0.803118,
    "macro_precision": 0.796478,
    "macro_recall": 0.743076,
    "macro_f1": 0.761736,
    "weighted_f1": 0.797371,
    "per_class_report": {
      "Good": {
        "precision": 0.7931034482758621,
        "recall": 0.46397694524495675,
        "f1-score": 0.5854545454545454,
        "support": 347.0
      },
      "Satisfactory": {
        "precision": 0.8065031982942431,
        "recall": 0.907618476304739,
        "f1-score": 0.8540784645780413,
        "support": 1667.0
      },
      "Moderate": {
        "precision": 0.8211462450592886,
        "recall": 0.7869318181818182,
        "f1-score": 0.8036750483558994,
        "support": 1056.0
      },
      "Poor": {
        "precision": 0.7278911564625851,
        "recall": 0.6729559748427673,
        "f1-score": 0.6993464052287581,
        "support": 318.0
      },
      "Very Poor": {
        "precision": 0.7393162393162394,
        "recall": 0.7723214285714286,
        "f1-score": 0.7554585152838428,
        "support": 224.0
      },
      "Severe": {
        "precision": 0.8909090909090909,
        "recall": 0.8546511627906976,
        "f1-score": 0.8724035608308606,
        "support": 172.0
      },
      "accuracy": 0.8031183932346723,
      "macro avg": {
        "precision": 0.7964782297195515,
        "recall": 0.743075967656068,
        "f1-score": 0.7617360899553246,
        "support": 3784.0
      },
      "weighted avg": {
        "precision": 0.8026138350658402,
        "recall": 0.8031183932346723,
        "f1-score": 0.797370680638844,
        "support": 3784.0
      }
    },
    "confusion_matrix": [
      [
        161,
        184,
        1,
        1,
        0,
        0
      ],
      [
        40,
        1513,
        113,
        0,
        1,
        0
      ],
      [
        2,
        175,
        831,
        46,
        2,
        0
      ],
      [
        0,
        4,
        65,
        214,
        34,
        1
      ],
      [
        0,
        0,
        2,
        32,
        173,
        17
      ],
      [
        0,
        0,
        0,
        1,
        24,
        147
      ]
    ],
    "class_labels": [
      "Good",
      "Satisfactory",
      "Moderate",
      "Poor",
      "Very Poor",
      "Severe"
    ],
    "inference_time_sec": 0.080901,
    "inference_runs_sec": [
      0.06782705900013752,
      0.10179866799990123,
      0.08090103499989709
    ],
    "inference_ms_per_row": 0.02137975687103594
  },
  "model_sha256": "9612276d16923543e1f9ab780634c289a6f4f1959eb6cdbfc365919550a63d7b"
}
```

## Selection and limits

Frozen candidate: **random_forest**, chosen by validation macro F1 across all successful baseline and tuned candidates. Negative deltas are retained; tuning is not guaranteed to help. One seasonally limited validation period and a small search budget make selection uncertain. CV scores on the SVM subset are not directly comparable to full-data CV scores. Within-city autocorrelation may persist beyond seven days. No city-disjoint generalization claim is made. A single seed is used; uncertainty across splits/seeds is not measured.

## Research context

Méndez, Merayo & Núñez (2023), Machine learning algorithms to forecast air quality: a survey. https://doi.org/10.1007/s10462-023-10424-4 (publisher verified 2026-11-01). Section 3.3 discusses predictor groups, including pollutant and weather variables. Section 3.4 distinguishes metric types and reports widespread use of RMSE/MSE and MAE as well as ACC, R² and MAPE. AirSense instead evaluates six-class, same-day AQI classification using macro F1 and per-class errors. Chronological splitting and macro-F1 selection here are our experimental choices, not attributed recommendations from the survey. Datasets, targets, horizons, splits and metrics differ, so our scores are not directly comparable with published scores. No universal ensemble advantage or macro-F1 recommendation is attributed to this survey.

## Reproduce

`python -m scripts.train_all_models` then `python -m scripts.tune_models`, then `python -m scripts.evaluate_all_models`. Full CV results are in `reports/tuning/`; search spaces, best parameters, actual row counts, failures and timing are stored above and in manifests.