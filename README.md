# SE-ventilation-prediction

Early prediction of persistent invasive mechanical ventilation or death in critically ill patients with status epilepticus using MIMIC-IV and eICU-CRD.

## Overview

This repository contains the analysis workflow for the study:

**Early prediction of persistent invasive mechanical ventilation or death by 72 h in critically ill patients with status epilepticus: model development, temporal validation, and multicenter external validation**

The study develops a parsimonious clinical prediction model for critically ill patients with status epilepticus (SE) who remain alive and invasively ventilated 6 hours after initiation of their first invasive mechanical ventilation episode.

MIMIC-IV was used for model development, repeated internal validation, and temporal validation. The eICU Collaborative Research Database (eICU-CRD) was used for independent multicenter external validation.

## Study design

The start of the first invasive mechanical ventilation episode is defined as time zero.

A fixed prediction landmark is set at **6 h after ventilation initiation**. Only patients who are alive and still receiving invasive mechanical ventilation at this landmark are included.

The primary outcome is defined as continued invasive mechanical ventilation or tracheostomy ventilation at 72 h after ventilation initiation, or death between the 6-h landmark and 72 h.

The outcome represents failure to enter an early ventilator-liberation trajectory and should not be interpreted as extubation failure or weaning failure.

## Predictors

The final model contains eight prespecified predictors:

1. Age
2. Sex
3. Glasgow Coma Scale motor score (GCS motor)
4. Minimum mean arterial pressure (MAP)
5. Minimum peripheral oxygen saturation (SpO2)
6. Maximum fraction of inspired oxygen (FiO2)
7. Maximum positive end-expiratory pressure (PEEP)
8. Most recent blood urea nitrogen (BUN)

Age is capped at 90 years. Female sex is coded as 1 and male sex as 0. FiO2 is represented as a proportion from 0 to 1.

### Prediction windows

- GCS motor: most recent value from ventilation initiation to the 6-h landmark
- MAP: minimum value from ventilation initiation to the 6-h landmark
- SpO2: minimum value from ventilation initiation to the 6-h landmark
- FiO2: maximum value from ventilation initiation to the 6-h landmark
- PEEP: maximum value from ventilation initiation to the 6-h landmark
- BUN: value closest to the landmark from 6 h before ventilation initiation through the 6-h landmark

## MIMIC-IV variable mapping

| Variable | Item ID / source |
|---|---|
| GCS motor | `223901` – GCS Motor Response |
| MAP | `220052` – Arterial Blood Pressure mean; `220181` – Non Invasive Blood Pressure mean |
| SpO2 | `220277` – O2 saturation pulseoxymetry |
| FiO2 | `223835` – Inspired O2 Fraction |
| PEEP | `224700` – Total PEEP Level; `220339` – PEEP set as fallback |
| BUN | `51006` – Urea Nitrogen |

## eICU-CRD variable mapping

| Variable | Main source |
|---|---|
| Age / sex | `patient` |
| GCS motor | `nurseCharting` |
| MAP | `vitalPeriodic` and `vitalAperiodic` |
| SpO2 | `vitalPeriodic` |
| FiO2 | `respiratoryCharting` |
| PEEP | `respiratoryCharting` |
| BUN | `lab` |

Invasive mechanical ventilation in eICU-CRD is adjudicated using multiple sources, including `respiratoryCare`, `respiratoryCharting`, `treatment`, and `carePlanGeneral`.

## Model development

The primary model is a ridge-penalized logistic regression model.

Preprocessing includes median imputation and standardization. All preprocessing parameters are estimated within the appropriate training data to avoid information leakage.

The regularization parameter `C` is selected using log loss from:

`0.003, 0.01, 0.03, 0.1, 0.3, 1, 3`

Repeated internal validation uses stratified five-fold outer cross-validation repeated five times, with inner cross-validation for hyperparameter selection.

A 7-variable model excluding GCS motor is used as a prespecified ablation comparison.

## Validation

The analysis includes:

- repeated nested internal cross-validation in MIMIC-IV;
- temporal validation using 2008–2016 for training and 2017–2022 for validation;
- independent multicenter external validation in eICU-CRD;
- patient-level and hospital-cluster bootstrap confidence intervals;
- calibration assessment;
- descriptive recalibration;
- prespecified sensitivity and data-completeness analyses.

## Repository structure

```text
SE-ventilation-prediction/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── CITATION.cff
├── src/
│   └── README.md
├── sql/
│   └── README.md
├── data/
│   └── README.md
└── docs/
    ├── MANUSCRIPT_MAPPING.md
    └── CODE_AVAILABILITY_STATEMENT.txt
```

Recommended analysis-script names:

```text
src/
├── 01_data_preprocessing.py
├── 02_model_development.py
├── 03_internal_validation.py
├── 04_temporal_validation.py
├── 05_external_validation.py
├── 06_sensitivity_analysis.py
└── 07_figures_tables.py
```

## Data availability and access

This repository **does not contain patient-level MIMIC-IV or eICU-CRD data**.

MIMIC-IV and eICU-CRD are available through PhysioNet to credentialed investigators who complete the required training and agree to the applicable data-use agreements.

Users who wish to reproduce the study must obtain independent authorization to access the source databases.

Do not upload protected or credentialed source data, patient-level exports, database credentials, API keys, or local connection strings to this repository.

## Reproducibility

The repository documents cohort construction, preprocessing, model development, validation, sensitivity analyses, and generation of the figures and tables reported in the manuscript.

Users may need to adjust local database connection settings and file paths before running the scripts.

## Software

Python is used for data processing, statistical modeling, validation, and visualization.

A minimal package list is provided in `requirements.txt`. Exact package versions used for the final manuscript should be added once confirmed from the analysis environment.

## Citation

If you use this repository, please cite the associated manuscript once published.

Repository citation metadata are provided in `CITATION.cff`.

## Authors

- Chao Song
- Yanqi Wang
- Meihan Liu
- Qingyu Zhang
- Qinghua Liu

Chao Song and Yanqi Wang contributed equally to the study and share first authorship.

## Corresponding author

Qinghua Liu

Department of Clinical Laboratory, The Second Affiliated Hospital of Shandong First Medical University, Taian 271000, China.

## Funding

This work was supported by the General Program of Shandong Provincial Natural Science Foundation (Grant No. ZR2024MH164).

## License

The source code in this repository is released under the MIT License. Database access and use remain subject to the independent terms and data-use agreements of MIMIC-IV, eICU-CRD, and PhysioNet.
