# Manuscript-to-code mapping

| Manuscript component | Suggested code location |
|---|---|
| Cohort construction | `sql/mimic_cohort.sql`, `sql/eicu_cohort.sql` |
| Predictor extraction | `sql/mimic_predictors.sql`, `sql/eicu_predictors.sql` |
| Data preprocessing | `src/01_data_preprocessing.py` |
| Final 8-variable ridge model | `src/02_model_development.py` |
| Repeated 5×5 nested cross-validation | `src/03_internal_validation.py` |
| 2008–2016 / 2017–2022 temporal validation | `src/04_temporal_validation.py` |
| eICU multicenter external validation | `src/05_external_validation.py` |
| Sensitivity analyses / GCS completeness analyses | `src/06_sensitivity_analysis.py` |
| Figures and tables | `src/07_figures_tables.py` |

Prediction landmark: 6 h after initiation of first invasive mechanical ventilation.

Primary outcome: continued invasive mechanical ventilation or tracheostomy ventilation at 72 h after ventilation initiation, or death between 6 h and 72 h.

Final predictors: age, sex, GCS motor, minimum MAP, minimum SpO2, maximum FiO2, maximum PEEP, and most recent BUN.

Update this mapping if your actual finalized scripts use different file names.
