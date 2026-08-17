# `sql/`

Place SQL or database-query scripts used for MIMIC-IV and eICU-CRD cohort construction and variable extraction in this directory.

Suggested organization:

```text
mimic_cohort.sql
mimic_predictors.sql
eicu_cohort.sql
eicu_predictors.sql
```

Do **not** upload database dumps, patient-level CSV files, credentials, connection strings, or restricted source data.

## Key MIMIC-IV item identifiers

| Variable | Item ID |
|---|---|
| GCS motor | 223901 |
| MAP | 220052; 220181 |
| SpO2 | 220277 |
| FiO2 | 223835 |
| PEEP | 224700; 220339 |
| BUN | 51006 |

## Main eICU-CRD sources

- `patient`
- `nurseCharting`
- `vitalPeriodic`
- `vitalAperiodic`
- `respiratoryCare`
- `respiratoryCharting`
- `treatment`
- `carePlanGeneral`
- `lab`
