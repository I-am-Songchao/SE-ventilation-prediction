# `src/`

Place the finalized Python scripts used to generate the manuscript results in this directory.

Recommended organization:

```text
01_data_preprocessing.py
02_model_development.py
03_internal_validation.py
04_temporal_validation.py
05_external_validation.py
06_sensitivity_analysis.py
07_figures_tables.py
```

Upload the **actual finalized analysis scripts** used for the manuscript rather than newly rewritten approximations.

Before uploading, remove or replace local absolute paths, usernames and passwords, database connection strings, API keys or tokens, patient-level exports, temporary files, and cached objects.

Prefer relative paths and configuration variables where possible.
