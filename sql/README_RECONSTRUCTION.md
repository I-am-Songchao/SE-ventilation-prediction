# Raw-database extraction note

The original upstream raw-database extraction scripts were not recovered.

The reconstructed Python model pipeline intentionally begins from two **analysis-ready patient-level tables** rather than pretending that an unverified raw-data query is the original extraction code.

The final manuscript specifies the following extraction rules that any new raw-data extraction implementation must reproduce.

## MIMIC-IV

- adult patients;
- first eligible ICU admission per patient;
- SE diagnosis titles containing:
  - `status epilepticus`
  - `grand mal status`
  - `petit mal status`
- standardized ICD-9 codes 3452/3453 were included;
- fallback title rule: contains `status` and either `seiz` or `epilep`;
- explicitly exclude `without status epilepticus`;
- first invasive mechanical ventilation episode begins between 6 h before and 6 h after ICU admission;
- exclude patients dead by 6 h or no longer invasively ventilated at 6 h;
- exclude cardiac arrest/anoxic/hypoxic-ischemic brain injury cases;
- predictors:
  - GCS motor: item 223901, most recent 0–6 h
  - MAP: 220052/220181, minimum 0–6 h
  - SpO2: 220277, minimum 0–6 h
  - FiO2: 223835, maximum 0–6 h
  - PEEP: 224700 preferred; 220339 fallback, maximum 0–6 h
  - BUN: 51006, closest value from -6 h to +6 h relative to ventilation start
- outcome:
  - continued invasive/tracheostomy ventilation at 72 h, or
  - death between 6 h and 72 h.

## eICU-CRD

- diagnosis path/text specifically indicates status epilepticus;
- main matched paths:
  - `neurologic|seizures|seizures|status epilepticus`
  - `surgery|neurosurgical issues|seizures|status epilepticus`
- code fields include 345.3 and G40.901;
- one SE-related ICU stay retained per patient;
- primary cohort requires first SE documentation within 24 h after ICU admission;
- exclude post-anoxic/cardiac-arrest-related cases;
- first IMV begins within the prespecified early window;
- invasive ventilation adjudication uses multiple sources:
  - respiratoryCare
  - respiratoryCharting
  - treatment
  - carePlanGeneral
- predictors:
  - age/sex: patient
  - GCS motor: nurseCharting
  - MAP: vitalPeriodic + vitalAperiodic
  - SpO2: vitalPeriodic
  - FiO2/PEEP: respiratoryCharting
  - BUN: lab
- strict primary external validation additionally requires observed GCS motor within the 6-h prediction window.

Because the exact original ventilation-adjudication and raw table joins were not recovered, a new raw-data SQL implementation must be locally verified against the manuscript cohort counts before public claims of exact reproducibility are made.
