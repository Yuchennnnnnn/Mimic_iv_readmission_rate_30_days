-- ============================================================================
-- MIMIC-IV Cohort Definition with 30-Day Readmission Label
-- ============================================================================
-- Purpose: Create cohort of hospital admissions with 30-day readmission labels
-- Exclusions: Age < 18, died in hospital, missing discharge time
-- Output: ~195K admissions with readmit_30d label (0 or 1)
-- ============================================================================

WITH 
-- Step 1: Get patient demographics and age
patients_with_age AS (
  SELECT 
    p.subject_id,
    p.gender,
    p.anchor_age,
    p.anchor_year,
    p.anchor_year_group,
    p.dod  -- date of death
  FROM `physionet-data.mimiciv_hosp.patients` AS p
),

-- Step 2: Get all admissions with key information
admissions_clean AS (
  SELECT 
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.insurance,
    a.language,
    a.marital_status,
    a.race,
    a.hospital_expire_flag,
    p.anchor_age,
    p.anchor_year,
    p.anchor_year_group,
    p.gender,
    -- Calculate age at admission
    p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age_at_admission
  FROM `physionet-data.mimiciv_hosp.admissions` AS a
  INNER JOIN patients_with_age AS p
    ON a.subject_id = p.subject_id
  WHERE 
    a.dischtime IS NOT NULL  -- Must have discharge time
    AND a.admittime IS NOT NULL
    AND a.admittime < a.dischtime  -- Valid admission
),

-- Step 3: Compute next admission time for each patient
next_admissions AS (
  SELECT 
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    -- Get the next admission time for this patient
    LEAD(admittime) OVER (
      PARTITION BY subject_id 
      ORDER BY admittime
    ) AS next_admittime,
    -- Get the next admission ID
    LEAD(hadm_id) OVER (
      PARTITION BY subject_id 
      ORDER BY admittime
    ) AS next_hadm_id
  FROM admissions_clean
),

-- Step 4: Calculate readmission label and time to readmission
readmit_labels AS (
  SELECT 
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    next_admittime,
    next_hadm_id,
    -- Days between discharge and next admission
    DATE_DIFF(
      CAST(next_admittime AS DATE), 
      CAST(dischtime AS DATE), 
      DAY
    ) AS days_to_readmit,
    -- 30-day readmission label
    CASE 
      WHEN DATE_DIFF(
        CAST(next_admittime AS DATE), 
        CAST(dischtime AS DATE), 
        DAY
      ) BETWEEN 1 AND 30 THEN 1
      ELSE 0
    END AS readmit_30d,
    -- 60-day readmission label (optional)
    CASE 
      WHEN DATE_DIFF(
        CAST(next_admittime AS DATE), 
        CAST(dischtime AS DATE), 
        DAY
      ) BETWEEN 1 AND 60 THEN 1
      ELSE 0
    END AS readmit_60d
  FROM next_admissions
),

-- Step 5: Combine all information
final_cohort AS (
  SELECT 
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.insurance,
    a.language,
    a.marital_status,
    a.race,
    a.hospital_expire_flag,
    a.gender,
    a.age_at_admission,
    a.anchor_year_group,
    r.next_hadm_id,
    r.next_admittime,
    r.days_to_readmit,
    r.readmit_30d,
    r.readmit_60d,
    -- Length of stay in days
    DATE_DIFF(
      CAST(a.dischtime AS DATE),
      CAST(a.admittime AS DATE),
      DAY
    ) AS los_days,
    -- Length of stay in hours (for binning)
    TIMESTAMP_DIFF(a.dischtime, a.admittime, HOUR) AS los_hours
  FROM admissions_clean AS a
  INNER JOIN readmit_labels AS r
    ON a.hadm_id = r.hadm_id
  WHERE 
    -- Exclusion criteria
    a.age_at_admission >= 18  -- Adults only
    AND a.hospital_expire_flag = 0  -- Exclude in-hospital deaths
    AND a.dischtime IS NOT NULL  -- Must have discharge time
    AND TIMESTAMP_DIFF(a.dischtime, a.admittime, HOUR) >= 48  -- At least 48 hours for time series
)

-- Final output: Save to table or export
SELECT 
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  deathtime,
  admission_type,
  admission_location,
  discharge_location,
  insurance,
  language,
  marital_status,
  race,
  hospital_expire_flag,
  gender,
  age_at_admission,
  anchor_year_group,
  next_hadm_id,
  next_admittime,
  days_to_readmit,
  readmit_30d,
  readmit_60d,
  los_days,
  los_hours
FROM final_cohort
ORDER BY subject_id, admittime;

-- ============================================================================
-- Expected Output Statistics:
-- - Total admissions: ~195,000
-- - Readmission rate (30d): ~26-28%
-- - Age distribution: 18-90+ years
-- - Anchor year groups: 2008-2010, 2011-2013, 2014-2016, 2017-2019
-- ============================================================================

-- To save as a table:
-- CREATE OR REPLACE TABLE `your-project.your_dataset.cohort_30d_readmit` AS
-- (... query above ...)

-- To export to GCS:
-- EXPORT DATA OPTIONS(
--   uri='gs://your-bucket/cohort_30d_readmit_*.parquet',
--   format='PARQUET',
--   overwrite=true
-- ) AS
-- (... query above ...)
