-- ============================================================================
-- Extract Chart Events for Cohort (First 48 Hours)
-- ============================================================================
-- Purpose: Extract vital signs and chart events for cohort admissions
-- Time window: First 48 hours after admission
-- Items: Vitals (HR, BP, SpO2, Temp), I/O, GCS, Respiratory
-- ============================================================================

WITH 
-- Get cohort hadm_ids (replace with your cohort table)
cohort AS (
  SELECT 
    hadm_id,
    admittime,
    TIMESTAMP_ADD(admittime, INTERVAL 48 HOUR) AS end_time
  FROM `your-project.your_dataset.cohort_30d_readmit`
),

-- Define itemids of interest (vitals and important chart events)
items_of_interest AS (
  SELECT itemid, label, category, unitname
  FROM `physionet-data.mimiciv_icu.d_items`
  WHERE itemid IN (
    -- Heart Rate
    220045,  -- Heart Rate
    
    -- Blood Pressure
    220050,  -- Arterial Blood Pressure systolic
    220051,  -- Arterial Blood Pressure diastolic
    220052,  -- Arterial Blood Pressure mean
    220179,  -- Non Invasive Blood Pressure systolic
    220180,  -- Non Invasive Blood Pressure diastolic
    220181,  -- Non Invasive Blood Pressure mean
    
    -- Respiratory
    220210,  -- Respiratory Rate
    223835,  -- Inspired O2 Fraction (FiO2)
    220277,  -- SpO2
    
    -- Temperature
    223761,  -- Temperature Fahrenheit
    223762,  -- Temperature Celsius
    
    -- Glasgow Coma Scale
    220739,  -- GCS - Eye Opening
    223900,  -- GCS - Verbal Response
    223901,  -- GCS - Motor Response
    198,     -- GCS Total (older)
    
    -- Glucose
    220621,  -- Glucose (serum)
    225664,  -- Glucose finger stick
    
    -- Fluid Input/Output
    220045,  -- Heart Rate (duplicate removed below)
    226559,  -- Foley
    226560,  -- Void
    226561,  -- Condom Cath
    226584,  -- Chest Tube #1
    226585,  -- Chest Tube #2
    
    -- Other vitals
    220615,  -- Respiratory Rate (Total)
    224690,  -- Respiratory Rate (spontaneous)
    
    -- Urine Output
    226627,  -- OR Urine
    226631   -- Urine
  )
)

-- Extract chart events
SELECT 
  ce.subject_id,
  ce.hadm_id,
  ce.stay_id,
  ce.charttime,
  ce.itemid,
  di.label AS item_label,
  di.category,
  ce.value,
  ce.valuenum,
  ce.valueuom AS unit,
  -- Time relative to admission (in hours)
  TIMESTAMP_DIFF(ce.charttime, c.admittime, HOUR) AS hours_since_admit,
  -- Time relative to admission (in minutes, for finer resolution)
  TIMESTAMP_DIFF(ce.charttime, c.admittime, MINUTE) AS minutes_since_admit
FROM `physionet-data.mimiciv_icu.chartevents` AS ce
INNER JOIN cohort AS c
  ON ce.hadm_id = c.hadm_id
INNER JOIN items_of_interest AS di
  ON ce.itemid = di.itemid
WHERE 
  ce.charttime >= c.admittime
  AND ce.charttime <= c.end_time  -- First 48 hours only
  AND ce.valuenum IS NOT NULL  -- Must have numeric value
  AND ce.valuenum > 0  -- Positive values only
ORDER BY ce.subject_id, ce.hadm_id, ce.charttime, ce.itemid;

-- ============================================================================
-- Expected Output:
-- - ~10-50 million rows (depends on cohort size)
-- - Temporal resolution: Irregular (every few minutes to hours)
-- - Will be binned into 48 1-hour windows in Python
-- ============================================================================

-- To save as a table:
-- CREATE OR REPLACE TABLE `your-project.your_dataset.chartevents_cohort` AS
-- (... query above ...)

-- To export to GCS (recommended for large data):
-- EXPORT DATA OPTIONS(
--   uri='gs://your-bucket/chartevents_cohort_*.parquet',
--   format='PARQUET',
--   overwrite=true
-- ) AS
-- (... query above ...)
