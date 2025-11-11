-- ============================================================================
-- Extract Lab Events for Cohort (First 48 Hours)
-- ============================================================================
-- Purpose: Extract laboratory test results for cohort admissions
-- Time window: First 48 hours after admission
-- Items: CBC, Chemistry, Coagulation, Blood Gas
-- ============================================================================

WITH 
-- Get cohort hadm_ids
cohort AS (
  SELECT 
    hadm_id,
    subject_id,
    admittime,
    TIMESTAMP_ADD(admittime, INTERVAL 48 HOUR) AS end_time
  FROM `your-project.your_dataset.cohort_30d_readmit`
),

-- Define lab itemids of interest
labs_of_interest AS (
  SELECT itemid, label, fluid, category
  FROM `physionet-data.mimiciv_hosp.d_labitems`
  WHERE itemid IN (
    -- Complete Blood Count (CBC)
    51279,  -- Red Blood Cells
    51222,  -- Hemoglobin
    51248,  -- Hematocrit
    51265,  -- Platelet Count
    51301,  -- White Blood Cells
    51133,  -- Absolute Neutrophil Count
    
    -- Chemistry/Metabolic Panel
    50912,  -- Creatinine
    50971,  -- Potassium
    50983,  -- Sodium
    50902,  -- Chloride
    50882,  -- Bicarbonate
    50868,  -- Anion Gap
    50813,  -- Lactate
    50931,  -- Glucose
    
    -- Liver Function
    50861,  -- Alanine Aminotransferase (ALT)
    50878,  -- Aspartate Aminotransferase (AST)
    50885,  -- Bilirubin, Total
    50863,  -- Alkaline Phosphatase
    50862,  -- Albumin
    
    -- Renal Function
    51006,  -- Urea Nitrogen (BUN)
    50912,  -- Creatinine (duplicate, removed in dedup)
    
    -- Coagulation
    51237,  -- INR(PT)
    51274,  -- PT
    51275,  -- PTT
    
    -- Blood Gas (Arterial)
    50821,  -- PO2
    50818,  -- PCO2
    50820,  -- pH
    50804,  -- Total CO2
    
    -- Cardiac Markers
    51003,  -- Troponin T
    50911,  -- Creatine Kinase (CK)
    50910,  -- Creatine Kinase, MB Isoenzyme
    
    -- Other Important
    51221,  -- Hematocrit (Arterial)
    50893,  -- Calcium, Total
    50960,  -- Magnesium
    50970   -- Phosphate
  )
)

-- Extract lab events
SELECT 
  le.subject_id,
  le.hadm_id,
  le.itemid,
  dl.label AS lab_label,
  dl.fluid AS specimen_type,
  dl.category,
  le.charttime,
  le.value,
  le.valuenum,
  le.valueuom AS unit,
  le.ref_range_lower,
  le.ref_range_upper,
  le.flag AS abnormal_flag,  -- 'abnormal' if outside reference range
  -- Time relative to admission (in hours)
  TIMESTAMP_DIFF(le.charttime, c.admittime, HOUR) AS hours_since_admit,
  -- Time relative to admission (in minutes)
  TIMESTAMP_DIFF(le.charttime, c.admittime, MINUTE) AS minutes_since_admit,
  -- Abnormality indicator (for future use)
  CASE 
    WHEN le.valuenum < le.ref_range_lower THEN -1  -- Below normal
    WHEN le.valuenum > le.ref_range_upper THEN 1   -- Above normal
    ELSE 0  -- Normal
  END AS abnormal_indicator
FROM `physionet-data.mimiciv_hosp.labevents` AS le
INNER JOIN cohort AS c
  ON le.hadm_id = c.hadm_id
INNER JOIN labs_of_interest AS dl
  ON le.itemid = dl.itemid
WHERE 
  le.charttime >= c.admittime
  AND le.charttime <= c.end_time  -- First 48 hours only
  AND le.valuenum IS NOT NULL  -- Must have numeric value
  -- Remove extreme outliers (likely data errors)
  AND le.valuenum >= 0  -- No negative lab values
ORDER BY le.subject_id, le.hadm_id, le.charttime, le.itemid;

-- ============================================================================
-- Expected Output:
-- - ~5-20 million rows (depends on cohort size)
-- - Temporal resolution: Irregular (labs drawn every 6-24 hours typically)
-- - Will be binned into 48 1-hour windows in Python
-- ============================================================================

-- To save as a table:
-- CREATE OR REPLACE TABLE `your-project.your_dataset.labevents_cohort` AS
-- (... query above ...)

-- To export to GCS:
-- EXPORT DATA OPTIONS(
--   uri='gs://your-bucket/labevents_cohort_*.parquet',
--   format='PARQUET',
--   overwrite=true
-- ) AS
-- (... query above ...)

-- ============================================================================
-- Note on Lab Frequencies:
-- - CBC: Usually every 6-12 hours in ICU
-- - Chemistry: Every 6-24 hours
-- - Coagulation: Every 12-24 hours (or more frequent if on anticoagulation)
-- - Blood Gas: Every 2-6 hours in critically ill patients
-- - Cardiac markers: Every 6-12 hours if suspected MI
-- ============================================================================
