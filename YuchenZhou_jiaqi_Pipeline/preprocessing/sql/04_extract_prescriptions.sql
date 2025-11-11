-- ============================================================================
-- Extract Prescriptions for Cohort (First 48 Hours)
-- ============================================================================
-- Purpose: Extract medication prescriptions for cohort admissions
-- Time window: First 48 hours after admission
-- Focus: High-impact medications (antibiotics, vasopressors, sedatives, etc.)
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

-- Define medication categories of interest
-- (Can be expanded based on your specific needs)
meds_of_interest AS (
  SELECT DISTINCT
    drug,
    drug_type,
    formulary_drug_cd
  FROM `physionet-data.mimiciv_hosp.prescriptions`
  WHERE 
    -- Antibiotics (broad categories)
    LOWER(drug) LIKE '%cillin%'  -- Penicillins
    OR LOWER(drug) LIKE '%mycin%'  -- Aminoglycosides, Macrolides
    OR LOWER(drug) LIKE '%cycline%'  -- Tetracyclines
    OR LOWER(drug) LIKE '%floxacin%'  -- Fluoroquinolones
    OR LOWER(drug) LIKE '%cef%'  -- Cephalosporins
    OR LOWER(drug) LIKE '%vancomycin%'
    OR LOWER(drug) LIKE '%metronidazole%'
    OR LOWER(drug) LIKE '%meropenem%'
    OR LOWER(drug) LIKE '%piperacillin%'
    
    -- Vasopressors/Inotropes
    OR LOWER(drug) LIKE '%norepinephrine%'
    OR LOWER(drug) LIKE '%epinephrine%'
    OR LOWER(drug) LIKE '%dopamine%'
    OR LOWER(drug) LIKE '%dobutamine%'
    OR LOWER(drug) LIKE '%vasopressin%'
    OR LOWER(drug) LIKE '%phenylephrine%'
    
    -- Sedatives/Analgesics
    OR LOWER(drug) LIKE '%propofol%'
    OR LOWER(drug) LIKE '%midazolam%'
    OR LOWER(drug) LIKE '%fentanyl%'
    OR LOWER(drug) LIKE '%morphine%'
    OR LOWER(drug) LIKE '%hydromorphone%'
    OR LOWER(drug) LIKE '%dexmedetomidine%'
    
    -- Anticoagulants
    OR LOWER(drug) LIKE '%heparin%'
    OR LOWER(drug) LIKE '%warfarin%'
    OR LOWER(drug) LIKE '%enoxaparin%'
    OR LOWER(drug) LIKE '%apixaban%'
    OR LOWER(drug) LIKE '%rivaroxaban%'
    
    -- Diuretics
    OR LOWER(drug) LIKE '%furosemide%'
    OR LOWER(drug) LIKE '%lasix%'
    OR LOWER(drug) LIKE '%torsemide%'
    OR LOWER(drug) LIKE '%bumetanide%'
    
    -- Antihypertensives
    OR LOWER(drug) LIKE '%metoprolol%'
    OR LOWER(drug) LIKE '%labetalol%'
    OR LOWER(drug) LIKE '%esmolol%'
    OR LOWER(drug) LIKE '%hydralazine%'
    OR LOWER(drug) LIKE '%nicardipine%'
    
    -- Insulin
    OR LOWER(drug) LIKE '%insulin%'
    
    -- Steroids
    OR LOWER(drug) LIKE '%dexamethasone%'
    OR LOWER(drug) LIKE '%methylprednisolone%'
    OR LOWER(drug) LIKE '%prednisone%'
    OR LOWER(drug) LIKE '%hydrocortisone%'
    
    -- Antiplatelet
    OR LOWER(drug) LIKE '%aspirin%'
    OR LOWER(drug) LIKE '%clopidogrel%'
    OR LOWER(drug) LIKE '%plavix%'
    
    -- Respiratory
    OR LOWER(drug) LIKE '%albuterol%'
    OR LOWER(drug) LIKE '%ipratropium%'
)

-- Extract prescriptions
SELECT 
  p.subject_id,
  p.hadm_id,
  p.pharmacy_id,
  p.starttime,
  p.stoptime,
  p.drug_type,
  p.drug,
  p.formulary_drug_cd,
  p.gsn,  -- Generic Sequence Number
  p.ndc,  -- National Drug Code
  p.prod_strength,
  p.form_val_disp,
  p.form_unit_disp,
  p.doses_per_24_hrs,
  p.route,
  -- Time relative to admission
  TIMESTAMP_DIFF(p.starttime, c.admittime, HOUR) AS start_hour_since_admit,
  TIMESTAMP_DIFF(p.stoptime, c.admittime, HOUR) AS stop_hour_since_admit,
  -- Duration of prescription (hours)
  TIMESTAMP_DIFF(p.stoptime, p.starttime, HOUR) AS duration_hours,
  -- Binary indicator: was drug active in first 48 hours?
  CASE 
    WHEN p.starttime <= c.end_time THEN 1
    ELSE 0
  END AS active_in_48h,
  -- Categorize medication type (simple heuristic)
  CASE
    WHEN LOWER(p.drug) LIKE '%cillin%' OR LOWER(p.drug) LIKE '%mycin%' 
         OR LOWER(p.drug) LIKE '%floxacin%' OR LOWER(p.drug) LIKE '%cef%' THEN 'antibiotic'
    WHEN LOWER(p.drug) LIKE '%norepinephrine%' OR LOWER(p.drug) LIKE '%epinephrine%' 
         OR LOWER(p.drug) LIKE '%dopamine%' OR LOWER(p.drug) LIKE '%vasopressin%' THEN 'vasopressor'
    WHEN LOWER(p.drug) LIKE '%propofol%' OR LOWER(p.drug) LIKE '%midazolam%' 
         OR LOWER(p.drug) LIKE '%fentanyl%' THEN 'sedative'
    WHEN LOWER(p.drug) LIKE '%heparin%' OR LOWER(p.drug) LIKE '%warfarin%' 
         OR LOWER(p.drug) LIKE '%enoxaparin%' THEN 'anticoagulant'
    WHEN LOWER(p.drug) LIKE '%furosemide%' OR LOWER(p.drug) LIKE '%lasix%' THEN 'diuretic'
    WHEN LOWER(p.drug) LIKE '%metoprolol%' OR LOWER(p.drug) LIKE '%labetalol%' THEN 'antihypertensive'
    WHEN LOWER(p.drug) LIKE '%insulin%' THEN 'insulin'
    WHEN LOWER(p.drug) LIKE '%dexamethasone%' OR LOWER(p.drug) LIKE '%prednisone%' THEN 'steroid'
    ELSE 'other'
  END AS medication_category
FROM `physionet-data.mimiciv_hosp.prescriptions` AS p
INNER JOIN cohort AS c
  ON p.hadm_id = c.hadm_id
WHERE 
  -- Prescription started within first 48 hours
  p.starttime >= c.admittime
  AND p.starttime <= c.end_time
  -- Filter to medications of interest
  AND p.drug IN (SELECT drug FROM meds_of_interest)
ORDER BY p.subject_id, p.hadm_id, p.starttime;

-- ============================================================================
-- Expected Output:
-- - ~1-5 million rows (depends on cohort size and medication filtering)
-- - Temporal resolution: Prescription start/stop times
-- - Will be converted to binary indicators or counts per hour in Python
-- ============================================================================

-- To save as a table:
-- CREATE OR REPLACE TABLE `your-project.your_dataset.prescriptions_cohort` AS
-- (... query above ...)

-- To export to GCS:
-- EXPORT DATA OPTIONS(
--   uri='gs://your-bucket/prescriptions_cohort_*.parquet',
--   format='PARQUET',
--   overwrite=true
-- ) AS
-- (... query above ...)

-- ============================================================================
-- Note on Medication Features:
-- For time-series modeling, we typically convert prescriptions to:
-- 1. Binary indicators: Was drug X active in hour t? (0/1)
-- 2. Counts: How many unique drugs active in hour t?
-- 3. Categories: How many vasopressors/antibiotics/etc active in hour t?
-- ============================================================================
