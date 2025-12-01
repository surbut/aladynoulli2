# MGB Medication Data Transformation Guide

## Overview

The MGB medication data needs to be transformed from its native format to match the format expected by the pathway analysis code.

## Input Format (MGB)

Your MGB medication data should have:
- `EMPI`: Patient ID (integer)
- `Medication`: Medication name (string)
- `Medication_Date`: Date in format like "6/7/1998" or similar
- `Clinic`: Clinic name (optional)
- `Inpatient_Outpatient`: Visit type (optional)

## Output Format (Pathway Analysis)

The transformed data will have:
- `eid`: Patient ID (renamed from EMPI)
- `drug_name`: Normalized medication name (lowercase, trimmed)
- `read_2`: Medication code (uses drug_name since no READ codes available)
- `bnf_code`: BNF category code (inferred from medication name when possible)
- `Medication_Date`: Standardized date format
- `Medication_Name`: Original medication name
- `age_at_prescription`: Age at prescription (if birth dates provided)

## Usage

### Basic Usage

```python
from transform_mgb_medications import transform_mgb_medications

# Transform MGB medication data
transformed_data = transform_mgb_medications(
    mgb_med_file='/path/to/mgb/medications.csv',
    output_file='/path/to/mgb/medications_transformed.csv'
)
```

### With Birth Dates (for age calculation)

```python
transformed_data = transform_mgb_medications(
    mgb_med_file='/path/to/mgb/medications.csv',
    patient_birth_dates_file='/path/to/patient_birth_dates.csv',  # Columns: EMPI, birth_date
    output_file='/path/to/mgb/medications_transformed.csv'
)
```

### Command Line Usage

```bash
python transform_mgb_medications.py \
    --input /path/to/mgb/medications.csv \
    --birth_dates /path/to/patient_birth_dates.csv \
    --output /path/to/mgb/medications_transformed.csv
```

## Birth Date File Format

If you want to calculate age at prescription, provide a CSV file with:
- `EMPI`: Patient ID (must match medication data)
- `birth_date`: Date of birth (any standard date format)

Example:
```csv
EMPI,birth_date
100035476,1950-01-15
100035477,1965-03-22
```

## Using Transformed Data in Pathway Analysis

Once you have the transformed medication data:

1. **Update the medication integration code** to point to your transformed file:

```python
from medication_integration import integrate_medications_with_pathways

medication_results = integrate_medications_with_pathways(
    pathway_data=pathway_data,
    Y=Y,
    thetas=thetas,
    disease_names=disease_names,
    processed_ids=processed_ids,
    gp_scripts_path='/path/to/mgb/medications_transformed.csv'  # Use transformed file
)
```

2. **Note**: The transformed file will be tab-separated (`.tsv` or `.txt`), so the code will automatically detect it.

## BNF Category Inference

The script includes basic BNF category inference from medication names. It matches common medications to BNF categories:

- **01 - Gastro-intestinal**: omeprazole, pantoprazole, etc.
- **02 - Cardiovascular**: statins, ACE inhibitors, beta blockers, etc.
- **03 - Respiratory**: inhalers, bronchodilators
- **04 - CNS**: pain medications, antidepressants
- **05 - Infections**: antibiotics
- **06 - Endocrine**: diabetes medications, thyroid hormones
- **10 - Musculoskeletal**: NSAIDs

For medications that don't match, `bnf_code` will be `None`. This is okay - the pathway analysis will still work, but BNF category analysis will be limited.

## Troubleshooting

### Dates not parsing correctly

If dates aren't parsing, try:
1. Check the date format in your input file
2. The script tries multiple date formats automatically
3. You can pre-process dates in your input file to a standard format (YYYY-MM-DD)

### Missing birth dates

Age calculation is optional. If you don't have birth dates:
- The pathway analysis will still work
- Age-based filtering won't be available
- Long-term medication analysis will use prescription counts only

### Patient ID mismatch

Make sure:
- The `EMPI` column in medication data matches patient IDs in your `processed_ids.csv`
- The transformed data uses `eid` column (which should match `processed_ids`)

## Example Workflow

```python
# Step 1: Transform MGB medication data
transformed_meds = transform_mgb_medications(
    mgb_med_file='mgb_medications.csv',
    patient_birth_dates_file='patient_birth_dates.csv',
    output_file='mgb_medications_transformed.csv'
)

# Step 2: Run pathway analysis with transformed medications
from run_complete_pathway_analysis_deviation_only import run_deviation_only_analysis

results = run_deviation_only_analysis(
    "myocardial infarction", 
    n_pathways=4, 
    output_dir='output_10yr', 
    lookback_years=10
)

# The medication integration will automatically use the transformed file
# if you update the path in medication_integration.py or pass it as parameter
```

## Notes

- The script preserves all original medication names in `Medication_Name` column
- Normalized names (`drug_name`) are used for matching and analysis
- Missing BNF codes are okay - the analysis will work with medication names
- The script handles common date formats automatically
- Large files may take a few minutes to process

