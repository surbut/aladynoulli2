# MGB Medication Data Transformation Guide (R Version)

## Overview

Transform MGB medication data from its native format to the format expected by pathway analysis code using R.

## Quick Start

```r
# Load the script
source("transform_mgb_medications.R")

# Transform your MGB medication data
transformed_data <- transform_mgb_medications(
  mgb_med_file = "mgb_medications.csv",
  output_file = "mgb_medications_transformed.csv"
)
```

## Usage

### Basic Usage

```r
source("transform_mgb_medications.R")

transformed_data <- transform_mgb_medications(
  mgb_med_file = "/path/to/mgb/medications.csv",
  output_file = "/path/to/mgb/medications_transformed.csv"
)
```

### With Birth Dates (for age calculation)

```r
transformed_data <- transform_mgb_medications(
  mgb_med_file = "/path/to/mgb/medications.csv",
  patient_birth_dates_file = "/path/to/patient_birth_dates.csv",  # Columns: EMPI, birth_date
  output_file = "/path/to/mgb/medications_transformed.csv"
)
```

### Command Line Usage

```bash
Rscript transform_mgb_medications.R mgb_medications.csv
Rscript transform_mgb_medications.R mgb_medications.csv birth_dates.csv output.csv
```

## Input Format

Your MGB medication data should have:
- `EMPI`: Patient ID (integer)
- `Medication`: Medication name (string)
- `Medication_Date`: Date in format like "6/7/1998" or similar
- `Clinic`: Clinic name (optional)
- `Inpatient_Outpatient`: Visit type (optional)

## Output Format

The transformed data will have:
- `eid`: Patient ID (renamed from EMPI)
- `drug_name`: Normalized medication name (lowercase, trimmed)
- `read_2`: Medication code (uses drug_name since no READ codes available)
- `bnf_code`: BNF category code (inferred from medication name when possible)
- `Medication_Date`: Standardized date format
- `Medication_Name`: Original medication name
- `age_at_prescription`: Age at prescription (if birth dates provided)

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

## Example Workflow in R

```r
# Load required libraries
library(data.table)
library(lubridate)

# Load the transformation script
source("transform_mgb_medications.R")

# Transform MGB medication data
transformed_meds <- transform_mgb_medications(
  mgb_med_file = "mgb_medications.csv",
  patient_birth_dates_file = "patient_birth_dates.csv",
  output_file = "mgb_medications_transformed.csv"
)

# Check the results
head(transformed_meds)
summary(transformed_meds)
```

## Medication Names

**Important**: MGB uses different medication naming conventions than UKB. This is expected and fine!

The pathway analysis will work with whatever medication names you have. The script will:
- Keep original medication names in `Medication_Name`
- Create normalized versions in `drug_name` for matching/grouping
- Use medication names directly for analysis (no need for exact UKB matches)

## BNF Category Inference (Optional)

The script includes basic BNF category inference from medication names. This is **optional** - if medications don't match, that's fine. The pathway analysis will work with medication names directly.

BNF categories are only used for broad category grouping. Common matches:
- **01 - Gastro-intestinal**: omeprazole, pantoprazole, etc.
- **02 - Cardiovascular**: statins, ACE inhibitors, beta blockers, etc.
- **03 - Respiratory**: inhalers, bronchodilators
- **04 - CNS**: pain medications, antidepressants
- **05 - Infections**: antibiotics
- **06 - Endocrine**: diabetes medications, thyroid hormones
- **10 - Musculoskeletal**: NSAIDs

For medications that don't match, `bnf_code` will be `NA`. This is perfectly fine - the analysis will identify long-term medication patterns regardless of BNF codes.

## Using with Pathway Analysis

Once you have the transformed medication data, you can use it with the Python pathway analysis:

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

## Troubleshooting

### Dates not parsing correctly

The script tries multiple date formats automatically:
- `%m/%d/%Y` (e.g., "6/7/1998")
- `%Y-%m-%d` (e.g., "1998-06-07")
- `%m-%d-%Y` (e.g., "06-07-1998")
- `%d/%m/%Y` (e.g., "07/06/1998")
- `%Y/%m/%d` (e.g., "1998/06/07")

If your dates still don't parse, check the format and you may need to pre-process them.

### Missing birth dates

Age calculation is optional. If you don't have birth dates:
- The pathway analysis will still work
- Age-based filtering won't be available
- Long-term medication analysis will use prescription counts only

### Patient ID mismatch

Make sure:
- The `EMPI` column in medication data matches patient IDs in your `processed_ids.csv`
- The transformed data uses `eid` column (which should match `processed_ids`)

## Required R Packages

```r
install.packages(c("data.table", "lubridate"))
```

## Notes

- The script uses `data.table` for efficient processing of large files
- Preserves all original medication names in `Medication_Name` column
- Normalized names (`drug_name`) are used for matching and analysis
- Missing BNF codes are okay - the analysis will work with medication names
- Large files may take a few minutes to process

