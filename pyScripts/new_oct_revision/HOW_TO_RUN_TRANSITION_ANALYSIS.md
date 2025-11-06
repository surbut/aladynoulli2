# How to Run Transition Analysis (UKB vs MGB)

## Quick Start

### Option 1: Run as a Python Script

```bash
cd /Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision
python run_transition_analysis_ukb_mgb.py
```

### Option 2: Run in Jupyter Notebook

Add this cell to your notebook:

```python
from run_transition_analysis_ukb_mgb import run_transition_analysis_both_cohorts

results = run_transition_analysis_both_cohorts(
    transition_disease_name='Rheumatoid arthritis',
    target_disease_name='myocardial infarction',
    years_before=10,
    age_tolerance=5,
    min_followup=5
)
```

### Option 3: Investigate Signature 3 Pattern

To investigate why Signature 3 is higher in non-progressors:

```python
from interpret_signature_3_pattern import investigate_signature_3_pattern

results = investigate_signature_3_pattern(
    transition_disease_name='Rheumatoid arthritis',
    target_disease_name='myocardial infarction',
    signature_idx=3,
    years_before=10
)
```

---

## What It Does

1. **Loads UKB data** (Y, thetas, disease_names)
2. **Finds diseases** using flexible matching (handles naming differences)
3. **Runs UKB transition analysis** (RA → MI vs RA, no MI)
4. **Loads MGB data** from model file
5. **Runs MGB transition analysis** (same analysis)
6. **Compares results**:
   - Sample sizes
   - Signature trajectory correlations
   - Side-by-side plots

---

## Output

The script will:
- Print sample sizes (now fixed to show correct counts!)
- Show signature trajectory comparisons
- Save plots to `transition_analysis_ukb_mgb/` directory:
  - `ukb_rheumatoid_arthritis_to_myocardial_infarction_progression.png`
  - `mgb_rheumatoid_arthritis_to_myocardial_infarction_progression.png`
  - `ukb_mgb_comparison_rheumatoid_arthritis_to_myocardial_infarction.png`

---

## Troubleshooting

### "Could not find disease"

The script uses flexible matching to find diseases. If it can't find a match, it will:
1. Show all potential matches with scores
2. Print sample disease names from the cohort
3. Suggest alternative disease names

### "0 progressors, 0 non-progressors"

This was a bug (now fixed!). The script was looking for keys that didn't exist. It now correctly extracts counts from:
- `progressor_deviations` (list length = number of progressors)
- `non_progressor_deviations` (list length = number of non-progressors)
- `matched_pairs` (list length = number of matched pairs)

---

## Customization

You can analyze different disease transitions:

```python
results = run_transition_analysis_both_cohorts(
    transition_disease_name='Breast cancer',  # Change this
    target_disease_name='myocardial infarction',
    years_before=10,
    age_tolerance=5,
    min_followup=5
)
```

Or change the time window:

```python
results = run_transition_analysis_both_cohorts(
    transition_disease_name='Rheumatoid arthritis',
    target_disease_name='myocardial infarction',
    years_before=5,  # Shorter window
    age_tolerance=3,  # Tighter age matching
    min_followup=3
)
```

