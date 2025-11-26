# Performance Notebook Organization

## Proper Execution Order

This document outlines the proper order for running cells in `performancen_notebook.ipynb`.

**Note**: Jupyter notebooks will skip cells that have already been executed, but it's best to run them in order to ensure dependencies are met.

---

## SECTION 1: SETUP & DATA PREPARATION (Run Once)

### Cell 1: Documentation Header
- **Purpose**: Overview of all steps
- **Action**: Read-only, no execution needed

### Cell 2: Assemble Full PI Tensors
- **Purpose**: Concatenate batch pi tensors into full tensors
- **Script**: `assemble_full_pi_tensor.py`
- **When to run**: ONCE before any generation steps
- **After running**: Mark as "not evaluated" to prevent re-running
- **Output**: `pi_enroll_fixedphi_sex_FULL.pt` files

---

## SECTION 2: GENERATE PREDICTIONS (Run Once Each)

### Cell 3: Generate Time Horizon Predictions
- **Purpose**: Generate 5yr, 10yr, 30yr, static 10yr predictions
- **Script**: `generate_time_horizon_predictions.py`
- **When to run**: ONCE after PI tensors are assembled
- **After running**: Mark as "not evaluated"
- **Output**: `results/time_horizons/{approach}/*.csv`

### Cell 4: Generate Washout Predictions
- **Purpose**: Generate 1yr predictions with 0yr, 1yr, 2yr washout
- **Script**: `generate_washout_predictions.py`
- **When to run**: ONCE after PI tensors are assembled
- **After running**: Mark as "not evaluated"
- **Output**: `results/washout/{approach}/*.csv`

### Cell 5: Generate Age Offset Predictions (Optional)
- **Purpose**: Rolling 1yr predictions with different training offsets
- **Script**: `generate_age_offset_predictions.py`
- **When to run**: ONCE (optional analysis)
- **After running**: Mark as "not evaluated"
- **Output**: `results/age_offset/{approach}/*.csv`

---

## SECTION 3: LOAD RESULTS (Safe to Run Multiple Times)

### Cell 6: Load Generated Results
- **Purpose**: Load all CSV files into dictionaries
- **Action**: Safe to run multiple times (reloads without regenerating)
- **Output**: `time_horizon_results`, `washout_results`, `age_offset_results`

---

## SECTION 4: COMPARISONS & VALIDATIONS (Run Once Each)

### Cell 7: Compare AWS vs Local
- **Purpose**: Validate reproducibility across environments
- **Script**: `compare_aws_local_cell.py`
- **When to run**: ONCE after results are generated
- **Output**: `results/validation/AWS_vs_Local_*.csv`

### Cell 8: Compare with External Scores
- **Purpose**: Compare with PCE, PREVENT, Gail, QRISK3
- **Script**: `compare_with_external_scores.py`
- **When to run**: ONCE after results are generated
- **Output**: `results/comparisons/{approach}/external_scores_comparison.csv`

### Cell 9: Compare with Delphi
- **Purpose**: Compare 1yr predictions with Delphi-2M
- **Script**: `compare_delphi_1yr_import.py`
- **When to run**: ONCE after washout results are generated
- **Output**: `results/comparisons/{approach}/delphi_comparison_*.csv`

### Cell 10: Compare Multi-Horizon with Delphi
- **Purpose**: Compare multiple horizons (5yr, 10yr, 30yr) with Delphi
- **Script**: `compare_delphi_multihorizon.py`
- **When to run**: ONCE after time horizon results are generated
- **Output**: `results/comparisons/{approach}/delphi_comparison_multihorizon.csv`

### Cell 11: Compare with Cox Baseline
- **Purpose**: Compare static 10yr with Cox baseline (age + sex only)
- **Script**: `compare_with_cox_baseline.py`
- **When to run**: ONCE after static 10yr results are generated
- **Output**: `results/comparisons/{approach}/cox_baseline_comparison_*.csv`

### Cell 12: Analyze Prediction Drops
- **Purpose**: Analyze why predictions drop between 0yr and 1yr washout
- **Script**: `analyze_prediction_drops.py`
- **When to run**: ONCE after washout results are generated
- **Output**: `results/analysis/prediction_drops_*.csv`

---

## SECTION 5: VISUALIZATIONS (Safe to Run Multiple Times)

### Cell 13: Visualize Prediction Drops
- **Purpose**: Create plots for prediction drops analysis
- **Script**: `visualize_prediction_drops.py`
- **When to run**: After prediction drops analysis is complete
- **Output**: `results/analysis/plots/*.png`

### Cell 14: Visualize All Comparisons
- **Purpose**: Create plots for all comparisons
- **Script**: `visualize_all_comparisons.py`
- **When to run**: After all comparisons are complete
- **Output**: `results/comparisons/plots/*.png`

---

## SECTION 6: SUMMARY & INTERPRETATION (Read-Only)

### Cell 15: Key Findings Summary
- **Purpose**: Summary of main results
- **Action**: Read-only, update manually

### Cell 16: Interpretation Notes
- **Purpose**: Interpretation of results
- **Action**: Read-only, update manually

---

## Execution Strategy

1. **First Time Setup**:
   - Run Section 1 (Setup)
   - Run Section 2 (Generate Predictions)
   - Mark generation cells as "not evaluated"

2. **Regular Analysis**:
   - Run Section 3 (Load Results) - safe to run anytime
   - Run Section 4 (Comparisons) - run once each
   - Run Section 5 (Visualizations) - safe to run anytime

3. **Re-running**:
   - Jupyter will skip cells that have already been executed
   - To re-run a cell, use "Run Cell" or "Run All Above"
   - Generation cells should remain "not evaluated" unless you want to regenerate

---

## File Dependencies

```
assemble_full_pi_tensor.py
    ↓
generate_time_horizon_predictions.py
generate_washout_predictions.py
generate_age_offset_predictions.py
    ↓
compare_aws_local_cell.py
compare_with_external_scores.py
compare_delphi_1yr_import.py
compare_delphi_multihorizon.py
compare_with_cox_baseline.py
analyze_prediction_drops.py
    ↓
visualize_prediction_drops.py
visualize_all_comparisons.py
```

---

## Notes

- **Mark cells as "not evaluated"**: Right-click cell → "Disable Output" or clear output, then mark as not evaluated
- **Skip already-run cells**: Jupyter automatically skips cells that have been executed
- **Re-run safely**: Section 3 and 5 can be run multiple times without issues
- **Regeneration**: Only re-run Section 2 if you need to regenerate predictions (takes time)

