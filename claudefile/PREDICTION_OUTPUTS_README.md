# Prediction Outputs Documentation

This document describes how the prediction outputs in `enrollment_predictions_fixedphi_RETROSPECTIVE_pooled` and `leave_one_out_validation` were generated.

## Overview

Both analyses use the Aladyn survival model with **fixed phi and psi parameters** (learned from pooled batches) and only estimate **lambda (genetic effects)** per batch. This approach:
- Uses pre-trained disease signatures (phi) and disease-to-signature mappings (psi) from pooled data
- Only estimates genetic effects (lambda) for each batch, making predictions faster and more consistent
- Processes data in batches of 10,000 samples

---

## 1. Enrollment Predictions with Retrospective Pooled Phi

**Directory:** `enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/`

### Purpose
Generate predictions on enrollment data using phi parameters pooled from **all retrospective data** (all batches combined).

### Methodology
1. **Master Checkpoint Creation**: Phi and psi were pooled from all retrospective batches using `create_master_checkpoints.py`, creating:
   - `master_for_fitting_pooled_all_data.pt`

2. **Prediction Generation**: The script `run_aladyn_predict_with_master_lap2.py` was used with:
   - Fixed phi from pooled retrospective data
   - Fixed psi from pooled retrospective data
   - Only lambda (genetic effects) estimated per batch

### Command Used
```bash
cd /Users/sarahurbut/aladynoulli2/claudefile

nohup python run_aladyn_predict_with_master_lap2.py \
    --trained_model_path /Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/master_for_fitting_pooled_all_data.pt \
    --output_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/ \
    --start_batch 10 \
    --max_batches 31 \
    > predict_retrospective_pooled_batches10-40.log 2>&1 &
```

### Output Files
- `pi_enroll_fixedphi_sex_{start}_{stop}.pt`: Predictions (pi) for each batch
- `model_enroll_fixedphi_sex_{start}_{stop}.pt`: Model state for each batch
- `pi_enroll_fixedphi_sex_FULL.pt`: Concatenated predictions from all batches
- `batch_info.pt`: Metadata about batches processed

### Key Parameters
- **Batch size**: 10,000 samples per batch
- **Batches processed**: 10-40 (31 batches total)
- **Phi source**: Pooled from all retrospective data
- **Estimated parameters**: Only lambda (genetic effects)

---

## 2. Leave-One-Out Validation

**Directory:** `leave_one_out_validation/batch_{N}/`

### Purpose
Validate the robustness of the pooling approach by excluding one batch at a time from phi estimation, then predicting on that excluded batch. This tests whether the pooled phi parameters generalize well to unseen batches.

### Methodology

#### Step 1: Create Leave-One-Out Checkpoints
For each test batch, a master checkpoint is created that **excludes** that batch from phi pooling:
- **Script**: `create_leave_one_out_checkpoints.py`
- **Process**: 
  1. Loads phi from all retrospective batch checkpoints
  2. Excludes the test batch (e.g., batch 0, 6, 15, etc.)
  3. Pools phi from all remaining batches (mean across batches)
  4. Saves checkpoint: `master_for_fitting_pooled_all_data_exclude_batch_{N}.pt`

**Checkpoint Creation Command** (for reference):
```bash
python create_leave_one_out_checkpoints.py \
    --exclude_batch {N} \
    --analysis_type retrospective \
    --data_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/ \
    --retrospective_pattern "/path/to/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/ \
    --total_batches 40
```

#### Step 2: Predict on Excluded Batch
Predictions are made on the excluded batch using phi learned from all other batches:
- Uses the leave-one-out checkpoint (phi from all batches except test batch)
- Predicts only on the excluded batch (`--start_batch {N} --max_batches 1`)
- Tests whether phi generalizes to unseen data

### Why This Matters
This validation approach tests:
1. **Robustness**: Does phi pooled from other batches work well on the excluded batch?
2. **Overfitting**: Is the pooling approach overfitting to specific batches?
3. **Generalization**: Can we trust that pooled phi will work on new data?

**Expected Result**: If predictions on excluded batches have similar performance (AUCs) to predictions using phi from all batches, this validates that:
- Pooling is robust and generalizes well
- No significant overfitting to specific batches
- The approach is suitable for making predictions on new data

### Command Used
```bash
cd /Users/sarahurbut/aladynoulli2/claudefile

nohup bash -c '
for batch in 0 6 15 17 18 20 24 34 35 37; do
    echo "Starting batch $batch..."
    python run_aladyn_predict_with_master_lap2.py \
        --trained_model_path /Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/master_for_fitting_pooled_all_data_exclude_batch_${batch}.pt \
        --output_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/leave_one_out_validation/batch_${batch}/ \
        --start_batch ${batch} \
        --max_batches 1 \
        --data_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/ \
        --covariates_path /Users/sarahurbut/Library/CloudStorage/Dropbox/baselinagefamh_withpcs.csv
    echo "Completed batch $batch"
done
echo "All batches completed"
' > predict_loo_all_batches.log 2>&1 &

echo "Monitor progress with: tail -f predict_loo_all_batches.log"
```

### Test Batches
Batches tested: 0, 6, 15, 17, 18, 20, 24, 34, 35, 37

### Output Structure
Each test batch has its own directory:
```
leave_one_out_validation/
├── batch_0/
│   ├── pi_enroll_fixedphi_sex_{start}_{stop}.pt
│   ├── model_enroll_fixedphi_sex_{start}_{stop}.pt
│   └── batch_info.pt
├── batch_6/
│   └── ...
└── ...
```

### Key Parameters
- **Batch size**: 10,000 samples per batch
- **Batches per run**: 1 (only the excluded batch)
- **Phi source**: Pooled from all batches EXCEPT the test batch
- **Estimated parameters**: Only lambda (genetic effects)

### Interpreting LOO Results

To validate the pooling approach, compare:

1. **LOO Predictions** (phi from all batches except test batch, predicting on test batch)
   - Located in: `leave_one_out_validation/batch_{N}/`
   - These represent "out-of-sample" performance

2. **Full Pooled Predictions** (phi from all batches including test batch)
   - Located in: `enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/`
   - These represent "in-sample" performance

**Validation Criteria**:
- If LOO AUCs ≈ Full pooled AUCs → Pooling is robust, no overfitting
- If LOO AUCs << Full pooled AUCs → Possible overfitting or batch-specific effects
- If LOO AUCs vary significantly across batches → Some batches may be outliers

**Next Steps for Analysis**:
1. Load predictions from both directories
2. Calculate AUCs for each batch in both scenarios
3. Compare AUC distributions (mean, std, range)
4. Test if differences are statistically significant
5. If similar → pooling validated; proceed with pooled approach

---

## Scripts Used

### Main Prediction Script
- **`run_aladyn_predict_with_master_lap2.py`**: Main script for generating predictions with fixed phi/psi
  - Uses `AladynSurvivalFixedPhi` model class
  - Processes data in batches
  - Saves predictions and model states
  - Concatenates results at the end

### Supporting Scripts
- **`create_master_checkpoints.py`**: Creates pooled phi/psi checkpoints from multiple batches
- **`create_leave_one_out_checkpoints.py`**: Creates checkpoints excluding specific batches for validation

---

## Model Architecture

Both analyses use the **AladynSurvivalFixedPhi** model which:
- **Fixed parameters**:
  - `phi`: Disease signatures (learned from pooled batches)
  - `psi`: Disease-to-signature mappings (learned from pooled batches)
  - `gamma`: Initialized using psi_total for consistency
  
- **Estimated parameters**:
  - `lambda`: Genetic effects per batch (only parameter being optimized)

- **Inputs**:
  - `Y`: Disease trajectories tensor
  - `E`: Enrollment/event data
  - `G`: Genetic data (with sex and PCs)
  
- **Outputs**:
  - `pi`: Disease probability predictions (N × D × T)

---

## Data Processing

### Batch Processing
- Data is processed in batches of 10,000 samples
- Each batch is processed independently
- Predictions are saved per batch, then concatenated

### Covariates Included
- Sex
- Principal components (PCs 1-10) if `--include_pcs True`

### Data Sources
- **Y_tensor.pt**: Disease trajectories
- **E_enrollment_full.pt**: Enrollment/event data
- **G_matrix.pt**: Genetic data
- **baselinagefamh_withpcs.csv**: Covariates (sex, PCs)

---

## Validation Approach

The leave-one-out validation tests:
1. **Robustness**: Whether phi pooled from other batches generalizes to excluded batch
2. **Overfitting**: Whether pooling overfits to specific batches
3. **Consistency**: Whether predictions are stable across different batch exclusions

**Expected Result**: If AUCs from leave-one-out predictions are similar to pooled predictions, this validates that:
- Pooling is robust
- No significant overfitting to specific batches
- The approach generalizes well

---

## Notes

- All predictions use enrollment data (`E_enrollment_full.pt`)
- Random seeds are fixed (42) for reproducibility
- Memory is cleaned between batches to handle large datasets
- Processing can be monitored via log files

---

## File Naming Conventions

- **Predictions**: `pi_enroll_fixedphi_sex_{start}_{stop}.pt`
- **Models**: `model_enroll_fixedphi_sex_{start}_{stop}.pt`
- **Full concatenated**: `pi_enroll_fixedphi_sex_FULL.pt`
- **Metadata**: `batch_info.pt`

Where `{start}` and `{stop}` are sample indices (e.g., 0-10000, 10000-20000, etc.)

