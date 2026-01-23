# Leave-One-Out Validation for Corrected E Data

This directory contains scripts for performing leave-one-out validation on the corrected E data to address concerns about in-sample bias.

## Overview

The validation process:
1. **Pool phi** from all batches EXCEPT one (e.g., exclude batch 0, pool batches 1-39)
2. **Create checkpoint** with pooled phi
3. **Run predictions** on the excluded batch using that checkpoint
4. **Calculate 10-year AUC** for that batch
5. **Repeat** for all 40 batches
6. **Compare** leave-one-out AUCs to overall pooled AUC

## Scripts

### 1. `create_leave_one_out_checkpoints_correctedE.py`
Creates leave-one-out checkpoints by pooling phi from all batches except specified ones.

**Usage:**
```bash
# Create checkpoint excluding batch 0
python claudefile/create_leave_one_out_checkpoints_correctedE.py --exclude_batch 0

# Create checkpoint excluding batch 1
python claudefile/create_leave_one_out_checkpoints_correctedE.py --exclude_batch 1

# Or create all at once (see shell script)
```

**Output:** `master_for_fitting_pooled_correctedE_exclude_batch_{N}.pt` in data_dir

### 2. `run_leave_one_out_predictions_correctedE.py`
Runs predictions on excluded batches using the leave-one-out checkpoints.

**Usage:**
```bash
# Run predictions for a single batch
python claudefile/run_leave_one_out_predictions_correctedE.py --batch 0

# Run predictions for all batches
python claudefile/run_leave_one_out_predictions_correctedE.py --all_batches
```

**Output:** Predictions saved to `leave_one_out_correctedE/batch_{N}/pi_enroll_fixedphi_sex_{start}_{end}.pt`

### 3. `calculate_leave_one_out_auc_correctedE.py` (Option A: Per-Batch AUC)
Calculates 10-year AUC for each excluded batch separately using the predictions.

**Usage:**
```bash
# Calculate AUC for a single batch
python claudefile/calculate_leave_one_out_auc_correctedE.py --batch 0

# Calculate AUC for all batches
python claudefile/calculate_leave_one_out_auc_correctedE.py --all_batches
```

**Output:** CSV file with AUC results per batch: `leave_one_out_auc_results_correctedE.csv`

### 3b. `calculate_leave_one_out_auc_pooled_correctedE.py` (Option B: Pooled AUC)
Alternative approach: pools all leave-one-out predictions together and calculates one AUC on the combined dataset.

**Usage:**
```bash
python claudefile/calculate_leave_one_out_auc_pooled_correctedE.py
```

**Output:** CSV file with pooled AUC results: `leave_one_out_auc_pooled_correctedE.csv`

**Note:** Both approaches should give similar results. Option A (per-batch) allows you to see variability across batches, while Option B (pooled) gives a single overall AUC comparable to the main results.

### 4. `compare_leave_one_out_auc_correctedE.py`
Compares leave-one-out AUCs to overall pooled AUC and generates summary statistics.

**Usage:**
```bash
python claudefile/compare_leave_one_out_auc_correctedE.py
```

**Output:** Comparison CSV: `leave_one_out_comparison_correctedE.csv`

### 5. `run_full_leave_one_out_validation_correctedE.sh`
Master script that runs the complete pipeline (checkpoint creation only - predictions and AUC calculation need to be run separately due to runtime).

**Usage:**
```bash
bash claudefile/run_full_leave_one_out_validation_correctedE.sh
```

## Complete Workflow

### Step 1: Create Leave-One-Out Checkpoints
```bash
# Create all 40 checkpoints (one for each excluded batch)
for i in {0..39}; do
    python claudefile/create_leave_one_out_checkpoints_correctedE.py --exclude_batch $i
done
```

Or use the shell script:
```bash
bash claudefile/run_full_leave_one_out_validation_correctedE.sh
```

### Step 2: Run Predictions
```bash
python claudefile/run_leave_one_out_predictions_correctedE.py --all_batches
```

**Note:** This will take a while (40 batches × prediction time per batch). You may want to run in background:
```bash
nohup python claudefile/run_leave_one_out_predictions_correctedE.py --all_batches > loo_predictions.log 2>&1 &
```

### Step 3: Calculate AUC

**Option A: Per-batch AUC (recommended for seeing variability)**
```bash
python claudefile/calculate_leave_one_out_auc_correctedE.py --all_batches
```

**Option B: Pooled AUC (alternative approach)**
```bash
python claudefile/calculate_leave_one_out_auc_pooled_correctedE.py
```

Both use the **exact same evaluation function and parameters** as the performance notebook:
- `evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi`
- `n_bootstraps=100`
- `follow_up_duration_years=10`
- `patient_indices=None`

### Step 4: Compare Results
```bash
python claudefile/compare_leave_one_out_auc_correctedE.py
```

## Expected Results

If the leave-one-out validation is successful:
- **AUCs should be similar** across batches (low coefficient of variation)
- **AUC range** (max - min) should be small
- **Mean leave-one-out AUC** should be similar to overall pooled AUC

This demonstrates that:
- Pooling phi is robust (no overfitting to specific batches)
- Model performance is consistent across different subsets
- The overall results are reliable and not biased by in-sample evaluation

## File Locations

- **Checkpoints:** `{data_dir}/master_for_fitting_pooled_correctedE_exclude_batch_{N}.pt`
- **Predictions:** `{data_dir}/leave_one_out_correctedE/batch_{N}/pi_enroll_fixedphi_sex_{start}_{end}.pt`
- **AUC Results:** `{data_dir}/leave_one_out_auc_results_correctedE.csv`
- **Comparison:** `{data_dir}/leave_one_out_comparison_correctedE.csv`

## Configuration

Default paths (can be overridden with command-line arguments):
- **Data directory:** `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/`
- **Batch pattern:** `/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt`
  
  **Note:** The pattern for `censor_e_batchrun_vectorized` is `enrollment_model_W0.0001_batch_*_*.pt` (without "VECTORIZED"), which is different from `censor_e_batchrun_vectorized_11726` which uses `enrollment_model_VECTORIZED_W0.0001_batch_*_*.pt`.
- **Total batches:** 40
- **Batch size:** 10,000 samples

## Notes

- The validation uses **10-year risk predictions** (as in the performance notebook)
- Uses `evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi` from `fig5utils.py`
- Bootstrap CIs are calculated (100 bootstraps by default)
- Results can be compared to overall pooled AUC from the performance notebook

