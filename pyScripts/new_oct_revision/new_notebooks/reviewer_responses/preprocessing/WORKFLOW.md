# Aladynoulli Model Workflow

This document describes the complete workflow for running the Aladynoulli model from preprocessing to prediction.

## Overview

The workflow consists of 4 main steps:

1. **Preprocessing** - Create initialization files (prevalence, clusters, psi, reference trajectories)
2. **Batch Training** - Train model on batches with FULL E matrix
3. **Create Master Checkpoint** - Pool phi from batches and create master checkpoint
4. **Predict** - Run predictions using master checkpoint

---

## Step 1: Preprocessing

Create initialization files needed for model training.

### Files Created:
- `prevalence_t` - Smoothed disease prevalence over time (D × T)
- `initial_clusters_400k.pt` - Disease-to-signature cluster assignments (D,)
- `initial_psi_400k.pt` - Initial signature-disease association parameters (K × D)
- `reference_trajectories.pt` - Signature reference trajectories (K × T) and healthy ref (T,)

### Usage:

**Option A: Use the notebook** (interactive)
```python
# In create_preprocessing_files.ipynb
from preprocessing_utils import (
    compute_smoothed_prevalence,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)

# Step 1: Compute prevalence
prevalence_t = compute_smoothed_prevalence(Y, window_size=5, smooth_on_logit=True)

# Step 2: Create clusters and psi
clusters, psi = create_initial_clusters_and_psi(
    Y=Y, K=20, random_state=42
)

# Step 3: Create reference trajectories
signature_refs, healthy_ref = create_reference_trajectories(
    Y, clusters, K=20
)
```

**Option B: Use as Python script** (programmatic)
```python
from preprocessing.preprocessing_utils import (
    compute_smoothed_prevalence,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)
# ... same as above
```

### Key Benefits:
- ✅ **Standalone functions** - No need to initialize full model
- ✅ **Much faster** - Avoids heavy model initialization
- ✅ **Reproducible** - Same `random_state=42` produces identical clusters
- ✅ **Reusable** - Can be imported in any script/notebook

---

## Step 2: Batch Training

Train the model on batches with FULL E matrix (enrollment data).

### Script:
```bash
python claudefile/run_aladyn_batch.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --use_full_E \
    --num_batches 40
```

### What it does:
- Trains model on batches of data
- Uses FULL E matrix (enrollment information)
- Saves checkpoints: `enrollment_model_W0.0001_batch_*_*.pt`
- Each checkpoint contains trained `phi` parameters

### Output:
- Batch checkpoints in output directory
- Each checkpoint has `phi` (K × D × T) learned from that batch

---

## Step 3: Create Master Checkpoint

Pool phi from all batches and create master checkpoint for fixed-phi predictions.

### Script:
```bash
python claudefile/create_master_checkpoints.py \
    --data_dir /path/to/data \
    --retrospective_pattern "/path/to/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /path/to/output
```

### What it does:
- Loads phi from all batch checkpoints
- Pools phi (mean across batches)
- Combines with `initial_psi_400k.pt`
- Creates master checkpoint: `master_for_fitting_pooled_all_data.pt`

### Output:
- `master_for_fitting_pooled_all_data.pt` - Master checkpoint with pooled phi + initial_psi
- Can be used for fixed-phi predictions

### Alternative: Use notebook
The `misc/evalmodel/lifetime.ipynb` notebook also shows how to create master checkpoints interactively.

---

## Step 4: Predict with Master

Run predictions using the master checkpoint (fixed phi).

### Script:
```bash
python claudefile/run_aladyn_predict_with_master.py \
    --trained_model_path /path/to/master_for_fitting_pooled_all_data.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions
```

### What it does:
- Loads master checkpoint (pooled phi + initial_psi)
- **Automatically loads E matrix** from `data_dir/E_enrollment_full.pt` (FULL E, enrollment data)
- Also loads Y, G, and model_essentials from data_dir
- Uses fixed phi for predictions (doesn't retrain phi)
- Generates predictions for all patients
- Saves prediction results

### Required files in `--data_dir`:
- `Y_tensor.pt` - Disease outcomes
- `E_enrollment_full.pt` - **Enrollment matrix (FULL E)** - automatically loaded
- `G_matrix.pt` - Genetic variants
- `model_essentials.pt` - Model metadata
- `reference_trajectories.pt` - Signature reference trajectories

---

## Complete Workflow Example

```bash
# Step 1: Preprocessing (in notebook or script)
# Creates: prevalence_t, initial_clusters_400k.pt, initial_psi_400k.pt, reference_trajectories.pt

# Step 2: Batch training
python claudefile/run_aladyn_batch.py \
    --data_dir /path/to/data \
    --output_dir /path/to/batch_output \
    --use_full_E \
    --num_batches 40

# Step 3: Create master checkpoint
python claudefile/create_master_checkpoints.py \
    --retrospective_pattern "/path/to/batch_output/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /path/to/data

# Step 4: Predict
python claudefile/run_aladyn_predict_with_master.py \
    --trained_model_path /path/to/data/master_for_fitting_pooled_all_data.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions
```

---

## File Locations

### Preprocessing Files:
- `preprocessing/preprocessing_utils.py` - Standalone utility functions
- `preprocessing/create_preprocessing_files.ipynb` - Interactive notebook

### Training Scripts:
- `claudefile/run_aladyn_batch.py` - Batch training script
- `claudefile/create_master_checkpoints.py` - Master checkpoint creation

### Prediction Scripts:
- `claudefile/run_aladyn_predict_with_master.py` - Predict with master checkpoint

---

## Notes

- **Preprocessing functions are standalone** - No model initialization needed
- **Clusters are deterministic** - Same `random_state=42` produces identical clusters
- **Master checkpoint pools phi** - Mean across all batches for stability
- **Fixed phi predictions** - Faster than retraining phi for each prediction

