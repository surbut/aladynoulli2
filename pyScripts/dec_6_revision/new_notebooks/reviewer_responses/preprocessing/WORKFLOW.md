# Aladynoulli Model Workflow

This document describes the complete workflow for running the Aladynoulli model from preprocessing to prediction.

> **📚 Related Documentation**: This workflow is part of the [Reviewer Response Analyses](../README.md) section, which contains all interactive notebooks, preprocessing utilities, and detailed analyses. See the [main repository README](../../../../README.md) for the complete overview.

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
# In reviewer_responses/preprocessing/create_preprocessing_files.ipynb
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
# From reviewer_responses/preprocessing/preprocessing_utils.py
import sys
sys.path.append('pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing')
from preprocessing_utils import (
    compute_smoothed_prevalence,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)
# ... same as above
```

**See also:**
- Interactive notebook: [`reviewer_responses/preprocessing/create_preprocessing_files.ipynb`](../preprocessing/create_preprocessing_files.ipynb)
- Simple example: [`reviewer_responses/preprocessing/SIMPLE_EXAMPLE.py`](../preprocessing/SIMPLE_EXAMPLE.py)

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
# From repository root
python claudefile/run_aladyn_batch.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --start_index 0 \
    --end_index 10000 \
    --num_epochs 200
```

**Script location:** [`claudefile/run_aladyn_batch.py`](../../../../claudefile/run_aladyn_batch.py)  
**Model file:** [`pyScripts_forPublish/clust_huge_amp.py`](../../../../pyScripts_forPublish/clust_huge_amp.py)

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

### Option A: Use the script (recommended)
```bash
# From repository root
python claudefile/create_master_checkpoints.py \
    --data_dir /path/to/data \
    --retrospective_pattern "/path/to/enrollment_model_W0.0001_batch_*_*.pt" \
    --enrollment_pattern "/path/to/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /path/to/output
```

**Script location:** [`claudefile/create_master_checkpoints.py`](../../../../claudefile/create_master_checkpoints.py)

**What it does:**
- Loads phi from all batch checkpoints
- Pools phi (mean across batches)
- Combines with `initial_psi_400k.pt`
- Creates master checkpoint: `master_for_fitting_pooled_all_data.pt`

### Option B: Use interactive notebook

You can also create master checkpoints interactively by importing the functions:

```python
import sys
sys.path.append('/path/to/aladynoulli2/claudefile/')
from create_master_checkpoints import pool_phi_from_batches, create_master_checkpoint
import torch
import numpy as np

# Load initial_psi
data_dir = '/path/to/data/'
initial_psi = torch.load(data_dir + 'initial_psi_400k.pt', weights_only=False)
if torch.is_tensor(initial_psi):
    initial_psi = initial_psi.cpu().numpy()

# Pool phi from retrospective batches
retrospective_pattern = '/path/to/enrollment_model_W0.0001_batch_*_*.pt'
phi_retrospective_pooled = pool_phi_from_batches(retrospective_pattern)

# Create master checkpoint
output_path = data_dir + 'master_for_fitting_pooled_all_data.pt'
create_master_checkpoint(
    phi_retrospective_pooled,
    initial_psi,
    output_path,
    description="Pooled phi from all retrospective batches + initial_psi"
)
```

**Example notebook:** [`misc/evalmodel/lifetime.ipynb`](../../../../misc/evalmodel/lifetime.ipynb) shows the complete interactive workflow with comparisons to notebook results.

### Output:
- `master_for_fitting_pooled_all_data.pt` - Master checkpoint with pooled phi + initial_psi
- `master_for_fitting_pooled_enrollment_data.pt` - Master checkpoint from enrollment batches (if created)
- Can be used for fixed-phi predictions

---

## Step 4: Predict with Master

Run predictions using the master checkpoint (fixed phi).

### Script:
```bash
# From repository root
python claudefile/version_from_ec2/run_aladyn_predict_with_master.py \
    --trained_model_path /path/to/master_for_fitting_pooled_all_data.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions \
    --batch_size 10000 \
    --num_epochs 200
```

**Script location:** [`claudefile/version_from_ec2/run_aladyn_predict_with_master.py`](../../../../claudefile/version_from_ec2/run_aladyn_predict_with_master.py)  
**Model file:** [`pyScripts_forPublish/clust_huge_amp_fixedPhi.py`](../../../../pyScripts_forPublish/clust_huge_amp_fixedPhi.py)

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
python claudefile/version_from_ec2/run_aladyn_predict_with_master.py \
    --trained_model_path /path/to/data/master_for_fitting_pooled_all_data.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions \
    --batch_size 10000
```

---

## File Locations

### Preprocessing Files (in `reviewer_responses/preprocessing/`):
- [`preprocessing_utils.py`](../preprocessing/preprocessing_utils.py) - Standalone utility functions
- [`create_preprocessing_files.ipynb`](../preprocessing/create_preprocessing_files.ipynb) - Interactive notebook
- [`SIMPLE_EXAMPLE.py`](../preprocessing/SIMPLE_EXAMPLE.py) - Simple usage example

### Training Scripts (in `claudefile/`):
- [`run_aladyn_batch.py`](../../../../claudefile/run_aladyn_batch.py) - Batch training script
- [`create_master_checkpoints.py`](../../../../claudefile/create_master_checkpoints.py) - Master checkpoint creation

### Prediction Scripts (in `claudefile/version_from_ec2/`):
- [`run_aladyn_predict_with_master.py`](../../../../claudefile/version_from_ec2/run_aladyn_predict_with_master.py) - Predict with master checkpoint

### Model Files (in `pyScripts_forPublish/`):
- [`clust_huge_amp.py`](../../../../pyScripts_forPublish/clust_huge_amp.py) - Discovery mode (learns phi)
- [`clust_huge_amp_fixedPhi.py`](../../../../pyScripts_forPublish/clust_huge_amp_fixedPhi.py) - Prediction mode (fixed phi)

### Related Documentation:
- [Reviewer Response Analyses](../README.md) - All analysis notebooks and preprocessing utilities
- [Main Repository README](../../../../README.md) - Complete overview and installation

---

## Notes

- **Preprocessing functions are standalone** - No model initialization needed
- **Clusters are deterministic** - Same `random_state=42` produces identical clusters
- **Master checkpoint pools phi** - Mean across all batches for stability
- **Fixed phi predictions** - Faster than retraining phi for each prediction

## Related Analyses

After completing the workflow, see the [Reviewer Response Analyses](../README.md) for:
- Performance evaluation notebooks
- Clinical utility analyses
- Model validation studies
- Comparison with established risk scores
- All interactive analyses addressing reviewer questions

