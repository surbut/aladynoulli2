# Aladynoulli Model Workflow

This document describes the complete workflow for running the Aladynoulli model from preprocessing to prediction.

> **📚 Related Documentation**: This workflow is part of the [Reviewer Response Analyses](../README.md) section, which contains all interactive notebooks, preprocessing utilities, and detailed analyses. See the [main repository README](../../../../README.md) for the complete overview.

## Overview

The workflow consists of 5 main steps:

1. **Preprocessing** - Create initialization files (prevalence, clusters, psi, reference trajectories)
2. **Batch Training** - Train model on batches with FULL E matrix
3. **Create Master Checkpoint** - Pool phi from batches and create master checkpoint
4. **Pool Gamma & Kappa** - Pool genetic effects (gamma) and calibration (kappa) from training batches
5. **Predict** - Run predictions using master checkpoint with fixed phi, gamma, and kappa (only lambda is learned)

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
    compute_smoothed_prevalence_at_risk,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)

# Step 1: Compute prevalence (with at-risk filtering using corrected E matrix)
# Load E_corrected for proper at-risk filtering
E_corrected = torch.load('E_matrix_corrected.pt', weights_only=False)
prevalence_t = compute_smoothed_prevalence_at_risk(Y, E_corrected, window_size=5, smooth_on_logit=True)

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
sys.path.append('pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing')
from preprocessing_utils import (
    compute_smoothed_prevalence_at_risk,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)
# Load E_corrected and compute prevalence
E_corrected = torch.load('E_matrix_corrected.pt', weights_only=False)
prevalence_t = compute_smoothed_prevalence_at_risk(Y, E_corrected, window_size=5, smooth_on_logit=True)
# ... same as above for clusters and trajectories
```

**See also:**
- Interactive notebook: [`create_preprocessing_files.ipynb`](./create_preprocessing_files.ipynb)
- Simple example: [`SIMPLE_EXAMPLE.py`](./SIMPLE_EXAMPLE.py)

### Key Benefits:
- ✅ **Standalone functions** - No need to initialize full model
- ✅ **Much faster** - Avoids heavy model initialization
- ✅ **Reproducible** - Same `random_state=42` produces identical clusters
- ✅ **Reusable** - Can be imported in any script/notebook

---

## Step 2: Batch Training

Train the model on batches with matrix that has seen fully censored information. 

### Script:
```bash
# From repository root
python claudefile/run_aladyn_batch_vector_e_censor_nolor.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --start_index 0 \
    --end_index 10000 \
    --num_epochs 200
```

**Script location:** [`claudefile/run_aladyn_batch_vector_e_censor_nolor.py`](../../../../claudefile/run_aladyn_batch_vector_e_censor_nolor.py)  
**Note:** This script uses no LR (log-likelihood regularization) penalty on gamma, which is the version used for fixing gamma and kappa in the final workflow.  
**Model file:** [`pyScripts_forPublish/clust_huge_amp_vectorized.py`](../../../../pyScripts_forPublish/clust_huge_amp_vectorized.py)

### What it does:
- Trains model on batches of data
- Uses FULL E matrix (enrollment information+follow up)
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



### Output:
- `master_for_fitting_pooled_all_data.pt` - Master checkpoint with pooled phi + initial_psi
- `master_for_fitting_pooled_enrollment_data.pt` - Master checkpoint from enrollment batches (if created)
- Can be used for fixed-phi predictions

---

## Step 4: Pool Gamma & Kappa

Pool genetic effects (gamma) and calibration parameter (kappa) from training batch checkpoints. These will be fixed during prediction, ensuring complete separation between training and testing—only individual-specific lambda (signature loadings) is learned at prediction time.

### Script:
```bash
# From repository root
# For nolr (no LRT regularization) batches:
python claudefile/pool_kappa_and_gamma_from_nolr_batches.py \
    --batch_pattern "/path/to/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt" \
    --output_dir /path/to/data

# For standard batches:
python claudefile/pool_kappa_and_gamma_from_batches.py \
    --batch_pattern "/path/to/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /path/to/data
```

**Script locations:**
- [`claudefile/pool_kappa_and_gamma_from_nolr_batches.py`](../../../../claudefile/pool_kappa_and_gamma_from_nolr_batches.py) - For nolr (unregularized) training batches
- [`claudefile/pool_kappa_and_gamma_from_batches.py`](../../../../claudefile/pool_kappa_and_gamma_from_batches.py) - For standard training batches

### What it does:
- Loads kappa and gamma from all batch checkpoints
- Computes mean across batches (pooled estimates)
- Saves to `pooled_kappa_gamma.pt` (or `pooled_kappa_gamma_nolr.pt` for nolr)

### Output:
- `pooled_kappa_gamma.pt` - Contains pooled kappa (scalar) and gamma (P × K matrix) for use in fixed-gamma/kappa prediction

---

## Step 5: Predict with Master (Fixed Phi, Gamma, Kappa)

Run predictions using the master checkpoint with fixed phi, gamma, and kappa. Only lambda (individual-specific signature loadings) is learned per batch—ensuring no information leakage from test data into population parameters.

### Script:
```bash
# From repository root
python claudefile/run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py \
    --trained_model_path /path/to/master_for_fitting_pooled_all_data.pt \
    --pooled_gamma_kappa_path /path/to/data/pooled_kappa_gamma.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions \
    --max_batches 40 \
    --num_epochs 200
```

**Script location:** [`claudefile/run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py`](../../../../claudefile/run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py)  
**Model file:** [`claudefile/aws_offsetmaster/clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py`](../../../../claudefile/aws_offsetmaster/clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py)

### What it does:
- Loads master checkpoint (pooled phi + initial_psi)
- Loads pooled gamma and kappa from Step 4
- **Automatically loads E matrix** from `data_dir/E_enrollment_full.pt` (enrollment data for prediction)
- Uses fixed phi, gamma, and kappa—only learns lambda per batch
- Generates predictions for all patients
- Saves prediction results

### Required files in `--data_dir`:
- `Y_tensor.pt` - Disease outcomes
- `E_enrollment_full.pt` - Enrollment matrix (automatically loaded)
- `G_matrix.pt` - Genetic variants
- `model_essentials.pt` - Model metadata
- `reference_trajectories.pt` - Signature reference trajectories

---

## Complete Workflow Example

```bash
# Step 1: Preprocessing (in notebook or script)
# Creates: prevalence_t, initial_clusters_400k.pt, initial_psi_400k.pt, reference_trajectories.pt

# Step 2: Batch training (no LR regularization on gamma)
python claudefile/run_aladyn_batch_vector_e_censor_nolor.py \
    --start_index 0 \
    --end_index 10000 \
    --data_dir /path/to/data \
    --output_dir /path/to/output

# Step 3: Create master checkpoint
python claudefile/create_master_checkpoints.py \
    --retrospective_pattern "/path/to/output/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt" \
    --output_dir /path/to/data

# Step 4: Pool gamma and kappa from training batches (using nolr pooling script for nolr batches)
python claudefile/pool_kappa_and_gamma_from_nolr_batches.py \
    --batch_pattern "/path/to/output/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt" \
    --output_dir /path/to/data

# Step 5: Predict (fixed phi, gamma, kappa; only lambda learned)
python claudefile/run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py \
    --trained_model_path /path/to/data/master_for_fitting_pooled_all_data.pt \
    --pooled_gamma_kappa_path /path/to/data/pooled_kappa_gamma.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions \
    --max_batches 40
```

---

## File Locations

### Preprocessing Files (in `preprocessing/`):
- [`preprocessing_utils.py`](./preprocessing_utils.py) - Standalone utility functions
- [`create_preprocessing_files.ipynb`](./create_preprocessing_files.ipynb) - Interactive notebook
- [`SIMPLE_EXAMPLE.py`](./SIMPLE_EXAMPLE.py) - Simple usage example

### Training Scripts (in `claudefile/`):
- [`run_aladyn_batch_vector_e_censor_nolor.py`](../../../../claudefile/run_aladyn_batch_vector_e_censor_nolor.py) - Batch training script (vectorized, corrected E, no LR regularization on gamma) - **Used for fixing gamma and kappa in final workflow**
- [`create_master_checkpoints.py`](../../../../claudefile/create_master_checkpoints.py) - Master checkpoint creation
- [`pool_kappa_and_gamma_from_batches.py`](../../../../claudefile/pool_kappa_and_gamma_from_batches.py) - Pool gamma and kappa from standard training batches
- [`pool_kappa_and_gamma_from_nolr_batches.py`](../../../../claudefile/pool_kappa_and_gamma_from_nolr_batches.py) - Pool gamma and kappa from nolr (unregularized) batches

### Prediction Scripts (in `claudefile/`):
- [`run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py`](../../../../claudefile/run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py) - Predict with fixed phi, gamma, kappa (only lambda learned)

### Model Files:
- [`pyScripts_forPublish/clust_huge_amp_vectorized.py`](../../../../pyScripts_forPublish/clust_huge_amp_vectorized.py) - Discovery mode (learns phi, gamma, kappa, lambda)
- [`claudefile/aws_offsetmaster/clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py`](../../../../claudefile/aws_offsetmaster/clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py) - Prediction mode (fixed phi, gamma, kappa; learns lambda only)

### Related Documentation:
- [Reviewer Response Analyses](../README.md) - All analysis notebooks and preprocessing utilities
- [Main Repository README](../../../../README.md) - Complete overview and installation

---

## Notes

- **Preprocessing functions are standalone** - No model initialization needed
- **Clusters are deterministic** - Same `random_state=42` produces identical clusters
- **Master checkpoint pools phi** - Mean across all batches for stability
- **Fixed phi, gamma, kappa at prediction** - Population parameters (phi, gamma, kappa) are fixed; only lambda (individual signature loadings) is learned per batch. This ensures complete separation between training and test data and enables fast predictions.

## Related Analyses

After completing the workflow, see the [Reviewer Response Analyses](../README.md) for:
- Performance evaluation notebooks
- Clinical utility analyses
- Model validation studies
- Comparison with established risk scores
- All interactive analyses addressing reviewer questions

