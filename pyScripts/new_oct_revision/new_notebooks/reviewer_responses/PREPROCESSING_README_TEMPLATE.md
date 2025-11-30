# Preprocessing Pipeline

This directory documents the actual workflow used to create initialization files for the Aladynoulli model, based on `with_bigdata.ipynb` and `minimalreprobigforinit.ipynb`.

## Overview

The preprocessing workflow creates several files needed to run the model:
1. **Prevalence** (`prevalence_t`) - Disease prevalence over time (computed from Y)
2. **Initial Clusters** (`initial_clusters_400k.pt`) - Disease-to-signature assignments (created by model initialization)
3. **Initial Psi** (`initial_psi_400k.pt`) - Initial signature-disease association parameters (created by model initialization)
4. **Reference Trajectories** (`reference_trajectories.pt`) - Population-level signature trajectories (computed from Y and clusters)
5. **Model Essentials** (`model_essentials.pt`) - Disease names, metadata, etc. (saved after model creation)
6. **IPW Weights** (optional) - Inverse probability weights for selection bias correction

---

## Scripts

### 1. `compute_prevalence.py` / `compute_smoothed_prevalence()`
**Purpose**: Calculate disease prevalence over time with Gaussian smoothing

**Location**: Function `compute_smoothed_prevalence()` is defined in the model files (e.g., `clust_huge_amp.py`)

**Inputs**:
- `Y`: Disease outcome tensor (N × D × T)
- `window_size`: Gaussian smoothing window size (default: 5)

**Outputs**:
- `prevalence_t`: Prevalence matrix (D × T)

**Usage**:
```python
from clust_huge_amp import compute_smoothed_prevalence

prevalence_t = compute_smoothed_prevalence(Y=Y, window_size=5)
```

**Method**:
1. Compute mean prevalence across individuals: `Y.mean(axis=0)` → (D × T)
2. Apply Gaussian smoothing to each disease trajectory: `gaussian_filter1d(prevalence_t[d, :], sigma=window_size)`

**Note**: For IPW-weighted prevalence, use `weightedprev.py` functions separately

---

### 2. Clusters and Psi Creation (via Model Initialization)
**Purpose**: Create initial clusters and psi by initializing the model

**Key Insight**: Clusters and psi are **created automatically** when you call `model.initialize_params()` with `psi_config`. This is NOT a separate preprocessing step - it's part of model initialization.

**Inputs**:
- `Y`: Disease outcome tensor (N × D × T)
- `prevalence_t`: Prevalence matrix (D × T)
- `K`: Number of signatures
- `psi_config`: Configuration dict

**Outputs**:
- `initial_clusters_400k.pt`: Array of cluster assignments [D] (0 to K-1)
- `initial_psi_400k.pt`: Psi matrix (K × D)

**Method** (inside `model.initialize_params()`):
1. Compute disease-disease correlation matrix from Y
2. Convert to similarity matrix: `(corr + 1) / 2`
3. Apply spectral clustering: `SpectralClustering(n_clusters=K, random_state=42)`
4. Assign diseases to clusters: `model.clusters = spectral.labels_`
5. Initialize psi based on clusters:
   - If `clusters[d] == k`: `psi[k, d] = psi_config['in_cluster'] + noise_in`
   - If `clusters[d] != k`: `psi[k, d] = psi_config['out_cluster'] + noise_out`
   - Healthy reference: `psi[K, :] = -5.0 + noise`

**Usage**:
```python
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

# Create model
model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=N, D=D, T=T, K=K, P=P, G=G, Y=Y, prevalence_t=prevalence_t
)

# Initialize with psi_config (creates clusters and psi)
psi_config = {
    'in_cluster': 1,      # High value for diseases in cluster
    'out_cluster': -2,    # Low value for diseases outside cluster
    'noise_in': 0.1,      # Noise for in-cluster
    'noise_out': 0.01     # Noise for out-cluster
}
model.initialize_params(psi_config=psi_config)

# Extract and save
torch.save(model.clusters, 'initial_clusters_400k.pt')
torch.save(model.psi.detach(), 'initial_psi_400k.pt')
```

**Important**: The clustering happens INSIDE the model's `initialize_params()` method, not in a separate script!

---

### 3. `create_reference_trajectories.py`
**Purpose**: Calculate population-level signature reference trajectories

**Inputs**:
- `Y`: Disease outcome tensor (N × D × T)
- `clusters`: Disease cluster assignments [D]
- `K`: Number of signatures
- `healthy_prop`: Proportion of healthy state (default: 0)

**Outputs**:
- `reference_trajectories.pt`: Dict with:
  - `signature_refs`: Reference trajectories (K × T) on logit scale
  - `healthy_ref`: Healthy reference trajectory (T) on logit scale

**Method**:
1. For each signature k, sum disease counts for diseases in cluster k
2. Normalize to proportions
3. Convert to logit scale
4. Apply LOWESS smoothing
5. Create healthy reference (constant at logit(healthy_prop))

**Usage**:
```python
from create_reference_trajectories import create_reference_trajectories

signature_refs, healthy_ref = create_reference_trajectories(
    Y, clusters, K=20, healthy_prop=0, frac=0.3
)

torch.save({
    'signature_refs': signature_refs,
    'healthy_ref': healthy_ref
}, 'reference_trajectories.pt')
```

---

### 4. `create_model_essentials.py` / `save_model_essentials()`
**Purpose**: Create model_essentials.pt with metadata

**Inputs**:
- `disease_names`: List of disease names [D]
- `prevalence_t`: Prevalence matrix (D × T)
- Other metadata (phecode mappings, etc.)

**Outputs**:
- `model_essentials.pt`: Dict with:
  - `disease_names`: List of disease names
  - `prevalence_t`: Prevalence matrix
  - `phecode_mappings`: (optional) Phecode to ICD mappings
  - Other metadata

**Usage**:
```python
from create_model_essentials import create_model_essentials

essentials = create_model_essentials(
    disease_names=disease_names,
    prevalence_t=prevalence_t,
    phecode_mappings=phecode_mappings
)

torch.save(essentials, 'model_essentials.pt')
```

---

### 5. `compute_ipw_weights.py` (if separate)
**Purpose**: Calculate inverse probability weights for selection bias correction

**Inputs**:
- Demographics data (age, sex, ethnicity, etc.)
- Participation indicators

**Outputs**:
- `ipw_weights.pt`: IPW weights [N]
- `ipw_weights.csv`: CSV version with IDs

**Method**:
- Lasso regression to predict participation
- Inverse of predicted probabilities as weights
- Truncation/calibration as needed

---

## Complete Pipeline (Actual Workflow)

**Order of operations** (based on `with_bigdata.ipynb` and `minimalreprobigforinit.ipynb`):

1. **Load data**: Y, E, G tensors
2. **Compute prevalence**: Using `compute_smoothed_prevalence()` function
3. **Create model**: Initialize model with prevalence
4. **Initialize model**: This creates clusters and psi automatically
5. **Save clusters and psi**: Extract from initialized model
6. **Create reference trajectories**: From Y and saved clusters
7. **Save model essentials**: Disease names, prevalence, etc.

**Actual workflow** (from notebooks):
```python
import torch
import numpy as np
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
from clust_huge_amp import compute_smoothed_prevalence  # Function in model file

# 1. Load data
Y = torch.load('Y_tensor.pt')
E = torch.load('E_matrix.pt')
G = torch.load('G_matrix.pt')

# 2. Compute prevalence
prevalence_t = compute_smoothed_prevalence(Y=Y, window_size=5)

# 3. Set up dimensions
K = 20
T = Y.shape[2]
N = Y.shape[0]
D = Y.shape[1]
P = G.shape[1]

# 4. Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 5. Create model
model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=N, D=D, T=T, K=K, P=P,
    G=G, Y=Y,
    prevalence_t=prevalence_t
)

# 6. Initialize with psi_config (this creates clusters via spectral clustering)
psi_config = {
    'in_cluster': 1,      # High value for diseases in cluster
    'out_cluster': -2,    # Low value for diseases outside cluster
    'noise_in': 0.1,      # Noise for in-cluster
    'noise_out': 0.01     # Noise for out-cluster
}
model.initialize_params(psi_config=psi_config)

# 7. Save clusters and psi from initialized model
torch.save(model.clusters, 'initial_clusters_400k.pt')
torch.save(model.psi.detach(), 'initial_psi_400k.pt')

# 8. Create reference trajectories from Y and clusters
def create_reference_trajectories(Y, initial_clusters, K, healthy_prop=0, frac=0.3):
    """Create reference trajectories using LOWESS smoothing on logit scale"""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.special import logit
    
    T = Y.shape[2]
    Y_counts = Y.sum(dim=0)  # D x T
    signature_props = torch.zeros(K, T)
    total_counts = Y_counts.sum(dim=0) + 1e-8
    
    for k in range(K):
        cluster_mask = (initial_clusters == k)
        signature_props[k] = Y_counts[cluster_mask].sum(dim=0) / total_counts
    
    # Normalize and clamp
    signature_props = torch.clamp(signature_props, min=1e-8, max=1-1e-8)
    signature_props = signature_props / signature_props.sum(dim=0, keepdim=True)
    signature_props *= (1 - healthy_prop)
    
    # Convert to logit and smooth
    logit_props = torch.tensor(logit(signature_props.numpy()))
    signature_refs = torch.zeros_like(logit_props)
    
    times = np.arange(T)
    for k in range(K):
        smoothed = lowess(
            logit_props[k].numpy(), 
            times,
            frac=frac,
            it=3,
            delta=0.0,
            return_sorted=False
        )
        signature_refs[k] = torch.tensor(smoothed)
    
    healthy_ref = torch.ones(T) * logit(torch.tensor(healthy_prop))
    return signature_refs, healthy_ref

signature_refs, healthy_ref = create_reference_trajectories(Y, model.clusters, K=20)
torch.save({
    'signature_refs': signature_refs,
    'healthy_ref': healthy_ref
}, 'reference_trajectories.pt')

# 9. Save model essentials
def save_model_essentials(Y, E, prevalence_t, G, P, K, disease_names, model, base_path='./'):
    essentials = {
        'prevalence_t': prevalence_t,
        'P': P,
        'K': K,
        'disease_names': disease_names,
        'clusters': model.clusters,
        'psi': model.psi.detach(),
        'model_state': model.state_dict()
    }
    torch.save(essentials, base_path + 'model_essentials.pt')

save_model_essentials(Y, E, prevalence_t, G, P, K, disease_names, model)
```

---

## Dependencies

All preprocessing scripts require:
- `torch` (PyTorch)
- `numpy`
- `scipy`
- `sklearn` (for spectral clustering)
- `pandas` (for IPW weights)
- `statsmodels` (for LOWESS smoothing in reference trajectories)

---

## Notes

- **Prevalence smoothing**: Uses Gaussian filter with `window_size=5` (sigma parameter)
- **Clustering**: Spectral clustering with `n_init=10` and `random_state=42` for reproducibility (inside `model.initialize_params()`)
- **Psi initialization**: Uses `psi_config` dict with `in_cluster=1`, `out_cluster=-2`, `noise_in=0.1`, `noise_out=0.01`
- **Reference trajectories**: LOWESS smoothing with `frac=0.3` (30% of data points) on logit scale
- **Healthy reference**: Default value is `-5.0` on logit scale (very low probability)
- **Random seeds**: Set `torch.manual_seed(42)` and `np.random.seed(42)` before model initialization for reproducibility

---

## References

- Prevalence calculation: See `compute_prevalence.py` (from `weightedprev.py`)
- Clustering: Spectral clustering on disease-disease correlation matrix
- Reference trajectories: LOWESS smoothing on logit-transformed proportions
- Model initialization: See `source/clust_huge_amp.py` `initialize_params()` method

