# Aladynoulli: A Bayesian Survival Model for Disease Trajectory Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A comprehensive Bayesian survival model for predicting disease trajectories using genetic and clinical data. This repository contains the complete implementation of the Aladynoulli model as described in our preprint.

**Preprint**: [medRxiv](https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v1)

**Reviewer Response Analyses** (comprehensive validation and analysis):
See [`pyScripts/new_oct_revision/new_notebooks/reviewer_responses/README.md`](pyScripts/new_oct_revision/new_notebooks/reviewer_responses/README.md) for a complete guide to all interactive analyses addressing reviewer questions, including clinical utility, lifetime risk, AUC comparisons, model validity, and more.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/surbut/aladynoulli2.git
cd aladynoulli2
pip install -r requirements.txt
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🔬 Overview

Aladynoulli is a Bayesian survival model that predicts disease trajectories by modeling:

- **Disease signatures**: Latent disease states that capture shared patterns across diseases
- **Genetic effects**: Individual-specific genetic contributions to disease risk
- **Temporal dynamics**: Time-varying disease probabilities using Gaussian processes
- **Censoring**: Proper handling of incomplete follow-up data

### Key Features

- ✅ **Scalable**: Handles large-scale genetic and clinical datasets
- ✅ **Flexible**: Supports both discovery and prediction modes
- ✅ **Robust**: Proper Bayesian uncertainty quantification
- ✅ **Fast**: GPU-accelerated training and inference
- ✅ **Reproducible**: Complete code and data processing pipelines

## 🏗️ Model Architecture

The model consists of several key components:

### Core Components

1. **Signature States (K)**: Latent disease signatures that capture shared patterns
2. **Genetic Effects (γ)**: Individual-specific genetic contributions
3. **Temporal Dynamics (λ)**: Time-varying signature proportions using GPs
4. **Disease Probabilities (φ)**: Signature-specific disease probabilities
5. **Censoring Matrix (E)**: Event times and censoring information

### Mathematical Framework

The model predicts disease probability at time t as:

```
π_i,d,t = κ × Σ_k θ_i,k,t × φ_k,d,t
```

Where:
- `θ_i,k,t` = softmax(λ_i,k,t) (signature proportions)
- `λ_i,k,t` ~ GP(μ_k + G_i γ_k, K_λ) (temporal dynamics)
- `φ_k,d,t` = sigmoid(ψ_k,d + GP(μ_φ, K_φ)) (disease probabilities)

## 📦 Installation

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB free space
- **GPU**: Optional but recommended (NVIDIA CUDA)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/surbut/aladynoulli2.git
   cd aladynoulli2
   ```

2. **Create virtual environment**:
   ```bash
   conda create -n aladynoulli python=3.9
   conda activate aladynoulli
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```



For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md) and [Documentation.md](pyScripts_forPublish/DOCUMENTATION.md)

## 💻 Usage

### Complete Workflow

The Aladynoulli workflow consists of 4 main steps: **Preprocessing → Batch Training → Master Checkpoint → Prediction**

For detailed step-by-step instructions, see [`pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md`](pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md).

#### Step 1: Preprocessing

Create initialization files (prevalence, clusters, psi, reference trajectories):

```python
# Option A: Use the interactive notebook
jupyter notebook pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/create_preprocessing_files.ipynb

# Option B: Use standalone functions
from pyScripts.new_oct_revision.new_notebooks.reviewer_responses.preprocessing.preprocessing_utils import (
    compute_smoothed_prevalence,
    create_initial_clusters_and_psi,
    create_reference_trajectories
)

# Compute smoothed prevalence
prevalence_t = compute_smoothed_prevalence(Y, window_size=5, smooth_on_logit=True)

# Create initial clusters and psi
clusters, psi = create_initial_clusters_and_psi(Y=Y, K=20, random_state=42)

# Create reference trajectories
signature_refs, healthy_ref = create_reference_trajectories(Y, clusters, K=20)
```

**Output files:**
- `prevalence_t` - Smoothed disease prevalence (D × T)
- `initial_clusters_400k.pt` - Disease-to-signature assignments
- `initial_psi_400k.pt` - Initial signature-disease parameters (K × D)
- `reference_trajectories.pt` - Signature reference trajectories

#### Step 2: Batch Training

Train the model on batches with full enrollment data:

```bash
python claudefile/run_aladyn_batch.py \
    --data_dir /path/to/data \
    --output_dir /path/to/batch_output \
    --start_index 0 \
    --end_index 10000 \
    --num_epochs 200
```

**What it does:**
- Trains model on batches using full enrollment E matrix
- Uses [`pyScripts_forPublish/clust_huge_amp.py`](pyScripts_forPublish/clust_huge_amp.py) for discovery mode
- Saves checkpoints: `enrollment_model_W0.0001_batch_*_*.pt`
- Each checkpoint contains learned `phi` parameters (K × D × T)

**Script location:** [`claudefile/run_aladyn_batch.py`](claudefile/run_aladyn_batch.py)

#### Step 3: Create Master Checkpoint

Pool phi from all batches and create master checkpoint:

**Option A: Use the script** (recommended)
```bash
python claudefile/create_master_checkpoints.py \
    --data_dir /path/to/data \
    --retrospective_pattern "/path/to/batch_output/enrollment_model_W0.0001_batch_*_*.pt" \
    --enrollment_pattern "/path/to/batch_output/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /path/to/data
```

**Option B: Use interactive notebook**
```python
import sys
sys.path.append('/path/to/aladynoulli2/claudefile/')
from create_master_checkpoints import pool_phi_from_batches, create_master_checkpoint
import torch
import numpy as np

# Load initial_psi
initial_psi = torch.load(data_dir + 'initial_psi_400k.pt', weights_only=False)
if torch.is_tensor(initial_psi):
    initial_psi = initial_psi.cpu().numpy()

# Pool phi from batches
phi_pooled = pool_phi_from_batches("/path/to/enrollment_model_W0.0001_batch_*_*.pt")

# Create master checkpoint
create_master_checkpoint(phi_pooled, initial_psi, output_path, description="...")
```

**What it does:**
- Loads phi from all batch checkpoints
- Pools phi (mean across batches) for stability
- Combines with `initial_psi_400k.pt`
- Creates master checkpoint: `master_for_fitting_pooled_all_data.pt`

**Script location:** [`claudefile/create_master_checkpoints.py`](claudefile/create_master_checkpoints.py)  
**Example notebook:** [`misc/evalmodel/lifetime.ipynb`](misc/evalmodel/lifetime.ipynb)

#### Step 4: Predict with Master Checkpoint

Run predictions using fixed phi from master checkpoint:

```bash
python claudefile/version_from_ec2/run_aladyn_predict_with_master.py \
    --trained_model_path /path/to/master_for_fitting_pooled_all_data.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions \
    --batch_size 10000 \
    --num_epochs 200
```

**What it does:**
- Loads master checkpoint (pooled phi + initial_psi)
- Automatically loads `E_enrollment_full.pt` (full enrollment matrix)
- Uses [`pyScripts_forPublish/clust_huge_amp_fixedPhi.py`](pyScripts_forPublish/clust_huge_amp_fixedPhi.py) for fixed-phi predictions
- Only estimates lambda (genetic effects) per batch
- Generates predictions (pi tensor) for all patients

**Script location:** [`claudefile/version_from_ec2/run_aladyn_predict_with_master.py`](claudefile/version_from_ec2/run_aladyn_predict_with_master.py)

**Required files in `--data_dir`:**
- `Y_tensor.pt` - Disease outcomes (N × D × T)
- `E_enrollment_full.pt` - Enrollment matrix (N × T) - **automatically loaded**
- `G_matrix.pt` - Genetic variants (N × P)
- `model_essentials.pt` - Model metadata
- `reference_trajectories.pt` - Signature reference trajectories
- `initial_psi_400k.pt` - Initial psi parameters

### Interactive Notebooks

For interactive exploration and development:

**Discovery Mode** (full model training):
```bash
jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery_noweights.ipynb
```

**Prediction Mode** (fixed phi):
```bash
jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb
```



### Programmatic Usage

Direct Python API:

```python
# Discovery mode (learns phi)
from pyScripts_forPublish.clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=Y.shape[0], 
    D=Y.shape[1], 
    T=Y.shape[2], 
    K=20,
    P=G_with_sex.shape[1],
    G=G_with_sex, 
    Y=Y,
    prevalence_t=prevalence_t,
    signature_references=signature_refs,
    healthy_reference=True,
    disease_names=disease_names
)

# Train model
history = model.fit(E, num_epochs=200)

# Make predictions
pi, theta, phi = model.forward()
```

```python
# Prediction mode (fixed phi)
from pyScripts_forPublish.clust_huge_amp_fixedPhi import AladynSurvivalFixedPhi

model = AladynSurvivalFixedPhi(
    N=Y.shape[0],
    D=Y.shape[1],
    T=Y.shape[2],
    K=20,
    P=G_with_sex.shape[1],
    G=G_with_sex,
    Y=Y,
    pretrained_phi=phi_pooled,  # Fixed phi from master checkpoint
    pretrained_psi=psi_pooled,  # Fixed psi from master checkpoint
    signature_references=signature_refs,
    healthy_reference=True,
    disease_names=disease_names
)


# Train (only estimates lambda)
history = model.fit(E, num_epochs=200)

# Make predictions
pi, theta, phi = model.forward()
```

## 📚 Documentation

### Core Model Files

- **Discovery Model**: [`pyScripts_forPublish/clust_huge_amp.py`](pyScripts_forPublish/clust_huge_amp.py) - Full model that learns phi and psi
- **Prediction Model**: [`pyScripts_forPublish/clust_huge_amp_fixedPhi.py`](pyScripts_forPublish/clust_huge_amp_fixedPhi.py) - Fixed-phi model for fast predictions
- **Discovery Notebook**: [`pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb`](pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb)
- **Prediction Notebook**: [`pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb`](pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb)

### Workflow Scripts

- **Preprocessing**: [`pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py`](pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py)
- **Batch Training**: [`claudefile/run_aladyn_batch.py`](claudefile/run_aladyn_batch.py)
- **Master Checkpoint**: [`claudefile/create_master_checkpoints.py`](claudefile/create_master_checkpoints.py)
- **Prediction**: [`claudefile/version_from_ec2/run_aladyn_predict_with_master.py`](claudefile/version_from_ec2/run_aladyn_predict_with_master.py)

### Workflow Documentation

- **Complete Workflow Guide**: [`pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md`](pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md)

### Reviewer Response Analyses

- **📊 Reviewer Response Navigation Hub**: [`pyScripts/new_oct_revision/new_notebooks/reviewer_responses/README.md`](pyScripts/new_oct_revision/new_notebooks/reviewer_responses/README.md) - Complete guide to all interactive analyses addressing reviewer questions, including:
  - Clinical utility and dynamic risk updating
  - Lifetime risk predictions
  - AUC comparisons with established scores
  - Model validity and learning analyses
  - Biological plausibility studies
  - Age-stratified performance
  - And more...

### Additional Tools

- **Streamlit App**: [`pyScripts_forPublish/patient_timeline_app`](pyScripts_forPublish/patient_timeline_app)
- **AWS Scripts**: [`pyScripts_forPublish/submit_script_aws_fixedph_40_70.py`](pyScripts_forPublish/submit_script_aws_fixedph_40_70.py)
- **Demo Script**: [`pyScripts_forPublish/newsm_3_71.ipynb`](pyScripts_forPublish/newsm_3_71.ipynb)

### Data Requirements

**Required input files** (created during preprocessing):
- `Y_tensor.pt`: Disease outcome tensor (N × D × T)
- `E_matrix.pt`: Censoring matrix (N × D) - for batch training
- `E_enrollment_full.pt`: Full enrollment matrix (N × T) - for predictions
- `G_matrix.pt`: Genetic data matrix (N × P)
- `model_essentials.pt`: Model configuration (disease names, etc.)
- `reference_trajectories.pt`: Signature reference trajectories (created in Step 1)
- `initial_psi_400k.pt`: Initial psi parameters (created in Step 1)
- `initial_clusters_400k.pt`: Initial cluster assignments (created in Step 1)

**Generated during workflow:**
- Batch checkpoints: `enrollment_model_W0.0001_batch_*_*.pt` (Step 2)
- Master checkpoint: `master_for_fitting_pooled_all_data.pt` (Step 3)
- Predictions: `pi_full_400k.pt` (Step 4)

See [WORKFLOW.md](pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md) for detailed file descriptions.

## ⏱️ Performance

### Typical Runtime (10,000 individuals, 50 diseases, 50 time points)

| Configuration | Training Time | Memory Usage |
|---------------|---------------|--------------|
| CPU (Apple M4 Max) | ~70 minutes | ~12GB  per 10000 people

### Scalability

- **Individuals**: Tested up to 400,000
- **Diseases**: Tested 350
- **Time Points**: Tested 52 (ages 30-81)
- **Genetic Features**: Tested with 36 PRS and 10 PCs


This will:
1. Generate synthetic data
2. Train a small model
3. Generate predictions
4. Create visualizations
5. Report timing information

## 📊 Results

The model has been validated on:
- UK Biobank data (500,000+ individuals)
- Multiple disease categories
- Cross-validation studies
- External validation cohorts

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{urbut2024aladynoulli,
  title={Aladynoulli: A Bayesian Survival Model for Disease Trajectory Prediction},
  author={Urbut, Sarah and others},
  journal={medRxiv},
  year={2024},
  doi={10.1101/2024.09.29.24314557}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Sarah Urbut
- **Email**: surbut@mgh.harvard.edu
- **Institution**: Massachusetts General Hospital

## 🙏 Acknowledgments

We thank the UK Biobank participants and the research community for making this work possible.

---

**Note**: This software is provided for research purposes. Please ensure you have appropriate data use agreements and ethical approvals before using with real patient data.
