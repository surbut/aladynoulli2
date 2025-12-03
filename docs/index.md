# Aladynoulli Documentation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**A Bayesian Survival Model for Disease Trajectory Prediction**

[Preprint](https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v1) • [GitHub Repository](../) • [Quick Start](#-quick-start)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Model Architecture](#-model-architecture)
- [Complete Workflow](#-complete-workflow)
- [Reviewer Response Analyses](#-reviewer-response-analyses)
- [Documentation](#-documentation)
- [Performance & Scalability](#-performance--scalability)
- [Citation](#-citation)

---

## 🔬 Overview

**Aladynoulli** is a comprehensive Bayesian survival model that predicts disease trajectories by integrating genetic and clinical data. The model captures:

- **Disease Signatures**: Latent disease states that capture shared patterns across diseases
- **Genetic Effects**: Individual-specific genetic contributions to disease risk  
- **Temporal Dynamics**: Time-varying disease probabilities using Gaussian processes
- **Censoring**: Proper handling of incomplete follow-up data

### Key Features

| Feature | Description |
|---------|-------------|
| ✅ **Scalable** | Handles large-scale genetic and clinical datasets (400K+ individuals) |
| ✅ **Flexible** | Supports both discovery and prediction modes |
| ✅ **Robust** | Proper Bayesian uncertainty quantification |
| ✅ **Fast** | GPU-accelerated training and inference |
| ✅ **Reproducible** | Complete code and data processing pipelines |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/surbut/aladynoulli2.git
cd aladynoulli2
pip install -r requirements.txt
```

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB free space
- **GPU**: Optional but recommended (NVIDIA CUDA)

For detailed installation instructions, see the [main README](../README.md) or [INSTALLATION.md](../INSTALLATION.md).

---

## 🏗️ Model Architecture

### Core Components

1. **Signature States (K)**: Latent disease signatures that capture shared patterns
2. **Genetic Effects (γ)**: Individual-specific genetic contributions
3. **Temporal Dynamics (λ)**: Time-varying signature proportions using GPs
4. **Disease Probabilities (φ)**: Signature-specific disease probabilities
5. **Censoring Matrix (E)**: Event times and censoring information

### Mathematical Framework

The model predicts disease probability at time `t` as:

```
π_i,d,t = κ × Σ_k θ_i,k,t × φ_k,d,t
```

Where:
- `θ_i,k,t` = softmax(λ_i,k,t) (signature proportions)
- `λ_i,k,t` ~ GP(μ_k + G_i γ_k, K_λ) (temporal dynamics)
- `φ_k,d,t` = sigmoid(ψ_k,d + GP(μ_φ, K_φ)) (disease probabilities)

---

## 💻 Complete Workflow

The Aladynoulli workflow consists of **4 main steps**:

```
Preprocessing → Batch Training → Master Checkpoint → Prediction
```

### Step 1: Preprocessing

Create initialization files (prevalence, clusters, psi, reference trajectories):

**Option A: Interactive Notebook**
```bash
jupyter notebook ../pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/create_preprocessing_files.ipynb
```

**Option B: Standalone Functions**
```python
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

### Step 2: Batch Training

Train the model on batches with full enrollment data:

```bash
python ../claudefile/run_aladyn_batch.py \
    --data_dir /path/to/data \
    --output_dir /path/to/batch_output \
    --start_index 0 \
    --end_index 10000 \
    --num_epochs 200
```

**What it does:**
- Trains model on batches using full enrollment E matrix
- Uses `clust_huge_amp.py` for discovery mode
- Saves checkpoints: `enrollment_model_W0.0001_batch_*_*.pt`
- Each checkpoint contains learned `phi` parameters (K × D × T)

### Step 3: Create Master Checkpoint

Pool phi from all batches and create master checkpoint:

```bash
python ../claudefile/create_master_checkpoints.py \
    --data_dir /path/to/data \
    --retrospective_pattern "/path/to/batch_output/enrollment_model_W0.0001_batch_*_*.pt" \
    --enrollment_pattern "/path/to/batch_output/enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir /path/to/data
```

**What it does:**
- Loads phi from all batch checkpoints
- Pools phi (mean across batches) for stability
- Combines with `initial_psi_400k.pt`
- Creates master checkpoint: `master_for_fitting_pooled_all_data.pt`

### Step 4: Predict with Master Checkpoint

Run predictions using fixed phi from master checkpoint:

```bash
python ../claudefile/version_from_ec2/run_aladyn_predict_with_master.py \
    --trained_model_path /path/to/master_for_fitting_pooled_all_data.pt \
    --data_dir /path/to/data \
    --output_dir /path/to/predictions \
    --batch_size 10000 \
    --num_epochs 200
```

**What it does:**
- Loads master checkpoint (pooled phi + initial_psi)
- Automatically loads `E_enrollment_full.pt` (full enrollment matrix)
- Uses `clust_huge_amp_fixedPhi.py` for fixed-phi predictions
- Only estimates lambda (genetic effects) per batch
- Generates predictions (pi tensor) for all patients

For detailed step-by-step instructions, see the [Complete Workflow Guide](../pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md).

---

## 📊 Reviewer Response Analyses

Comprehensive interactive analyses addressing reviewer questions and model validation:

### 📚 Navigation Hub

**[Reviewer Response README](reviewer_responses/README.html)** - Complete guide to all interactive analyses

### 🔬 Analysis Categories

#### **Referee #1 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Clinical Utility** | Dynamic risk updating and clinical decision-making | [R1_Clinical_Utility_Dynamic_Risk_Updating.html](reviewer_responses/notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.html) |
| **Lifetime Risk** | Long-term risk predictions across the lifespan | [R1_Q2_Lifetime_Risk.html](reviewer_responses/notebooks/R1/R1_Q2_Lifetime_Risk.html) |
| **AUC Comparisons** | Performance vs. established clinical risk scores | [R1_Q9_AUC_Comparisons.html](reviewer_responses/notebooks/R1/R1_Q9_AUC_Comparisons.html) |
| **Age-Stratified** | Performance across different age groups | [R1_Q10_Age_Specific.html](reviewer_responses/notebooks/R1/R1_Q10_Age_Specific.html) |
| **Heritability** | Genetic architecture and heritability estimates | [R1_Q7_Heritability.html](reviewer_responses/notebooks/R1/R1_Q7_Heritability.html) |
| **GWAS Validation** | Genome-wide association studies on signatures | [R1_Genetic_Validation_GWAS.html](reviewer_responses/notebooks/R1/R1_Genetic_Validation_GWAS.html) |
| **Gene-Based RVAS** | Rare variant association studies on signatures | [R1_Genetic_Validation_Gene_Based_RVAS.html](reviewer_responses/notebooks/R1/R1_Genetic_Validation_Gene_Based_RVAS.html) |
| **Biological Plausibility** | CHIP analysis and biological validation | [R1_Biological_Plausibility_CHIP.html](reviewer_responses/notebooks/R1/R1_Biological_Plausibility_CHIP.html) |
| **LOO Validation** | Leave-one-out cross-validation robustness | [R1_Robustness_LOO_Validation.html](reviewer_responses/notebooks/R1/R1_Robustness_LOO_Validation.html) |
| **Selection Bias** | Assessment of selection bias and participation | [R1_Q1_Selection_Bias.html](reviewer_responses/notebooks/R1/R1_Q1_Selection_Bias.html) |
| **Clinical Meaning** | ICD vs PheCode comparison and clinical interpretation | [R1_Q3_Clinical_Meaning.html](reviewer_responses/notebooks/R1/R1_Q3_Clinical_Meaning.html) |
| **ICD vs PheCode** | Detailed comparison of coding systems | [R1_Q3_ICD_vs_PheCode_Comparison.html](reviewer_responses/notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.html) |
| **Competing Risks** | Multi-disease patterns and competing risks | [R1_Multi_Disease_Patterns_Competing_Risks.html](reviewer_responses/notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.html) |

#### **Referee #2 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Temporal Leakage** | Washout analysis and temporal leakage assessment | [R2_Temporal_Leakage.html](reviewer_responses/notebooks/R2/R2_Temporal_Leakage.html) |
| **Washout Continued** | Comprehensive washout analysis across timepoints | [R2_Washout_Continued.html](reviewer_responses/notebooks/R2/R2_Washout_Continued.html) |
| **Washout Clean** | Clean version of washout analysis | [R2_Washout_Continued_clean.html](reviewer_responses/notebooks/R2/R2_Washout_Continued_clean.html) |
| **Model Validity** | Model learning and validity assessment | [R2_R3_Model_Validity_Learning.html](reviewer_responses/notebooks/R2/R2_R3_Model_Validity_Learning.html) |

#### **Referee #3 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Competing Risks** | Detailed competing risks analysis | [R3_Competing_Risks.html](reviewer_responses/notebooks/R3/R3_Competing_Risks.html) |
| **Fixed vs Joint Phi** | Comparison of fixed vs joint phi estimation | [R3_Fixed_vs_Joint_Phi_Comparison.html](reviewer_responses/notebooks/R3/R3_Fixed_vs_Joint_Phi_Comparison.html) |
| **FullE vs ReducedE** | Full vs reduced event matrix comparison | [R3_FullE_vs_ReducedE_Comparison.html](reviewer_responses/notebooks/R3/R3_FullE_vs_ReducedE_Comparison.html) |
| **Linear vs Nonlinear** | Linear vs nonlinear mixing approaches | [R3_Linear_vs_NonLinear_Mixing.html](reviewer_responses/notebooks/R3/R3_Linear_vs_NonLinear_Mixing.html) |
| **Population Stratification** | Ancestry-stratified analysis | [R3_Population_Stratification_Ancestry.html](reviewer_responses/notebooks/R3/R3_Population_Stratification_Ancestry.html) |
| **Heterogeneity** | Patient heterogeneity analysis | [R3_Q8_Heterogeneity.html](reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity.html) |
| **Heterogeneity (Main Paper Method)** | Main paper method with PRS validation (MI and breast cancer) | [R3_Q8_Heterogeneity_MainPaper_Method.html](reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity_MainPaper_Method.html) |
| **Heterogeneity (Continued)** | Complete pathway analysis demonstrating biological heterogeneity | [R3_Q8_Heterogeneity_Continued.html](reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity_Continued.html) |

#### **Framework & Preprocessing**

| Resource | Description | Link |
|----------|-------------|------|
| **Framework Overview** | Discovery vs prediction framework | [Discovery_Prediction_Framework_Overview.html](reviewer_responses/notebooks/framework/Discovery_Prediction_Framework_Overview.html) |
| **Preprocessing** | Preprocessing file creation guide | [create_preprocessing_files.html](reviewer_responses/preprocessing/create_preprocessing_files.html) |

---

## 📚 Documentation

### Core Model Files

| Component | File | Description |
|-----------|------|-------------|
| **Discovery Model** | [`../pyScripts_forPublish/clust_huge_amp.py`](../pyScripts_forPublish/clust_huge_amp.py) | Full model that learns phi and psi |
| **Prediction Model** | [`../pyScripts_forPublish/clust_huge_amp_fixedPhi.py`](../pyScripts_forPublish/clust_huge_amp_fixedPhi.py) | Fixed-phi model for fast predictions |
| **Discovery Notebook** | [`../pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb`](../pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb) | Interactive discovery mode |
| **Prediction Notebook** | [`../pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb`](../pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb) | Interactive prediction mode |

### Workflow Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **Preprocessing** | [`../pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py`](../pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py) | Preprocessing utilities |
| **Batch Training** | [`../claudefile/run_aladyn_batch.py`](../claudefile/run_aladyn_batch.py) | Batch model training |
| **Master Checkpoint** | [`../claudefile/create_master_checkpoints.py`](../claudefile/create_master_checkpoints.py) | Create pooled checkpoints |
| **Prediction** | [`../claudefile/version_from_ec2/run_aladyn_predict_with_master.py`](../claudefile/version_from_ec2/run_aladyn_predict_with_master.py) | Run predictions |

### Additional Resources

- **Complete Workflow Guide**: [`../pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md`](../pyScripts/new_oct_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md)
- **Main Repository README**: [`../README.md`](../README.md)
- **Installation Guide**: [`../INSTALLATION.md`](../INSTALLATION.md)

---

## ⏱️ Performance & Scalability

### Typical Runtime

| Configuration | Training Time | Memory Usage |
|---------------|---------------|--------------|
| CPU (Apple M4 Max) | ~70 minutes | ~12GB per 10,000 individuals |

### Scalability

- **Individuals**: Tested up to **400,000**
- **Diseases**: Tested **350**
- **Time Points**: Tested **52** (ages 30-81)
- **Genetic Features**: Tested with **36 PRS** and **10 PCs**

---

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

---

## 📞 Contact & Support

- **Author**: Sarah Urbut
- **Email**: surbut@mgh.harvard.edu
- **Institution**: Massachusetts General Hospital
- **GitHub**: [@surbut](https://github.com/surbut)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

We thank the UK Biobank participants and the research community for making this work possible.

---

<div align="center">

**Note**: This software is provided for research purposes. Please ensure you have appropriate data use agreements and ethical approvals before using with real patient data.

</div>
