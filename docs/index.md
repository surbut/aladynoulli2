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

1. **Preprocessing**: Create smoothed prevalence, initial clusters, and reference trajectories
2. **Batch Training**: Train models on data batches with full E matrix
3. **Master Checkpoint**: Generate pooled checkpoint (phi and psi)
4. **Prediction**: Run predictions using master checkpoint

For detailed step-by-step instructions, see the [Complete Workflow Guide](../pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md).

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
| **Model Validity** | Model learning and validity assessment | [R2_R3_Model_Validity_Learning.html](reviewer_responses/notebooks/R2/R2_R3_Model_Validity_Learning.html) |

#### **Referee #3 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Competing Risks** | Detailed competing risks analysis | [R3_Competing_Risks.html](reviewer_responses/notebooks/R3/R3_Competing_Risks.html) |
| **Decreasing Hazards (Censoring Bias)** | Analysis of decreasing hazards at older ages due to censoring bias | [R3_Q4_Decreasing_Hazards_Censoring_Bias.html](reviewer_responses/notebooks/R3/R3_Q4_Decreasing_Hazards_Censoring_Bias.html) |
| **Verify Corrected Data** | Verification of corrected E matrix and prevalence calculations | [R3_Verify_Corrected_Data.html](reviewer_responses/notebooks/R3/R3_Verify_Corrected_Data.html) |
| **FullE vs ReducedE** | Full vs reduced event matrix comparison | [R3_FullE_vs_ReducedE_Comparison.html](reviewer_responses/notebooks/R3/R3_FullE_vs_ReducedE_Comparison.html) |
| **Linear vs Nonlinear** | Linear vs nonlinear mixing approaches | [R3_Linear_vs_NonLinear_Mixing.html](reviewer_responses/notebooks/R3/R3_Linear_vs_NonLinear_Mixing.html) |
| **Population Stratification** | Ancestry-stratified analysis | [R3_Population_Stratification_Ancestry.html](reviewer_responses/notebooks/R3/R3_Population_Stratification_Ancestry.html) |
| **Heterogeneity** | Patient heterogeneity analysis | [R3_Q8_Heterogeneity.html](reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity.html) |
| **Heterogeneity (Main Paper Method)** | Main paper method with PRS validation (MI and breast cancer) | [R3_Q8_Heterogeneity_MainPaper_Method.html](reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity_MainPaper_Method.html) |
| **Heterogeneity (Continued)** | Complete pathway analysis demonstrating biological heterogeneity | [R3_Q8_Heterogeneity_Continued.html](reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity_Continued.html) |
| **Cross-Cohort Similarity** | Cross-cohort signature correspondence analysis | [R3_Cross_Cohort_Similarity.html](reviewer_responses/notebooks/R3/R3_Cross_Cohort_Similarity.html) |

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
| **Preprocessing** | [`../pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py`](../pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py) | Preprocessing utilities |
| **Batch Training** | [`../claudefile/run_aladyn_batch_vector_e_censor.py`](../claudefile/run_aladyn_batch_vector_e_censor.py) | Batch model training with corrected E |
| **Master Checkpoint** | [`../claudefile/create_master_checkpoints.py`](../claudefile/create_master_checkpoints.py) | Create pooled checkpoints |
| **Prediction** | [`../claudefile/run_aladyn_predict_with_master_vector_cenosrE_fullEtest.py`](../claudefile/run_aladyn_predict_with_master_vector_cenosrE_fullEtest.py) | Run predictions with corrected E |

---

## 📈 Performance & Scalability

- **Dataset Size**: 400K+ individuals, 348 diseases, 52 timepoints
- **Training Time**: ~2-3 hours per 10K batch on GPU
- **Prediction Time**: ~30 minutes for 400K individuals
- **Memory**: ~8GB GPU memory for batch training

---

## 📝 Citation

If you use Aladynoulli in your research, please cite:

```bibtex
@article{aladynoulli2024,
  title={Aladynoulli: A Bayesian Survival Model for Disease Trajectory Prediction},
  author={Sur, P. and others},
  journal={medRxiv},
  year={2024},
  doi={10.1101/2024.09.29.24314557}
}
```

---

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/surbut/aladynoulli2/issues).

---

<div align="center">

**Last Updated**: December 2024

</div>
