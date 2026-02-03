# Aladynoulli Documentation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**A Bayesian Survival Model for Disease Trajectory Prediction**

[Preprint](https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v1) 

</div>

---

## 📋 How to Use This Documentation

This documentation is organized into **four main sections** for reviewers:

1. **[Model Architecture](#model-architecture)** - Understand how the model works: core components, mathematical framework, and key concepts
2. **[Reviewer Response Analyses](#reviewer-response-analyses)** - Interactive analyses addressing all reviewer questions, organized by referee
3. **[Complete Workflow](#complete-workflow)** - Step-by-step guide to running the model: preprocessing, training, and prediction
4. **[Performance & Scalability](#performance-scalability)** - Computational requirements and scaling characteristics

**Note**: Installation instructions are not required for reviewers. A pre-configured environment will be provided for running the code.

---

## 📖 Table of Contents

- [How to Use This Documentation](#how-to-use-this-documentation)
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Reviewer Response Analyses](#reviewer-response-analyses)
- [Complete Workflow](#complete-workflow)
- [Performance & Scalability](#performance-scalability)
- [Citation](#citation)

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
- `φ_k,d,t` = sigmoid(μ_d+ψ_k,d, K_φ) (disease probabilities)

---


## 📊 Reviewer Response Analyses

Comprehensive interactive analyses addressing reviewer questions and model validation.

### 📚 Navigation Hub

- **[Reviewer Response README](reviewer_responses/README.html)** - Complete guide to all interactive analyses
- **[Framework Overview](https://surbut.github.io/aladynoulli2/reviewer_responses/notebooks/framework/Discovery_Prediction_Framework_Overview.html)** - Discovery vs prediction framework (essential reading)

### 🔬 Analysis Categories

#### **Referee #1 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Clinical Utility** | Dynamic risk updating and clinical decision-making | [R1_Clinical_Utility_Dynamic_Risk_Updating.html](reviewer_responses/notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.html) |
| **AUC Comparisons** | Performance vs. established clinical risk scores | [R1_Q9_AUC_Comparisons.html](reviewer_responses/notebooks/R1/R1_Q9_AUC_Comparisons.html) |
| **Age-Stratified** | Performance across different age groups | [R1_Q10_Age_Specific.html](reviewer_responses/notebooks/R1/R1_Q10_Age_Specific.html) |
| **Heritability** | Genetic architecture and heritability estimates | [R1_Q7_Heritability.html](reviewer_responses/notebooks/R1/R1_Q7_Heritability.html) |
| **GWAS Validation** | Genome-wide association studies on signatures; identifies 10 novel loci for Signature 5 not found in individual trait GWAS | [R1_Genetic_Validation_GWAS.html](reviewer_responses/notebooks/R1/R1_Genetic_Validation_GWAS.html) |
| **Gene-Based RVAS** | Rare variant association studies on signatures | [R1_Genetic_Validation_Gene_Based_RVAS.html](reviewer_responses/notebooks/R1/R1_Genetic_Validation_Gene_Based_RVAS.html) |
| **Biological Plausibility** | CHIP analysis and biological validation | [R1_Biological_Plausibility_CHIP.html](reviewer_responses/notebooks/R1/R1_Biological_Plausibility_CHIP.html) |
| **LOO Validation** | Leave-one-out cross-validation robustness | [R1_Robustness_LOO_Validation.html](reviewer_responses/notebooks/R1/R1_Robustness_LOO_Validation.html) |
| **Selection Bias** | Assessment of selection bias and participation | [R1_Q1_Selection_Bias.html](reviewer_responses/notebooks/R1/R1_Q1_Selection_Bias.html) |
| **Clinical Meaning** | Analysis of Familial hypercholesterolemia patients | [R1_Q3_Clinical_Meaning.html](reviewer_responses/notebooks/R1/R1_Q3_Clinical_Meaning.html) |
| **ICD vs PheCode** | Detailed comparison of coding systems | [R1_Q3_ICD_vs_PheCode_Comparison.html](reviewer_responses/notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.html) |
| **Competing Risks** | Multi-disease patterns and competing risks | [R1_Multi_Disease_Patterns_Competing_Risks.html](reviewer_responses/notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.html) |

#### **Referee #2 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Temporal Leakage** | Assessment of temporal leakage and prediction accuracy | [R2_Temporal_Leakage.html](reviewer_responses/notebooks/R2/R2_Temporal_Leakage.html) |
| **Washout Comparisons** | Multi-approach washout analysis (time horizon, floating prediction, fixed timepoint) | [R2_Washout_Comparisons.html](reviewer_responses/notebooks/R2/R2_Washout_Comparisons.html) |
| **Delphi Phecode Mapping** | Principled Delphi comparison using Phecode-based ICD mapping | [R2_Delphi_Phecode_Mapping.html](reviewer_responses/notebooks/R2/R2_Delphi_Phecode_Mapping.html) |
| **Model Validity** | Model learning and validity assessment | [R2_R3_Model_Validity_Learning.html](reviewer_responses/notebooks/R2/R2_R3_Model_Validity_Learning.html) |

#### **Referee #3 Analyses**

| Analysis | Description | Link |
|----------|-------------|------|
| **Avoiding Reverse Causation** | Reverse causation assessment with 0, 1, 3, 6-month washout periods | [R3_AvoidingReverseCausation.html](reviewer_responses/notebooks/R3/R3_AvoidingReverseCausation.html) |
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


## 💻 Complete Workflow

The Aladynoulli workflow consists of **5 main steps**:

1. **Preprocessing**: Create smoothed prevalence, initial clusters, and reference trajectories
2. **Batch Training**: Train models on data batches with full E matrix
3. **Master Checkpoint**: Generate pooled checkpoint (phi and psi)
4. **Pool Gamma & Kappa**: Pool gamma (genetic effects) and kappa (calibration) from training batches
5. **Prediction**: Run predictions using master checkpoint with fixed gamma and kappa (only lambda is learned)

### Essential Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Framework Overview** | **Discovery vs prediction framework - Essential reading** | [Discovery_Prediction_Framework_Overview.html](https://surbut.github.io/aladynoulli2/reviewer_responses/notebooks/framework/Discovery_Prediction_Framework_Overview.html) |
| **Complete Workflow Guide** | Step-by-step preprocessing → training → prediction | [WORKFLOW.md](https://github.com/surbut/aladynoulli2/blob/main/pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md) |
| **Preprocessing Guide** | Preprocessing file creation guide | [create_preprocessing_files.html](https://surbut.github.io/aladynoulli2/reviewer_responses/preprocessing/create_preprocessing_files.html) |

### Core Model Files

| Component | File | Description |
|-----------|------|-------------|
| **Discovery Model** | [clust_huge_amp_vectorized.py](https://github.com/surbut/aladynoulli2/blob/main/pyScripts_forPublish/clust_huge_amp_vectorized.py) | Full model that learns phi and psi |
| **Prediction Model** | [clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py](https://github.com/surbut/aladynoulli2/blob/main/claudefile/aws_offsetmaster/clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py) | Fixed-phi, fixed-gamma, fixed-kappa model for fast predictions |

**Note**: The prediction model uses fixed gamma (genetic effects) and kappa (calibration parameter) from pooled training batches. This ensures complete separation between training and testing data in each validation fold. Only lambda (individual-specific signature loadings) is learned during prediction.

### Workflow Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **Preprocessing** | [preprocessing_utils.py](https://github.com/surbut/aladynoulli2/blob/main/pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/preprocessing_utils.py) | Preprocessing utilities |
| **Batch Training** | [run_aladyn_batch_vector_e_censor.py](https://github.com/surbut/aladynoulli2/blob/main/claudefile/run_aladyn_batch_vector_e_censor.py) | Batch model training with corrected E |
| **Master Checkpoint** | [create_master_checkpoints.py](https://github.com/surbut/aladynoulli2/blob/main/claudefile/create_master_checkpoints.py) | Create pooled checkpoints (phi and psi) |
| **Pool Gamma & Kappa** | [pool_kappa_and_gamma_from_nolr_batches.py](https://github.com/surbut/aladynoulli2/blob/main/claudefile/pool_kappa_and_gamma_from_nolr_batches.py) | Pool gamma (genetic effects) and kappa (calibration) from training batch checkpoints |
| **Prediction** | [run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py](https://github.com/surbut/aladynoulli2/blob/main/claudefile/run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py) | Run enrollment-based predictions using enrollment E matrix (E_enrollment_full.pt) with master checkpoint from corrected E training, using fixed gamma and kappa from pooled training batches (only lambda is learned per batch)

---

## 📈 Performance & Scalability

### Computational Requirements

**For 10K individuals, 348 diseases, 52 timepoints:**
- **Training Time**: ~8-10 minutes per batch (converges after ~200 epochs)
- **Prediction Time**: ~8 minutes per batch
- **Memory**: ~8GB RAM (peak usage during training)
- **CPU**: Multi-core recommended (4+ cores); PyTorch uses BLAS for parallel matrix operations

**Scaling:**
- **Full UK Biobank (400K individuals)**: Processed in 39 batches of ~10K each
- **Total training time**: ~5-7 hours for all batches (can be parallelized)
- **Memory scales linearly**: ~8GB per 10K batch

**Why it's fast:**
- Vectorized PyTorch operations (batched matrix decompositions)
- BLAS Level 3 operations for efficient linear algebra
- ~100-fold speedup compared to loop-based implementation

---



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
