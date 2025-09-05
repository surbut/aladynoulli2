# Aladynoulli: A Bayesian Survival Model for Disease Trajectory Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A comprehensive Bayesian survival model for predicting disease trajectories using genetic and clinical data. This repository contains the complete implementation of the Aladynoulli model as described in our preprint.

**Preprint**: [medRxiv](https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v1)

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

### Discovery Mode

For full model discovery and signature learning:

```python
# Run the discovery notebook
jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb
```

### Prediction Mode

For prediction using pre-trained signatures:

```python
# Run the prediction notebook
jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb
```

### Programmatic Usage

```python
from pyScripts_forPublish.clust_huge_amp import *

# Initialize model

model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=Y_100k.shape[0], 
    D=Y_100k.shape[1], 
    T=Y_100k.shape[2], 
    K=20,
    P=G_with_sex.shape[1],
    init_sd_scaler=1e-1,
    G=G_with_sex, 
    Y=Y_100k,
    genetic_scale=1,
    W=0.0001,
    R=0,
    prevalence_t=essentials['prevalence_t'],
    signature_references=signature_refs,  # Only pass signature refs
    healthy_reference=True,  # Explicitly set to None
    disease_names=essentials['disease_names']
)

# Train model
history = model.fit(event_times, num_epochs=200)

# Make predictions
pi, theta, phi = model.forward()
```

## 📚 Documentation

### Core Scripts

- **Main Model**: [`pyScripts_forPublish/clust_huge_amp_fixedPhi.py`](pyScripts_forPublish/clust_huge_amp_fixedPhi.py)
- **Discovery Notebook**: [`pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb`](pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb)
- **Prediction Notebook**: [`pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb`](pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb)

### Additional Tools

- **Streamlit App**: [`pyScripts_forPublish/patient_timeline_app`](pyScripts_forPublish/patient_timeline_app)
- **AWS Scripts**: [`pyScripts_forPublish/submit_script_aws_fixedph_40_70.py`](pyScripts_forPublish/submit_script_aws_fixedph_40_70.py)
- **Demo Script**: [`pyScripts_forPublish/newsm_3_71.ipynb`](pyScripts_forPublish/newsm_3_71.ipynb)

### Data Requirements

The model requires the following data files:
- `Y_tensor.pt`: Disease outcome tensor (N × D × T)
- `E_matrix.pt`: Censoring matrix (N × D)
- `G_matrix.pt`: Genetic data matrix (N × P)
- `model_essentials.pt`: Model configuration
- `reference_trajectories.pt`: Reference trajectories
- `initial_psi_400k.pt`: Initial psi parameters
- `initial_clusters_400k.pt`: Initial cluster assignments

## ⏱️ Performance

### Typical Runtime (10,000 individuals, 50 diseases, 50 time points)

| Configuration | Training Time | Memory Usage |
|---------------|---------------|--------------|
| CPU (8 cores) | ~15 minutes | ~4GB |
| GPU (RTX 3080) | ~3 minutes | ~6GB |
| Cloud (AWS t3.xlarge) | ~8 minutes | ~8GB |

### Scalability

- **Individuals**: Tested up to 400,000
- **Diseases**: Tested up to 400
- **Time Points**: Tested up to 70
- **Genetic Features**: Tested up to 1M SNPs


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
