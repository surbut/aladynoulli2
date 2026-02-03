# Aladynoulli: A Bayesian Survival Model for Disease Trajectory Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A comprehensive Bayesian survival model for predicting disease trajectories using genetic and clinical data.

**Preprint**: [medRxiv](https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v1)

**Documentation & Reviewer Analyses**: [https://surbut.github.io/aladynoulli2/](https://surbut.github.io/aladynoulli2/)

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

The Aladynoulli workflow consists of 5 main steps:

1. **Preprocessing**: Create smoothed prevalence, initial clusters, and reference trajectories
2. **Batch Training**: Train models on data batches with full E matrix
3. **Master Checkpoint**: Generate pooled checkpoint (phi and psi)
4. **Pool Gamma & Kappa**: Pool genetic effects and calibration from training batches
5. **Prediction**: Run predictions using master checkpoint

**For detailed step-by-step instructions**, see the [Complete Workflow Guide](https://github.com/surbut/aladynoulli2/blob/main/pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md).

### Quick Start

```python
# Discovery mode (learns phi)
from pyScripts_forPublish.clust_huge_amp_vectorized import AladynSurvivalFixedKernelsAvgLoss

model = AladynSurvivalFixedKernelsAvgLoss(
    N=Y.shape[0], D=Y.shape[1], T=Y.shape[2], K=20,
    P=G.shape[1], G=G, Y=Y,
    prevalence_t=prevalence_t,
    signature_references=signature_refs
)
history = model.fit(E, num_epochs=200)
pi, theta, phi = model.forward()
```

### Interactive Notebooks

**Discovery Mode** (full model training):
```bash
jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery_noweights.ipynb
```

**Prediction Mode** (fixed phi):
```bash
jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb
```



## 📚 Documentation

**Full documentation available at**: [https://surbut.github.io/aladynoulli2/](https://surbut.github.io/aladynoulli2/)

### Core Model Files

- **Discovery Model**: [`pyScripts_forPublish/clust_huge_amp_vectorized.py`](pyScripts_forPublish/clust_huge_amp_vectorized.py) - Full model that learns phi and psi
- **Prediction Model**: [`pyScripts_forPublish/clust_huge_amp_fixedPhi.py`](pyScripts_forPublish/clust_huge_amp_fixedPhi.py) - Fixed-phi model for fast predictions

### Key Resources

| Resource | Description |
|----------|-------------|
| [Framework Overview](https://surbut.github.io/aladynoulli2/reviewer_responses/notebooks/framework/Discovery_Prediction_Framework_Overview.html) | Discovery vs prediction framework |
| [Reviewer Response Analyses](https://surbut.github.io/aladynoulli2/reviewer_responses/README.html) | Complete guide to all validation analyses |
| [Workflow Guide](https://github.com/surbut/aladynoulli2/blob/main/pyScripts/dec_6_revision/new_notebooks/reviewer_responses/preprocessing/WORKFLOW.md) | Step-by-step preprocessing → training → prediction |

### Data Requirements

**Required input files** (created during preprocessing):
- `Y_tensor.pt`: Disease outcome tensor (N × D × T)
- `E_matrix.pt`: Censoring matrix (N × D)
- `G_matrix.pt`: Genetic data matrix (N × P)
- `model_essentials.pt`: Model configuration (disease names, etc.)
- `reference_trajectories.pt`: Signature reference trajectories
- `initial_psi_400k.pt`: Initial psi parameters
- `initial_clusters_400k.pt`: Initial cluster assignments

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
