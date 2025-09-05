# Installation Guide for Aladynoulli

## System Requirements

### Operating System
- **macOS**: 10.14 (Mojave) or later
- **Linux**: Ubuntu 18.04+ or equivalent
- **Windows**: Windows 10 or later (with WSL2 recommended)

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended for large datasets
- **Storage**: 5GB free space for installation and data
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

### Software Dependencies
- **Python**: 3.8 or higher
- **R**: 4.0 or higher (for RDS file support)
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/aladynoulli2.git
cd aladynoulli2
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using conda (recommended)
conda create -n aladynoulli python=3.9
conda activate aladynoulli

# OR using venv
python -m venv aladynoulli_env
source aladynoulli_env/bin/activate  # On Windows: aladynoulli_env\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install R Dependencies (if using R integration)
```r
# In R console
install.packages(c("rpy2", "readr", "dplyr"))
```

### 5. Verify Installation
```bash
python -c "import torch; import numpy; import pandas; print('Installation successful!')"
```

## Installation Time Estimates

| System Configuration | Estimated Time |
|---------------------|----------------|
| Modern laptop (8GB RAM, SSD) | 5-10 minutes |
| High-end workstation (32GB RAM, NVMe SSD) | 3-5 minutes |
| Cloud instance (AWS EC2 t3.medium) | 8-12 minutes |
| GPU-enabled system (with CUDA) | 10-15 minutes |

## Troubleshooting

### Common Issues

1. **PyTorch Installation Issues**
   ```bash
   # For CPU-only installation
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **R Integration Issues**
   ```bash
   # Install R development tools
   sudo apt-get install r-base-dev  # Ubuntu/Debian
   brew install r  # macOS
   ```

3. **Memory Issues**
   - Reduce batch size in notebooks
   - Use data subsetting functions provided
   - Consider using a machine with more RAM

### Platform-Specific Notes

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for R installation: `brew install r`

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip r-base r-base-dev
```

#### Windows
- Use Windows Subsystem for Linux (WSL2) for best compatibility
- Install Visual Studio Build Tools for C++ compilation
- Use Anaconda/Miniconda for easier package management

## Data Requirements

The model requires the following data files (not included in repository):
- `Y_tensor.pt`: Disease outcome tensor
- `E_matrix.pt`: Censoring matrix
- `G_matrix.pt`: Genetic data matrix
- `model_essentials.pt`: Model configuration
- `reference_trajectories.pt`: Reference trajectories
- `initial_psi_400k.pt`: Initial psi parameters
- `initial_clusters_400k.pt`: Initial cluster assignments

## Quick Start

1. Place your data files in the appropriate directory
2. Run the discovery notebook:
   ```bash
   jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb
   ```
3. Run the prediction notebook:
   ```bash
   jupyter notebook pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb
   ```

## Support

For installation issues, please contact: surbut@mgh.harvard.edu
