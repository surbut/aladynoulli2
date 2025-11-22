# Setting Up Python Environment on AWS EC2

## Step 1: Connect to Your EC2 Instance

```bash
ssh -i ~/Downloads/sarahkey.pem ec2-user@YOUR_EC2_IP
# or
ssh -i ~/Downloads/sarahkey.pem ec2-user@ec2-3-81-0-40.compute-1.amazonaws.com
```

## Step 2: Check if Conda is Already Installed

```bash
which conda
conda --version
```

If conda is NOT installed, install Miniconda (lightweight):

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make it executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer (follow prompts, say yes to all)
./Miniconda3-latest-Linux-x86_64.sh

# When asked, say yes to initialize conda
# Then reload your shell or run:
source ~/.bashrc
```

## Step 3: Create and Activate Environment

```bash
# Create environment with Python 3.9
conda create -n aladyn python=3.9 -y

# Activate environment
conda activate aladyn

# Verify Python version
python --version  # Should show Python 3.9.x
which python      # Should point to conda env
```

## Step 4: Install Packages

**Option A: Install all at once (recommended)**
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn statsmodels
```

**Option B: Install PyTorch separately (for CPU version)**
```bash
# Install PyTorch CPU version (no CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install rest of packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn statsmodels
```

**Option C: Use requirements.txt (if you uploaded it)**
```bash
# If you're in the scripts directory
pip install -r requirements.txt
```

## Step 5: Verify Installation

```bash
# Check all packages are installed
python -c "import torch; import numpy; import pandas; import scipy; import sklearn; import matplotlib; import seaborn; import statsmodels; print('All packages imported successfully!')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### If pip is not found:
```bash
# Install pip in conda environment
conda install pip -y
```

### If conda command not found after installation:
```bash
# Reload your shell configuration
source ~/.bashrc
# or
source ~/.zshrc

# Or manually add to PATH (temporary)
export PATH="$HOME/miniconda3/bin:$PATH"
```

### If installation is slow:
```bash
# Upgrade pip first
pip install --upgrade pip

# Then install packages
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn statsmodels
```

### To check what's installed:
```bash
pip list
# or
conda list
```

### If you need to reinstall:
```bash
# Deactivate environment
conda deactivate

# Remove and recreate
conda env remove -n aladyn
conda create -n aladyn python=3.9 -y
conda activate aladyn
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn statsmodels
```

## Quick Setup Script

You can also create a setup script:

```bash
# Create setup script
cat > setup_env.sh << 'EOF'
#!/bin/bash
set -e  # Exit on error

echo "Creating conda environment..."
conda create -n aladyn python=3.9 -y

echo "Activating environment..."
source activate aladyn

echo "Installing packages..."
pip install --upgrade pip
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn statsmodels

echo "Verifying installation..."
python -c "import torch, numpy, pandas, scipy, sklearn, matplotlib, seaborn, statsmodels; print('✓ All packages installed successfully!')"

echo "Setup complete!"
EOF

# Make executable and run
chmod +x setup_env.sh
./setup_env.sh
```

