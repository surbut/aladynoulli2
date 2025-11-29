#!/bin/bash
# Quick setup script for running run_aladyn_batch.py on EC2
# Usage: bash setup_and_run_ec2.sh [start_index] [end_index]

set -e

START_INDEX="${1:-0}"
END_INDEX="${2:-10000}"

echo "=========================================="
echo "EC2 Setup and Run Script"
echo "=========================================="
echo "Start Index: $START_INDEX"
echo "End Index: $END_INDEX"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "=========================================="
    echo "Installing AWS CLI v2..."
    echo "=========================================="
    # Install required tools
    sudo apt-get update
    sudo apt-get install -y curl unzip
    
    cd /tmp
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    # Verify installation
    aws --version
fi

# Check if AWS CLI can access S3 (either via IAM role or configured credentials)
echo ""
echo "=========================================="
echo "Checking AWS access..."
echo "=========================================="

# Test if we can access S3 (this works with IAM role OR configured credentials)
if aws s3 ls s3://sarah-research-aladynoulli/ &>/dev/null; then
    echo "✓ AWS access confirmed (using IAM role or existing credentials)"
else
    echo "⚠️  Cannot access S3. Need to configure AWS credentials."
    
    # Check if credentials file exists
    if [ ! -f ~/.aws/credentials ]; then
        echo ""
        echo "=========================================="
        echo "Configuring AWS CLI..."
        echo "=========================================="
        echo "You need to enter your AWS Access Key ID and Secret Access Key."
        echo "These can be found in: AWS Console > IAM > Users > Your User > Security Credentials"
        echo ""
        echo "If you don't have access keys, you can:"
        echo "  1. Create them in IAM Console (Security Credentials tab)"
        echo "  2. Or attach an IAM role to this EC2 instance (no keys needed)"
        echo ""
        read -p "Press Enter to start AWS configuration (or Ctrl+C to exit and set up IAM role)..."
        
        # Run aws configure interactively
        aws configure
        
        # Test again after configuration
        echo ""
        echo "Testing S3 access..."
        if aws s3 ls s3://sarah-research-aladynoulli/ &>/dev/null; then
            echo "✓ AWS access confirmed!"
        else
            echo "✗ Still cannot access S3. Please check your credentials and try again."
            exit 1
        fi
    else
        echo "Credentials file exists but S3 access failed."
        echo "Please check your credentials or run 'aws configure' to update them."
        exit 1
    fi
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo ""
    echo "=========================================="
    echo "Installing Miniconda..."
    echo "=========================================="
    cd ~
    # Check if miniconda3 directory already exists
    if [ -d "$HOME/miniconda3" ]; then
        echo "Miniconda directory exists, initializing..."
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    else
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda3
        rm miniconda.sh
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash
        source ~/.bashrc
    fi
else
    echo ""
    echo "Conda is already installed."
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || \
    eval "$($(which conda | head -1 | xargs dirname)/../etc/profile.d/conda.sh)" 2>/dev/null || true
fi

# Check for existing conda environments
echo ""
echo "Checking for conda environments..."
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null

# Create conda environment if it doesn't exist
if conda env list | grep -q "^aladyn "; then
    echo ""
    echo "✓ Conda environment 'aladyn' already exists"
    echo "Activating conda environment..."
    conda activate aladyn
    echo "Checking if required packages are installed..."
    # Check if key packages are installed
    if python -c "import torch, numpy, pandas, scipy, sklearn, matplotlib" 2>/dev/null; then
        echo "✓ All required packages are installed"
    else
        echo "Installing missing packages..."
        pip install torch numpy pandas scipy scikit-learn matplotlib seaborn
    fi
else
    echo ""
    echo "=========================================="
    echo "Creating conda environment 'aladyn'..."
    echo "=========================================="
    conda create -n aladyn python=3.9 -y
    conda activate aladyn
    echo "Installing required packages..."
    pip install torch numpy pandas scipy scikit-learn matplotlib seaborn
fi

# Create directories
echo ""
echo "Creating directories..."
sudo mkdir -p /data
sudo mkdir -p /results
sudo chown ubuntu:ubuntu /data /results 2>/dev/null || sudo chown $USER:$USER /data /results

# Download data from S3
echo ""
echo "=========================================="
echo "Downloading data from S3..."
echo "=========================================="
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ /data/ --no-progress

# Verify required files
echo ""
echo "Verifying required files..."
REQUIRED_FILES=(
    "Y_tensor.pt"
    "E_matrix.pt"
    "G_matrix.pt"
    "model_essentials.pt"
    "reference_trajectories.pt"
    "initial_psi_400k.pt"
    "initial_clusters_400k.pt"
    "baselinagefamh_withpcs.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "/data/$file" ]; then
        size=$(du -h "/data/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file - NOT FOUND"
        exit 1
    fi
done

echo ""
echo "All required files present!"
echo ""

# Run the batch script
echo "=========================================="
echo "Running batch script..."
echo "=========================================="
# Ensure we're using the conda environment Python
PYTHON=$(which python)
echo "Using Python: $PYTHON"

$PYTHON run_aladyn_batch.py \
    --start_index $START_INDEX \
    --end_index $END_INDEX \
    --data_dir /data \
    --output_dir /results \
    --covariates_path /data/baselinagefamh_withpcs.csv

echo ""
echo "=========================================="
echo "Complete! Check /results for output"
echo "=========================================="

