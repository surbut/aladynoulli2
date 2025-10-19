#!/bin/bash
# EC2 Setup Script for Aladyn Predictions
# Run this on your EC2 instance after copying files

set -e  # Exit on error

echo "=========================================="
echo "Aladyn EC2 Setup Script"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y wget git

# Install Miniconda
echo "Installing Miniconda..."
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh

# Initialize conda
echo "Initializing conda..."
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Create project directory
echo "Creating project directories..."
mkdir -p ~/aladyn_project/data_for_running
mkdir -p ~/aladyn_project/output
mkdir -p ~/aladyn_project/logs

# Install AWS CLI
echo "Installing AWS CLI..."
sudo apt-get install -y awscli

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your code and environment.yml to the instance"
echo "2. Download data from S3"
echo "3. Create conda environment from environment.yml"
echo "4. Run the prediction script"
echo ""
echo "See ec2_run_predictions.sh for the remaining steps"
