#!/bin/bash
# EC2 Instance Setup Script for Aladynoulli Training
# Run this script on a fresh EC2 instance to set up the environment

set -e  # Exit on error

echo "========================================="
echo "Aladynoulli EC2 Setup Script"
echo "========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install AWS CLI v2
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Install git
echo "Installing git..."
sudo apt-get install -y git

# Optional: Install NVIDIA Docker for GPU support
read -p "Install NVIDIA Docker for GPU support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing NVIDIA Docker..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
fi

# Create directories
echo "Creating directories..."
mkdir -p ~/aladynoulli2/data
mkdir -p ~/aladynoulli2/results
mkdir -p ~/aladynoulli2/logs

# Configure AWS credentials (if needed)
read -p "Configure AWS credentials for S3 access? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Configuring AWS CLI..."
    aws configure
fi

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in (or run: newgrp docker)"
echo "2. Clone your repository: git clone your-repo-url"
echo "3. Copy your data to ~/aladynoulli2/data/"
echo "4. Build Docker image: cd aladynoulli2 && docker build -t aladynoulli-training -f claudefile/Dockerfile ."
echo "5. Run training: ./claudefile/run-training.sh"
echo ""
echo "For GPU instances, verify GPU access with: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"
