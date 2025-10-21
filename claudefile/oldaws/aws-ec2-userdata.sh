#!/bin/bash
# EC2 User Data Script for Automatic Setup
# This script runs when an EC2 instance is first launched
# Use this in the "User Data" field when launching an EC2 instance

set -e

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting EC2 instance setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install AWS CLI v2
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Install git
apt-get install -y git

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Create directories
mkdir -p /home/ubuntu/aladynoulli2/{data,results,logs}
chown -R ubuntu:ubuntu /home/ubuntu/aladynoulli2

# Optional: Download data from S3 (configure S3_BUCKET_PATH as needed)
# Uncomment and configure the following lines if you want to auto-download data
# S3_BUCKET_PATH="s3://your-bucket/data/"
# su - ubuntu -c "aws s3 sync $S3_BUCKET_PATH /home/ubuntu/aladynoulli2/data/"

# Optional: Clone your repository
# GIT_REPO="https://github.com/yourusername/aladynoulli2.git"
# su - ubuntu -c "cd /home/ubuntu && git clone $GIT_REPO"

# Optional: Build Docker image
# su - ubuntu -c "cd /home/ubuntu/aladynoulli2 && docker build -t aladynoulli-training -f claudefile/Dockerfile ."

# Optional: Auto-shutdown after training completes (for cost savings)
# Uncomment the following to enable auto-shutdown
# cat > /home/ubuntu/shutdown-after-training.sh << 'EOF'
# #!/bin/bash
# # Wait for training to complete, then shutdown
# while docker ps | grep -q aladynoulli-trainer; do
#     sleep 60
# done
# sleep 300  # Wait 5 minutes after training completes
# shutdown -h now
# EOF
# chmod +x /home/ubuntu/shutdown-after-training.sh
# chown ubuntu:ubuntu /home/ubuntu/shutdown-after-training.sh

echo "EC2 setup complete!" > /home/ubuntu/setup-complete.txt
echo "Setup complete at $(date)" >> /var/log/user-data.log
