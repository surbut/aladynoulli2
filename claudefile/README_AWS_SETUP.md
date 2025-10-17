# AWS EC2 Setup for Aladynoulli Python Training

This guide shows you how to set up a reproducible environment on AWS EC2 to run your Python training scripts repeatedly using Docker containers.

## Overview

Your main training script is: `pyScripts/local_survival_training.py`

This setup will:
1. Create a Docker container with all dependencies
2. Allow you to run training jobs on AWS EC2 instances
3. Provide a reproducible environment that can be reused
4. Save results back to your data storage

## Quick Start

### 1. On Your Local Machine

```bash
# Navigate to your project
cd /Users/sarahurbut/aladynoulli2

# Build the Docker image
docker build -t aladynoulli-training -f claudefile/Dockerfile .

# Test locally (optional)
docker run -v /path/to/your/data:/data \
  -v $(pwd)/results:/results \
  aladynoulli-training python pyScripts/local_survival_training.py
```

### 2. On AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-instance

# Clone your repository or copy files
git clone your-repo-url

# Build the Docker image on EC2
cd aladynoulli2
docker build -t aladynoulli-training -f claudefile/Dockerfile .

# Run training
docker run -v /path/to/data:/data \
  -v $(pwd)/results:/results \
  aladynoulli-training python pyScripts/local_survival_training.py
```

## File Structure

```
claudefile/
├── README_AWS_SETUP.md          # This file
├── Dockerfile                    # Docker container definition
├── docker-compose.yml            # Easy container orchestration
├── ec2-setup.sh                  # EC2 instance setup script
├── run-training.sh               # Script to run training jobs
├── requirements-docker.txt       # Python dependencies for Docker
└── aws-ec2-userdata.sh          # EC2 user data script for auto-setup
```

## Detailed Instructions

See the individual files for more details on each component.

## Data Requirements

The training script expects these files in your data directory:
- `Y_tensor.pt`
- `E_matrix.pt`
- `G_matrix.pt`
- `model_essentials.pt`
- `reference_trajectories.pt`
- `baselinagefamh.csv`
- `enrollment_model_W0.0001_fulldata_sexspecific.pt`
- `initial_psi_400k.pt` (optional)
- `initial_clusters_400k.pt` (optional)

## Cost Optimization Tips

1. Use Spot Instances for training (much cheaper)
2. Use EBS snapshots to save data between runs
3. Auto-shutdown instances when training completes
4. Use S3 for data storage instead of EBS when possible

## Next Steps

1. Review and customize the Dockerfile
2. Set up your EC2 instance using `ec2-setup.sh`
3. Upload your data to S3 or attach EBS volume
4. Run training jobs using `run-training.sh`
