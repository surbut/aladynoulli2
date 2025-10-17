# AWS EC2 Workflow for Aladyn Training

## Overview
This document outlines the recommended workflow for running Aladyn training jobs on AWS EC2, solving the cost vs. setup-time tradeoff with AMI snapshots.

## The Problem
- **Option 1:** Keep EC2 running 24/7 → Expensive (~$1-3/hour for compute instances)
- **Option 2:** Shut down when not in use → Have to reinstall environment every time → Annoying

## The Solution: AMI Snapshots
Use Amazon Machine Images (AMI) to save your configured environment once, then launch new instances from it instantly.

---

## One-Time Setup (Do This Once)

### Step 1: Upload Data to S3

```bash
# On your local Mac, upload data to S3 (one time only)
aws s3 mb s3://aladynoulli-data  # Create bucket

# Upload input data
aws s3 cp /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/ \
  s3://aladynoulli-data/input/ --recursive

# Verify upload
aws s3 ls s3://aladynoulli-data/input/ --recursive --human-readable
```

**Cost:** ~$0.023/GB/month (very cheap for storage)

### Step 2: Launch EC2 Instance

```bash
# Launch Ubuntu instance (t3.xlarge or GPU instance)
# Use Ubuntu 22.04 LTS AMI
# Add IAM role with S3 access permissions
```

### Step 3: Run Setup Script (ONE TIME)

SSH into instance and run:
```bash
# Copy ec2-setup.sh to instance
scp claudefile/ec2-setup.sh ubuntu@<instance-ip>:~

# Run setup
ssh ubuntu@<instance-ip>
chmod +x ec2-setup.sh
./ec2-setup.sh
```

This installs:
- Docker
- AWS CLI
- Git
- Python dependencies

### Step 4: Clone Your Repo and Build Docker Image

```bash
# Clone repository
cd /home/ubuntu
git clone <your-repo-url> aladynoulli2

# Build Docker image
cd aladynoulli2
docker build -t aladynoulli-training -f claudefile/Dockerfile .
```

### Step 5: Test Data Download and Training

```bash
# Download data from S3
./claudefile/s3-sync-script.sh download

# Test training on small batch
cd claudefile
python run_aladyn_batch.py \
  --start_index 0 \
  --end_index 1000 \
  --num_epochs 10 \
  --data_dir /home/ubuntu/aladynoulli2/data \
  --output_dir /home/ubuntu/aladynoulli2/results
```

### Step 6: Create AMI Snapshot

**In AWS Console:**
1. Go to EC2 → Instances
2. Select your instance
3. Actions → Image and templates → Create image
4. Name it: `aladynoulli-env-v1`
5. Wait ~10 minutes for AMI to be created

**Cost:** ~$0.05/GB/month for AMI storage (cheap!)

### Step 7: Terminate Instance
You can now terminate this instance. Your environment is saved in the AMI!

---

## Running Jobs (Every Time You Need to Train)

### Launch Instance from AMI

**In AWS Console:**
1. EC2 → Images → AMIs
2. Select `aladynoulli-env-v1`
3. Launch instance from AMI
4. Choose instance type (t3.xlarge for CPU, g4dn.xlarge for GPU)
5. Add IAM role with S3 permissions
6. Launch

**Result:** Instance boots with Docker, code, and environment already configured!

### Run Training Job

```bash
# SSH into instance
ssh ubuntu@<instance-ip>

# Download latest data (if updated)
cd ~/aladynoulli2
./claudefile/s3-sync-script.sh download

# Run training
cd claudefile
python run_aladyn_batch.py \
  --start_index 0 \
  --end_index 10000 \
  --num_epochs 200 \
  --data_dir /home/ubuntu/aladynoulli2/data \
  --output_dir /home/ubuntu/aladynoulli2/results

# Upload results to S3
./s3-sync-script.sh upload
```

### Shut Down Instance

```bash
# In AWS Console or via CLI
aws ec2 stop-instances --instance-ids <instance-id>

# Or terminate if done
aws ec2 terminate-instances --instance-ids <instance-id>
```

**You only pay for the hours the instance was running!**

---

## Running Multiple Parallel Batches

Use the existing `run-parallel-jobs.sh` script:

```bash
# Launch larger instance (e.g., c5.9xlarge with 36 vCPUs)
# Then run parallel jobs:

./claudefile/run-parallel-jobs.sh
```

This will run multiple batches in parallel containers.

---

## Cost Breakdown

### Without AMI (Old Way)
- Instance running 24/7: ~$720-2,160/month
- Or: Setup time every launch: ~30-60 minutes

### With AMI (New Way)
- AMI storage: ~$5-10/month
- S3 data storage: ~$10-50/month depending on data size
- Compute: Only when running (e.g., 10 hours/month × $1-3/hour = $10-30/month)
- **Total: ~$25-90/month** instead of $720+

---

## Data Flow

```
Local Mac (Dropbox)
    ↓ (one-time upload)
S3 Bucket (persistent storage)
    ↓ (download when needed)
EC2 Instance (temporary compute)
    ↓ (upload results)
S3 Bucket (results)
    ↓ (download for analysis)
Local Mac
```

---

## Scripts Reference

### In `claudefile/` directory:

1. **`run_aladyn_batch.py`** - Main training script
   - Works locally (with Dropbox paths)
   - Works on AWS (with `/data` and `/results` paths)

2. **`ec2-setup.sh`** - One-time setup for fresh EC2 instance

3. **`s3-sync-script.sh`** - Data transfer to/from S3
   - `./s3-sync-script.sh download` - Get data from S3
   - `./s3-sync-script.sh upload` - Save results to S3

4. **`Dockerfile`** - Docker image with all dependencies

5. **`run-training.sh`** - Wrapper to run training in Docker container

6. **`run-parallel-jobs.sh`** - Run multiple batches in parallel

---

## Updating Your AMI

When you add new dependencies or change your code:

1. Launch instance from current AMI
2. Make changes (update code, install packages, etc.)
3. Test changes
4. Create new AMI (e.g., `aladynoulli-env-v2`)
5. Terminate instance

---

## Tips

1. **Use Spot Instances** for even cheaper compute (up to 90% off!)
   - Good for long training jobs that can handle interruptions
   - AWS will give you 2-minute warning before termination

2. **Monitor costs** with AWS Cost Explorer

3. **Tag your resources**:
   - Tag AMIs, instances with `Project: Aladynoulli`
   - Easier to track costs

4. **Automate shutdown** after training completes:
   ```bash
   # Add to end of training script
   sudo shutdown -h now
   ```

5. **Use AWS CloudWatch** to monitor:
   - CPU/GPU usage
   - Training progress (write metrics to CloudWatch Logs)

---

## Next Steps

1. ✅ Test `run_aladyn_batch.py` locally first
2. ✅ Upload data to S3 (one-time)
3. ✅ Create initial AMI with environment
4. ✅ Test launching from AMI and running job
5. ✅ Scale to larger batches and parallel jobs

---

## Questions?

- Check AWS costs: https://calculator.aws
- EC2 pricing: https://aws.amazon.com/ec2/pricing/
- S3 pricing: https://aws.amazon.com/s3/pricing/
