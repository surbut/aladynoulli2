# Running Aladyn Predictions on EC2 - Master Checkpoint Workflow

## Overview
This guide covers running the three prediction analyses on AWS EC2:
1. **Fixed phi from retrospective (all data)** - Uses pooled phi from all retrospective batches
2. **Fixed phi from enrollment data** - Uses pooled phi from enrollment batches  
3. **Joint phi** - Uses joint estimation (separate script)

## Prerequisites
- AWS Account with EC2 access
- Your SSH key pair (.pem file)
- Data uploaded to S3: `s3://sarah-research-aladynoulli/data_for_running/`
- Master checkpoint files in S3:
  - `master_for_fitting_pooled_all_data.pt`
  - `master_for_fitting_pooled_enrollment_data.pt`

## Step-by-Step Instructions

### 1. Launch EC2 Instance

**Via AWS Console:**
1. Go to EC2 → Launch Instance
2. **Name**: `aladyn-predictions-master`
3. **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
4. **Instance type**: `r6i.4xlarge` (128 GB RAM, 16 vCPUs)
   - For faster processing: `r6i.8xlarge` (256 GB RAM, 32 vCPUs)
5. **Key pair**: Select your existing key pair
6. **Storage**: Configure 200 GB gp3 volume (for data + outputs)
7. **IAM Role**:
   - Create/select IAM role with `AmazonS3ReadOnlyAccess` policy
   - Also add `AmazonS3FullAccess` if you want to upload results
8. **Security group**: Allow SSH (port 22) from your IP
9. Click "Launch Instance"

### 2. Connect to EC2

```bash
# Wait for instance to be "running", then:
ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### 3. Upload Files from Local Machine

**From your local machine** (open a new terminal):

```bash
# Navigate to your project directory
cd /Users/sarahurbut/aladynoulli2/claudefile/aladyn_project

# Upload the scripts
scp -i /path/to/your-key.pem ec2_setup.sh ubuntu@<EC2-IP>:~/
scp -i /path/to/your-key.pem ec2_run_predictions_with_master.sh ubuntu@<EC2-IP>:~/
scp -i /path/to/your-key.pem environment.yml ubuntu@<EC2-IP>:~/

# Upload the Python scripts
scp -i /path/to/your-key.pem ../run_aladyn_predict_with_master.py ubuntu@<EC2-IP>:~/
scp -i /path/to/your-key.pem ../create_master_checkpoints.py ubuntu@<EC2-IP>:~/

# Upload the pyScripts_forPublish directory (contains model code)
scp -i /path/to/your-key.pem -r ../../pyScripts_forPublish ubuntu@<EC2-IP>:~/
```

### 4. Run Setup Script (on EC2)

```bash
# Make scripts executable
chmod +x ~/ec2_setup.sh ~/ec2_run_predictions_with_master.sh

# Run initial setup
bash ~/ec2_setup.sh

# IMPORTANT: After setup completes, logout and login again to refresh shell
exit
# Then SSH back in
ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### 5. Move Files and Create Environment (on EC2)

```bash
# Move files to project directory
mkdir -p ~/aladyn_project
mv ~/run_aladyn_predict_with_master.py ~/aladyn_project/
mv ~/create_master_checkpoints.py ~/aladyn_project/
mv ~/environment.yml ~/aladyn_project/
mv ~/pyScripts_forPublish ~/aladyn_project/
mv ~/ec2_run_predictions_with_master.sh ~/aladyn_project/

# Create conda environment
cd ~/aladyn_project
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda env create -f environment.yml

# This takes ~10-15 minutes
```

### 6. Download Data from S3 (on EC2)

```bash
# Activate environment
conda activate new_env_pyro2

# Create data directory
mkdir -p ~/aladyn_project/data_for_running

# Download all data (including master checkpoints)
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/ --no-progress

# Verify files downloaded
ls -lh ~/aladyn_project/data_for_running/
```

**Expected files:**
- `Y_tensor.pt`
- `E_enrollment_full.pt`
- `G_matrix.pt`
- `model_essentials.pt`
- `reference_trajectories.pt`
- `baselinagefamh_withpcs.csv`
- `master_for_fitting_pooled_all_data.pt` ⭐ (for retrospective analysis)
- `master_for_fitting_pooled_enrollment_data.pt` ⭐ (for enrollment analysis)

### 7. Run Predictions (on EC2)

You can run any of the three analyses:

#### Option A: Fixed Phi from Retrospective (All Data)

```bash
cd ~/aladyn_project
bash ec2_run_predictions_with_master.sh retrospective

# Or with max batches limit (for testing):
bash ec2_run_predictions_with_master.sh retrospective 10
```

#### Option B: Fixed Phi from Enrollment Data

```bash
cd ~/aladyn_project
bash ec2_run_predictions_with_master.sh enrollment

# Or with max batches limit:
bash ec2_run_predictions_with_master.sh enrollment 10
```

#### Option C: Joint Phi (Separate Script)

For joint estimation, you'll need to use the joint estimation script (not covered in this guide, but similar workflow).

### 8. Monitor Progress (on EC2)

```bash
# Watch the log file (for retrospective)
tail -f ~/aladyn_project/logs/predict_retrospective_pooled.log

# Or for enrollment
tail -f ~/aladyn_project/logs/predict_enrollment_pooled.log

# To exit tail: press Ctrl+C

# Check if still running
ps aux | grep run_aladyn_predict_with_master

# Check CPU/memory usage
htop
```

**Expected output in log:**
```
================================================================================
Aladyn Prediction Script - Fixed Phi/Psi Mode
================================================================================
Trained model: .../master_for_fitting_pooled_enrollment_data.pt
Output directory: .../output/fixedphi_enrollment_pooled
Batch size: 10000
================================================================================

Loading components...
Loaded all components successfully!
Loading master checkpoint...
BATCH 1/XX: Processing samples 0 to 10000
...
```

### 9. Run Multiple Analyses in Parallel

If you have enough resources, you can run multiple analyses simultaneously:

```bash
# Terminal 1: Retrospective
bash ec2_run_predictions_with_master.sh retrospective

# Terminal 2: Enrollment (SSH in again)
bash ec2_run_predictions_with_master.sh enrollment
```

**Note**: Make sure you have enough memory. Each analysis uses ~50-100 GB RAM.

### 10. Upload Results to S3 (on EC2)

```bash
# After script completes (check with ps aux | grep run_aladyn_predict_with_master)

# Upload retrospective results
aws s3 sync ~/aladyn_project/output/fixedphi_retrospective_pooled/ \
    s3://sarah-research-aladynoulli/predictions/fixedphi_retrospective_pooled/

# Upload enrollment results
aws s3 sync ~/aladyn_project/output/fixedphi_enrollment_pooled/ \
    s3://sarah-research-aladynoulli/predictions/fixedphi_enrollment_pooled/

# Upload logs
aws s3 sync ~/aladyn_project/logs/ \
    s3://sarah-research-aladynoulli/predictions/logs/

# Verify upload
aws s3 ls s3://sarah-research-aladynoulli/predictions/
```

### 11. Download Results to Local Machine

**From your local machine:**

```bash
# Download from S3
aws s3 sync s3://sarah-research-aladynoulli/predictions/fixedphi_retrospective_pooled/ ./local_predictions/retrospective/
aws s3 sync s3://sarah-research-aladynoulli/predictions/fixedphi_enrollment_pooled/ ./local_predictions/enrollment/

# Or download directly from EC2
scp -i /path/to/your-key.pem -r ubuntu@<EC2-IP>:~/aladyn_project/output/ ./
```

### 12. Cleanup (when done)

**Terminate EC2 instance** via AWS Console to stop charges

---

## Troubleshooting

### Issue: "Master checkpoint not found"
**Solution**: Ensure master checkpoint files are in S3:
```bash
# Check S3
aws s3 ls s3://sarah-research-aladynoulli/data_for_running/ | grep master

# If missing, upload from local:
aws s3 cp /path/to/master_for_fitting_pooled_all_data.pt \
    s3://sarah-research-aladynoulli/data_for_running/
```

### Issue: "Permission denied" when downloading from S3
**Solution**: Check IAM role is attached to instance
```bash
# On EC2, verify IAM role
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

### Issue: Out of memory during batch processing
**Solution**: 
1. Use larger instance (r6i.8xlarge)
2. Reduce batch size in script
3. Process fewer batches at a time

### Issue: Script crashes during processing
**Solution**: Check logs
```bash
# View full log
cat ~/aladyn_project/logs/predict_enrollment_pooled.log

# View last 100 lines
tail -n 100 ~/aladyn_project/logs/predict_enrollment_pooled.log
```

---

## Expected Runtime

- **Data download from S3**: 10-20 minutes (depends on file sizes)
- **Conda environment setup**: 10-15 minutes
- **Predictions per analysis**: 
  - ~10k samples/batch on `r6i.4xlarge`: ~5-10 min/batch
  - ~10k samples/batch on `r6i.8xlarge`: ~3-5 min/batch
  - Full dataset (~250k samples): ~2-4 hours per analysis

---

## Cost Estimates (us-east-1)

- **r6i.4xlarge**: ~$1.01/hour
- **r6i.8xlarge**: ~$2.02/hour
- **Storage (200 GB)**: ~$0.40/day
- **Data transfer**: Minimal (data in same region)

**Example**: Running all 3 analyses for 4 hours on r6i.4xlarge ≈ $12-15 total

---

## Quick Reference Commands

```bash
# Setup (one time)
bash ~/ec2_setup.sh
# Logout and login, then:
conda env create -f ~/aladyn_project/environment.yml
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/

# Run analyses
cd ~/aladyn_project
bash ec2_run_predictions_with_master.sh retrospective
bash ec2_run_predictions_with_master.sh enrollment

# Monitor
tail -f ~/aladyn_project/logs/predict_enrollment_pooled.log
ps aux | grep run_aladyn_predict_with_master

# Upload results
aws s3 sync ~/aladyn_project/output/ s3://sarah-research-aladynoulli/predictions/
```

