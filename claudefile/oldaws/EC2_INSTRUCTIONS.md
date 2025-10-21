# Running Aladyn Predictions on EC2 - Complete Guide

## Prerequisites
- AWS Account with EC2 access
- Your SSH key pair (.pem file)
- Data uploaded to S3: `s3://sarah-research-aladynoulli/data_for_running/`

## Step-by-Step Instructions

### 1. Launch EC2 Instance

**Via AWS Console:**
1. Go to EC2 → Launch Instance
2. **Name**: `aladyn-predictions`
3. **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
4. **Instance type**: `r6i.4xlarge` (128 GB RAM, 16 vCPUs)
   - For faster processing, use `r6i.8xlarge` (256 GB RAM, 32 vCPUs)
5. **Key pair**: Select your existing key pair
6. **Storage**: Configure 150 GB gp3 volume
7. **IAM Role**:
   - Click "Create new IAM role" → EC2 → Next
   - Attach policy: `AmazonS3ReadOnlyAccess`
   - Name: `EC2-S3-ReadOnly`
   - Select this role
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
cd /Users/sarahurbut/aladynoulli2/claudefile

# Upload the scripts and code
scp -i /path/to/your-key.pem ec2_setup.sh ubuntu@<EC2-IP>:~/
scp -i /path/to/your-key.pem ec2_run_predictions.sh ubuntu@<EC2-IP>:~/
scp -i /path/to/your-key.pem run_aladyn_predict.py ubuntu@<EC2-IP>:~/
scp -i /path/to/your-key.pem environment.yml ubuntu@<EC2-IP>:~/

# Upload the pyScripts_forPublish directory
scp -i /path/to/your-key.pem -r ../pyScripts_forPublish ubuntu@<EC2-IP>:~/
```

### 4. Run Setup Script (on EC2)

```bash
# Make scripts executable
chmod +x ~/ec2_setup.sh ~/ec2_run_predictions.sh

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
mv ~/run_aladyn_predict.py ~/aladyn_project/
mv ~/environment.yml ~/aladyn_project/
mv ~/pyScripts_forPublish ~/aladyn_project/

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

# Download all data
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/ --no-progress

# Verify files downloaded (should see ~6 files)
ls -lh ~/aladyn_project/data_for_running/
```

**Expected files:**
- `Y_tensor.pt`
- `E_enrollment_full.pt`
- `G_matrix.pt`
- `model_essentials.pt`
- `reference_trajectories.pt`
- `baselinagefamh_withpcs.csv`
- `enrollment_model_W0.0001_fulldata_sexspecific.pt` (trained model)

### 7. Run Predictions (on EC2)

```bash
# Run the prediction script
bash ~/ec2_run_predictions.sh

# You should see:
# "Prediction script started with PID: XXXXX"
```

### 8. Monitor Progress (on EC2)

```bash
# Watch the log file
tail -f ~/aladyn_project/logs/predict.log

# To exit tail: press Ctrl+C

# Check if still running
ps aux | grep run_aladyn_predict

# Check GPU/CPU usage (if curious)
htop
```

**Expected output in log:**
```
================================================================================
Aladyn Prediction Script - Fixed Phi/Psi Mode
================================================================================
Trained model: .../enrollment_model_W0.0001_fulldata_sexspecific.pt
Output directory: .../output/
Batch size: 10000
================================================================================

Loading components...
Loaded all components successfully!
...
BATCH 1/XX: Processing samples 0 to 10000
...
```

### 9. Upload Results to S3 (on EC2)

```bash
# After script completes (check with ps aux | grep run_aladyn_predict)
# Upload results
aws s3 sync ~/aladyn_project/output/ s3://sarah-research-aladynoulli/predictions/

# Verify upload
aws s3 ls s3://sarah-research-aladynoulli/predictions/
```

### 10. Download Results to Local Machine

**From your local machine:**

```bash
# Download from S3
aws s3 sync s3://sarah-research-aladynoulli/predictions/ ./local_predictions/

# Or download directly from EC2
scp -i /path/to/your-key.pem -r ubuntu@<EC2-IP>:~/aladyn_project/output/ ./
```

### 11. Cleanup (when done)

**Terminate EC2 instance** via AWS Console to stop charges

---

## Troubleshooting

### Issue: "Permission denied" when downloading from S3
**Solution**: Check IAM role is attached to instance
```bash
# On EC2, verify IAM role
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

### Issue: Out of memory during batch processing
**Solution**: Reduce batch size
```bash
# Edit the script to use smaller batch size
nano ~/ec2_run_predictions.sh
# Change: --batch_size 10000 → --batch_size 5000
```

### Issue: conda environment creation fails
**Solution**:
```bash
# Try updating conda first
conda update -n base conda
conda env create -f environment.yml
```

### Issue: Script crashes during processing
**Solution**: Check logs
```bash
# View full log
cat ~/aladyn_project/logs/predict.log

# View last 100 lines
tail -n 100 ~/aladyn_project/logs/predict.log
```

---

## Expected Runtime

- **Data download from S3**: 5-15 minutes (depends on file sizes)
- **Conda environment setup**: 10-15 minutes
- **Predictions**: Varies by dataset size
  - ~10k samples/batch on `r6i.4xlarge`: ~5-10 min/batch
  - ~10k samples/batch on `r6i.8xlarge`: ~3-5 min/batch

---

## Cost Estimates (us-east-1)

- **r6i.4xlarge**: ~$1.01/hour
- **r6i.8xlarge**: ~$2.02/hour
- **Storage (150 GB)**: ~$0.30/day
- **Data transfer**: Minimal (data in same region)

**Example**: Running for 6 hours on r6i.4xlarge ≈ $6-7 total
