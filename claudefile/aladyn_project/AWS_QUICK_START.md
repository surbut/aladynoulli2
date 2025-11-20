# AWS EC2 Quick Start Guide

## Prerequisites Checklist
- [ ] Master checkpoint files uploaded to S3:
  - `master_for_fitting_pooled_all_data.pt`
  - `master_for_fitting_pooled_enrollment_data.pt`
- [ ] All data files in S3: `s3://sarah-research-aladynoulli/data_for_running/`
- [ ] EC2 instance launched with IAM role (S3 access)

## One-Time Setup (First Time Only)

### 1. Launch EC2 Instance
- Type: `r6i.4xlarge` or `r6i.8xlarge`
- Storage: 200 GB
- IAM Role: S3 read/write access
- Security Group: SSH from your IP

### 2. Upload Files to EC2

**From your local machine:**
```bash
cd /Users/sarahurbut/aladynoulli2/claudefile/aladyn_project

# Set your EC2 IP and key path
export EC2_IP="<your-ec2-ip>"
export EC2_KEY="/path/to/your-key.pem"

# Upload files
scp -i $EC2_KEY ec2_setup.sh ubuntu@$EC2_IP:~/
scp -i $EC2_KEY ec2_run_predictions_with_master.sh ubuntu@$EC2_IP:~/
scp -i $EC2_KEY environment.yml ubuntu@$EC2_IP:~/
scp -i $EC2_KEY ../run_aladyn_predict_with_master.py ubuntu@$EC2_IP:~/
scp -i $EC2_KEY -r ../../pyScripts_forPublish ubuntu@$EC2_IP:~/
```

### 3. Setup on EC2

**SSH into EC2:**
```bash
ssh -i $EC2_KEY ubuntu@$EC2_IP
```

**Run setup:**
```bash
chmod +x ~/ec2_setup.sh ~/ec2_run_predictions_with_master.sh
bash ~/ec2_setup.sh

# Logout and login again
exit
ssh -i $EC2_KEY ubuntu@$EC2_IP
```

**Move files and create environment:**
```bash
mkdir -p ~/aladyn_project
mv ~/run_aladyn_predict_with_master.py ~/aladyn_project/
mv ~/environment.yml ~/aladyn_project/
mv ~/pyScripts_forPublish ~/aladyn_project/
mv ~/ec2_run_predictions_with_master.sh ~/aladyn_project/

cd ~/aladyn_project
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda env create -f environment.yml  # Takes ~10-15 min
```

**Download data:**
```bash
conda activate new_env_pyro2
mkdir -p ~/aladyn_project/data_for_running
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/
```

## Running Analyses

### Fixed Phi from Enrollment Data
```bash
cd ~/aladyn_project
bash ec2_run_predictions_with_master.sh enrollment
```

### Fixed Phi from Retrospective (All Data)
```bash
cd ~/aladyn_project
bash ec2_run_predictions_with_master.sh retrospective
```

### With Batch Limit (for testing)
```bash
bash ec2_run_predictions_with_master.sh enrollment 10  # Only 10 batches
```

## Monitoring

```bash
# Watch logs
tail -f ~/aladyn_project/logs/predict_enrollment_pooled.log

# Check if running
ps aux | grep run_aladyn_predict_with_master

# Check resource usage
htop
```

## Upload Results

```bash
# Upload to S3
aws s3 sync ~/aladyn_project/output/fixedphi_enrollment_pooled/ \
    s3://sarah-research-aladynoulli/predictions/fixedphi_enrollment_pooled/

aws s3 sync ~/aladyn_project/output/fixedphi_retrospective_pooled/ \
    s3://sarah-research-aladynoulli/predictions/fixedphi_retrospective_pooled/
```

## Download to Local

```bash
# From your local machine
aws s3 sync s3://sarah-research-aladynoulli/predictions/fixedphi_enrollment_pooled/ \
    ./local_predictions/enrollment/
```

## Troubleshooting

**Master checkpoint not found:**
```bash
# Check S3
aws s3 ls s3://sarah-research-aladynoulli/data_for_running/ | grep master

# Upload if missing (from local)
aws s3 cp /path/to/master_for_fitting_pooled_all_data.pt \
    s3://sarah-research-aladynoulli/data_for_running/
```

**Out of memory:**
- Use larger instance (r6i.8xlarge)
- Or reduce batch size in script

**Check logs:**
```bash
tail -n 100 ~/aladyn_project/logs/predict_enrollment_pooled.log
```

