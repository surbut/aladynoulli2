# Instructions for Backing Up Results to S3 and Shutting Down EC2

## On EC2 Instance

### 1. Update the backup script with your paths:
```bash
# Edit the script
nano backup_results_to_s3.sh

# Update these variables:
RESULTS_DIR="/home/ubuntu/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results"
S3_BUCKET="your-bucket-name"  # e.g., "aladynoulli-results"
```

### 2. Make script executable and run:
```bash
chmod +x backup_results_to_s3.sh
./backup_results_to_s3.sh
```

### 3. Alternative: Manual commands

#### Create tar archive:
```bash
cd /home/ubuntu/aladynoulli2/pyScripts/new_oct_revision/new_notebooks
tar -czf results_backup_$(date +%Y%m%d_%H%M%S).tar.gz results/
```

#### Upload to S3:
```bash
# Set your bucket name
BUCKET="your-bucket-name"
ARCHIVE="results_backup_YYYYMMDD_HHMMSS.tar.gz"

# Upload
aws s3 cp ${ARCHIVE} s3://${BUCKET}/results/${ARCHIVE}

# Verify upload
aws s3 ls s3://${BUCKET}/results/
```

#### Shut down EC2:
```bash
# Option 1: Shutdown (stops instance)
sudo shutdown -h now

# Option 2: Stop instance via AWS CLI (if you have instance ID)
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx

# Option 3: Terminate instance (DESTRUCTIVE - cannot restart)
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```

## On Local Machine

### Download from S3:

#### Option 1: Using the download script
```bash
# Edit script with your bucket and key
nano download_results_from_s3.sh

# Run it
chmod +x download_results_from_s3.sh
./download_results_from_s3.sh
```

#### Option 2: Manual download
```bash
# List available files
aws s3 ls s3://your-bucket-name/results/

# Download specific file
aws s3 cp s3://your-bucket-name/results/results_backup_YYYYMMDD_HHMMSS.tar.gz ~/Downloads/

# Extract
cd ~/Downloads
tar -xzf results_backup_YYYYMMDD_HHMMSS.tar.gz
```

## Quick Reference Commands

### On EC2:
```bash
# Create archive
cd /path/to/results/parent
tar -czf results_backup.tar.gz results/

# Upload
aws s3 cp results_backup.tar.gz s3://your-bucket/results/

# Shutdown
sudo shutdown -h now
```

### On Local:
```bash
# Download
aws s3 cp s3://your-bucket/results/results_backup.tar.gz ~/Downloads/

# Extract
cd ~/Downloads && tar -xzf results_backup.tar.gz
```

## Notes

- **S3 Bucket**: Make sure you have an S3 bucket created and proper IAM permissions
- **Archive Size**: Large archives (>5GB) may take time to upload/download
- **Cost**: S3 storage is cheap, but data transfer out of S3 has costs
- **EC2 Shutdown**: Use `stop-instances` to preserve the instance, `terminate-instances` to delete it permanently


