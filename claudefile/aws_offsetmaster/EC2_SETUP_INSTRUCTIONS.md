# EC2 Setup Instructions for Joint Phi Estimation

## Step 1: Download Scripts from S3 (Easier!)

Since scripts are already in S3 at `s3://sarah-research-aladynoulli/scripts/`, download them on EC2:

```bash
# Download scripts from S3
aws s3 sync s3://sarah-research-aladynoulli/scripts/ ~/scripts/

# Create directory structure
mkdir -p ~/pyScripts_forPublish

# Copy clust_huge_amp.py to the right location
cp ~/scripts/clust_huge_amp.py ~/pyScripts_forPublish/ 2>/dev/null || \
cp ~/scripts/pyScripts_forPublish/clust_huge_amp.py ~/pyScripts_forPublish/ 2>/dev/null || \
echo "⚠️  clust_huge_amp.py not found in scripts, will need to upload manually"

# Copy run_aladyn_batch.py
cp ~/scripts/run_aladyn_batch.py ~/ 2>/dev/null || \
echo "⚠️  run_aladyn_batch.py not found in scripts, will need to upload manually"

# Verify
ls ~/pyScripts_forPublish/clust_huge_amp.py
ls ~/run_aladyn_batch.py
```

**Alternative: Upload from Local Machine**

If scripts aren't in S3, upload from local:

```bash
# From local machine
scp -i sarahkey.pem claudefile/aws_offsetmaster/run_aladyn_batch.py ubuntu@ec2-98-83-212-179.compute-1.amazonaws.com:~/
scp -i sarahkey.pem pyScripts_forPublish/clust_huge_amp.py ubuntu@ec2-98-83-212-179.compute-1.amazonaws.com:~/

# Then on EC2:
mkdir -p ~/pyScripts_forPublish
mv ~/clust_huge_amp.py ~/pyScripts_forPublish/
```

## Step 3: Run Setup Script

```bash
# Make script executable
chmod +x setup_and_run_ec2.sh

# Run for batch 0-10000
bash setup_and_run_ec2.sh 0 10000
```

## Step 4: Run Additional Batches (Optional)

To run 2 batches in parallel:

```bash
# Terminal 1
bash setup_and_run_ec2.sh 0 10000 > batch0.log 2>&1 &

# Terminal 2 (or same terminal)
bash setup_and_run_ec2.sh 10000 20000 > batch1.log 2>&1 &

# Wait for both
wait
```

## Files Needed on EC2

- `~/run_aladyn_batch.py` - The batch script
- `~/pyScripts_forPublish/clust_huge_amp.py` - The model class
- `~/setup_and_run_ec2.sh` - Setup script (already uploaded)

That's it! The script will download everything else from S3.

