# Simple AWS Setup - 3 Steps

## If you already have an EC2 instance with the environment set up:

### Step 1: Upload files (one time)
```bash
# From your local machine
cd /Users/sarahurbut/aladynoulli2/claudefile/aladyn_project
scp -i ~/.ssh/your-key.pem run_aladyn_predict_with_master.py ubuntu@<EC2-IP>:~/aladyn_project/
scp -i ~/.ssh/your-key.pem aws_simple_run.sh ubuntu@<EC2-IP>:~/aladyn_project/
scp -i ~/.ssh/your-key.pem -r ../../pyScripts_forPublish ubuntu@<EC2-IP>:~/aladyn_project/
```

### Step 2: Run predictions
```bash
# SSH into EC2
ssh -i ~/.ssh/your-key.pem ubuntu@<EC2-IP>

# Make script executable
chmod +x ~/aladyn_project/aws_simple_run.sh

# Run enrollment analysis
cd ~/aladyn_project
bash aws_simple_run.sh enrollment

# Or retrospective
bash aws_simple_run.sh retrospective

# Or with batch limit (for testing)
bash aws_simple_run.sh enrollment 10
```

### Step 3: Monitor & download
```bash
# Watch logs
tail -f ~/aladyn_project/logs/enrollment.log

# When done, upload to S3 (optional)
aws s3 sync ~/aladyn_project/output/ s3://sarah-research-aladynoulli/predictions/
```

That's it!

---

## If you need to set up a new EC2 instance:

### Option A: Clone existing instance (easiest)
1. Go to EC2 Console
2. Find your existing instance
3. Right-click → **Image → Create Image**
4. Once image is ready, **Launch Instance from Template**
5. SSH in and run Step 2 above

### Option B: Fresh instance (5 commands)
```bash
# 1. SSH into new EC2 instance
ssh -i ~/.ssh/your-key.pem ubuntu@<EC2-IP>

# 2. Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# 3. Create environment (download environment.yml first)
conda env create -f environment.yml

# 4. Upload files (from local machine - see Step 1 above)

# 5. Run (see Step 2 above)
```

---

## That's really all you need!

The script automatically:
- Downloads data from S3
- Runs the right analysis
- Saves logs
- Everything in background

No complicated setup needed!

