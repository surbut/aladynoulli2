# AWS Setup for Age Offset Predictions

This folder contains everything needed to run the age offset predictions script on AWS.

## Files Overview

- **`forAWS_offsetmasterfix.py`** - Main Python script (AWS-compatible version)
- **`run_aws.sh`** - Main runner script that orchestrates everything
- **`download_from_s3.sh`** - Downloads required data files from S3
- **`upload_to_s3.sh`** - Uploads results back to S3
- **`README.md`** - This file

## Prerequisites

### 1. AWS CLI Setup

Make sure AWS CLI is installed and configured:

```bash
# Install AWS CLI (if not already installed)
sudo apt-get update
sudo apt-get install -y awscli

# Configure AWS credentials
aws configure
```

You'll need:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)

### 2. S3 Bucket Setup

Your S3 bucket should have the following structure:

```
s3://sarah-research-aladynoulli/
├── data_for_running/
│   ├── Y_tensor.pt
│   ├── E_matrix.pt
│   ├── G_matrix.pt
│   ├── model_essentials.pt
│   ├── reference_trajectories.pt
│   ├── master_for_fitting_pooled_all_data.pt
│   └── baselinagefamh_withpcs.csv
└── results/  (will be created automatically)
```

**Your S3 bucket is already set up:** `s3://sarah-research-aladynoulli/data_for_running/`

### 3. Python Environment

Set up a conda environment with required packages:

```bash
# Create conda environment (adjust environment.yml path as needed)
conda env create -f ../aladyn_project/environment.yml

# Or create from scratch:
conda create -n new_env_pyro2 python=3.9
conda activate new_env_pyro2
pip install torch numpy pandas scipy scikit-learn matplotlib
```

### 4. Required Python Modules

The script needs these Python files to be available:

- `utils.py` - Utility functions
- `clust_huge_amp_fixedPhi.py` - Model class definition

These should be in your `PYTHONPATH` or in the same directory. The script will try to find them automatically.

## Usage

### Quick Start

1. **Make scripts executable:**
   ```bash
   cd claudefile/aws_offsetmaster
   chmod +x *.sh
   ```

2. **S3 bucket is already configured** to use `s3://sarah-research-aladynoulli`
   
   If you need to use a different bucket, you can pass it as an argument:
   ```bash
   ./run_aws.sh 0 10000 s3://different-bucket-name
   ```

3. **Run the script:**
   ```bash
   ./run_aws.sh [start_index] [end_index] [s3_bucket] [max_age_offset]
   ```

### Example Commands

**Basic usage (defaults: 0-10000, age offsets 0-10):**
```bash
./run_aws.sh
```

**Custom indices:**
```bash
./run_aws.sh 0 5000
```

**Specify different S3 bucket (default is already set):**
```bash
./run_aws.sh 0 10000 s3://different-bucket-name
```

**Full custom:**
```bash
./run_aws.sh 0 10000 s3://my-bucket-name 10
```

### Manual Steps (if you prefer)

You can also run the steps manually:

1. **Download data:**
   ```bash
   ./download_from_s3.sh s3://sarah-research-aladynoulli/data_for_running ./data_for_running
   ```

2. **Run Python script:**
   ```bash
   python forAWS_offsetmasterfix.py \
       --data_dir ./data_for_running \
       --output_dir ./output \
       --start_index 0 \
       --end_index 10000 \
       --max_age_offset 10
   ```

3. **Upload results:**
   ```bash
   ./upload_to_s3.sh ./output s3://sarah-research-aladynoulli/results my_run_name
   ```

## Output Files

The script generates:

1. **Prediction files:**
   - `pi_enroll_fixedphi_age_offset_{N}_sex_{start}_{end}_try2_withpcs_newrun.pt`
   - One for each age offset (0-10)

2. **Model checkpoints:**
   - `model_enroll_fixedphi_age_offset_{N}_sex_{start}_{end}_try2_withpcs_newrun.pt`
   - One for each age offset (0-10)

3. **Log file:**
   - `logs/run_{start}_{end}_{timestamp}.log`

All output files are automatically uploaded to S3 after completion.

## EC2 Setup (First Time)

If you're setting up a new EC2 instance:

1. **Connect to your EC2 instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

2. **Install dependencies:**
   ```bash
   # Update system
   sudo apt-get update
   sudo apt-get install -y git wget

   # Install Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b
   source ~/miniconda3/etc/profile.d/conda.sh

   # Install AWS CLI
   sudo apt-get install -y awscli
   aws configure
   ```

3. **Clone/copy this repository:**
   ```bash
   git clone <your-repo-url>
   # or
   scp -r claudefile/aws_offsetmaster ubuntu@your-ec2-ip:~/
   ```

4. **Set up Python environment** (see Prerequisites section above)

5. **Run the script:**
   ```bash
   cd ~/aws_offsetmaster  # or wherever you put it
   ./run_aws.sh
   ```

## Monitoring

While the script is running:

- **Check progress:**
  ```bash
  tail -f logs/run_*.log
  ```

- **Check if it's still running:**
  ```bash
  ps aux | grep forAWS_offsetmasterfix
  ```

- **Check output files:**
  ```bash
  ls -lh output/
  ```

## Troubleshooting

### Missing files in S3

Make sure all required files are in your S3 bucket:
```bash
aws s3 ls s3://sarah-research-aladynoulli/data_for_running/
```

### Python import errors

Make sure `utils.py` and `clust_huge_amp_fixedPhi.py` are in your Python path:
```bash
export PYTHONPATH=/path/to/pyScripts:$PYTHONPATH
```

Or copy them to the `aws_offsetmaster` directory.

### Out of memory

If you run into memory issues:
- Use smaller batch sizes (smaller `end_index - start_index`)
- Use a GPU instance if available
- Process fewer age offsets at a time

### Upload failures

Check your AWS credentials:
```bash
aws s3 ls s3://sarah-research-aladynoulli/
```

## Cost Optimization

- **Use spot instances** for cost savings
- **Download only what you need** - the script downloads all files each time
- **Clean up local files** after uploading to S3:
  ```bash
  rm -rf data_for_running/*.pt
  rm -rf output/*.pt
  ```

## Support

For issues or questions, check:
- The log files in `logs/`
- AWS CloudWatch logs (if configured)
- The main repository documentation

