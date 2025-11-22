# Quick Start Guide

## Your S3 Bucket
**Bucket:** `s3://sarah-research-aladynoulli/data_for_running/`

All files from your Dropbox `data_for_running` folder should be in this S3 bucket.

## First Time Setup

**If you haven't set up an EC2 instance yet, start here:**

👉 **Instance Type:** **`c7i.24xlarge`** (see `INSTANCE_RECOMMENDATION.md`)

👉 **Storage:** **150 GB** root volume (see `STORAGE_REQUIREMENTS.md`)

👉 **Setup Guide:** See `EC2_SETUP_CONSOLE.md` for step-by-step instructions on launching an instance from the AWS Console.

## Quick Run

### Single Batch

1. **Make scripts executable** (if not already):
   ```bash
   cd claudefile/aws_offsetmaster
   chmod +x *.sh
   ```

2. **Run with defaults** (0-10000, age offsets 0-10):
   ```bash
   ./run_aws.sh
   ```

3. **Or customize**:
   ```bash
   ./run_aws.sh [start_index] [end_index] [s3_bucket] [max_age_offset]
   ```

   Example:
   ```bash
   ./run_aws.sh 0 5000 s3://sarah-research-aladynoulli 10
   ```

### Multiple Batches (10 batches total)

**Run all 10 batches automatically:**
```bash
./run_all_batches.sh
```

This will run:
- Batch 1: 0-10000
- Batch 2: 10000-20000
- Batch 3: 20000-30000
- ... and so on
- Batch 10: 90000-100000

**Estimated total time:** ~5-20 hours on c7i.24xlarge
**Estimated total cost:** ~$30-120

See `MULTI_BATCH_SETUP.md` for details.

## What It Does

1. Downloads all required files from S3 to `./data_for_running/`
2. Runs the prediction script for each age offset (0-10 years)
3. Saves results to `./output/`
4. Uploads all results back to S3: `s3://sarah-research-aladynoulli/results/[run_name]/`

## Required Files in S3

Make sure these are in `s3://sarah-research-aladynoulli/data_for_running/`:
- `Y_tensor.pt`
- `E_matrix.pt`
- `G_matrix.pt`
- `model_essentials.pt`
- `reference_trajectories.pt`
- `master_for_fitting_pooled_all_data.pt`
- `baselinagefamh_withpcs.csv`

## Output Files

For each age offset (0-10), you'll get:
- `pi_enroll_fixedphi_age_offset_{N}_sex_{start}_{end}_try2_withpcs_newrun.pt`
- `model_enroll_fixedphi_age_offset_{N}_sex_{start}_{end}_try2_withpcs_newrun.pt`

## Check S3 Files

```bash
aws s3 ls s3://sarah-research-aladynoulli/data_for_running/
```

## Monitor Progress

```bash
tail -f logs/run_*.log
```

## See Full Documentation

See `README.md` for detailed instructions and troubleshooting.

