# AWS vs Local Workflow Guide

## Strategy: Hybrid Approach

**AWS**: Heavy computation (generating pi tensors)  
**Local**: Analysis and visualization (interactive notebooks)

---

## 🔄 Workflow

### Step 1: Generate Pi Tensors on AWS

**On EC2**, run prediction scripts that generate pi tensors:

```bash
# On EC2
cd ~/aladyn_project

# 1. Washout predictions (if not already done)
nohup python scripts/generate_washout_predictions.py \
    --approach pooled_retrospective \
    --washout_years 0 1 2 \
    > washout_predictions.log 2>&1 &

# 2. Age offset predictions (if not already done)
nohup python scripts/forAWS_offsetmasterfix.py \
    --data_dir data_for_running/ \
    --output_dir output/ \
    --start_index 0 \
    --end_index 10000 \
    --max_age_offset 9 \
    > age_offset_predictions.log 2>&1 &

# 3. Full pi tensor (if needed)
# This generates pi_full_400k.pt by assembling batches
```

**Outputs on AWS**:
- `output/pi_enroll_fixedphi_sex_*.pt` (washout)
- `output/pi_enroll_fixedphi_age_offset_*_sex_*.pt` (age offset)
- `output/pi_full_400k.pt` (full tensor)

---

### Step 2: Download Results to Local

**On Local**, sync results from AWS:

```bash
# Option 1: rsync (recommended)
rsync -avz --progress \
    ec2-user@your-ec2-ip:~/aladyn_project/output/ \
    ~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/

# Option 2: scp specific files
scp ec2-user@your-ec2-ip:~/aladyn_project/output/pi_enroll_fixedphi_sex_*.pt \
    ~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/

# Option 3: Use Dropbox sync (if AWS has Dropbox)
# Files automatically sync if both have Dropbox installed
```

**Local destination**:
- `~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/`
- Or wherever your notebooks expect pi tensors

---

### Step 3: Run Analysis Locally

**On Local**, run notebooks that use downloaded pi tensors:

```python
# In notebooks, update paths to point to downloaded files
pi_path = Path('~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/pi_enroll_fixedphi_sex_0_10000.pt')
```

**Notebooks to run locally**:
- `R2_Temporal_Leakage.ipynb` - Loads washout results
- `R1_Q2_Lifetime_Risk.ipynb` - Loads age offset results
- `R1_Q9_AUC_Comparisons.ipynb` - Loads pi tensors for comparison
- `performancen_notebook_clean.ipynb` - Main performance notebook

---

## 📁 Recommended Directory Structure

### On AWS
```
~/aladyn_project/
├── scripts/
│   ├── generate_washout_predictions.py
│   ├── forAWS_offsetmasterfix.py
│   └── run_aladyn_predict_with_master.py
├── output/
│   ├── pi_enroll_fixedphi_sex_*.pt
│   ├── pi_enroll_fixedphi_age_offset_*_sex_*.pt
│   └── pi_full_400k.pt
└── logs/
    └── *.log
```

### On Local
```
~/Library/CloudStorage/Dropbox-Personal/
├── predictions_from_aws/          # Synced from AWS
│   ├── pi_enroll_fixedphi_sex_*.pt
│   ├── pi_enroll_fixedphi_age_offset_*_sex_*.pt
│   └── pi_full_400k.pt
└── (your existing data directories)

~/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/
├── R2_Temporal_Leakage.ipynb      # Uses downloaded pi tensors
├── R1_Q2_Lifetime_Risk.ipynb      # Uses downloaded pi tensors
└── ...
```

---

## 🔧 Helper Scripts

### Script 1: Sync from AWS

Create `sync_from_aws.sh`:

```bash
#!/bin/bash
# sync_from_aws.sh

AWS_HOST="ec2-user@your-ec2-ip"
AWS_DIR="~/aladyn_project/output"
LOCAL_DIR="~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws"

echo "Syncing pi tensors from AWS..."
rsync -avz --progress \
    ${AWS_HOST}:${AWS_DIR}/pi_enroll_fixedphi_sex_*.pt \
    ${AWS_HOST}:${AWS_DIR}/pi_enroll_fixedphi_age_offset_*_sex_*.pt \
    ${AWS_HOST}:${AWS_DIR}/pi_full_400k.pt \
    ${LOCAL_DIR}/

echo "✅ Sync complete!"
```

### Script 2: Check What's Available

Create `check_pi_tensors.py`:

```python
# check_pi_tensors.py
from pathlib import Path
import glob

aws_dir = Path('~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws').expanduser()

print("Checking for pi tensors...")
print("="*80)

# Check washout
washout_files = list(aws_dir.glob('pi_enroll_fixedphi_sex_*.pt'))
print(f"Washout pi tensors: {len(washout_files)}")
for f in sorted(washout_files)[:5]:
    print(f"  {f.name}")

# Check age offset
offset_files = list(aws_dir.glob('pi_enroll_fixedphi_age_offset_*_sex_*.pt'))
print(f"\nAge offset pi tensors: {len(offset_files)}")
for f in sorted(offset_files)[:5]:
    print(f"  {f.name}")

# Check full tensor
full_tensor = aws_dir / 'pi_full_400k.pt'
print(f"\nFull pi tensor: {'✅ Found' if full_tensor.exists() else '❌ Missing'}")
```

---

## ⚠️ Considerations

### Pros of AWS
- ✅ More compute power
- ✅ Can run multiple batches in parallel
- ✅ Don't tie up local machine
- ✅ Better for large-scale runs

### Cons of AWS
- ❌ Can't use Cursor (no interactive editing)
- ❌ Need to sync files back
- ❌ Harder to debug interactively

### Pros of Local
- ✅ Use Cursor for interactive development
- ✅ Immediate results
- ✅ Easier debugging
- ✅ Can iterate quickly

### Cons of Local
- ❌ Limited by local machine resources
- ❌ May be slower for large batches
- ❌ Ties up your machine

---

## 🎯 Recommended Approach

**For this revision**:

1. **AWS**: Generate all pi tensors (washout, age offset, full tensor)
   - Use `nohup` and let run in background
   - Check logs periodically

2. **Local**: Run all analysis notebooks
   - Sync pi tensors from AWS
   - Update notebook paths to point to synced files
   - Run interactively in Cursor

3. **Hybrid**: If you need to debug prediction generation
   - Run small test batch locally first
   - Then scale up on AWS

---

## 📝 Quick Commands

```bash
# On AWS: Check if predictions are running
ps aux | grep python | grep -E "(washout|offset|predict)"

# On AWS: Check output directory
ls -lh ~/aladyn_project/output/pi_*.pt

# On Local: Sync from AWS
rsync -avz ec2-user@your-ec2-ip:~/aladyn_project/output/pi_*.pt \
    ~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/

# On Local: Check what you have
ls -lh ~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/pi_*.pt
```

---

## 🔄 Update Notebook Paths

After syncing, update notebook paths:

```python
# Old (if pointing to AWS directly - won't work)
# pi_path = Path('/path/on/aws/pi_enroll_fixedphi_sex_0_10000.pt')

# New (pointing to synced local copy)
pi_path = Path('~/Library/CloudStorage/Dropbox-Personal/predictions_from_aws/pi_enroll_fixedphi_sex_0_10000.pt').expanduser()
```

---

**Bottom Line**: Generate pi tensors on AWS, sync to local, run notebooks locally in Cursor. Best of both worlds! 🚀

