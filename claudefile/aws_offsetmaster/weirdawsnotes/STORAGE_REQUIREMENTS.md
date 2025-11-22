# Storage Requirements for EC2 Instance

## Storage Breakdown

### Input Files (Downloaded from S3, kept on disk)

These files are downloaded once and stay on the instance:

| File | Estimated Size | Notes |
|------|---------------|-------|
| `Y_tensor.pt` | ~5-20 GB | Large tensor (N × D × T) |
| `E_matrix.pt` | ~1-5 GB | Censoring matrix (N × D) |
| `G_matrix.pt` | ~1-5 GB | Genetic matrix (N × P) |
| `model_essentials.pt` | ~50-200 MB | Small config file |
| `reference_trajectories.pt` | ~100-500 MB | Reference data |
| `master_for_fitting_pooled_all_data.pt` | ~100-500 MB | Master checkpoint |
| `baselinagefamh_withpcs.csv` | ~50-200 MB | CSV file |
| **Total Input Files** | **~8-32 GB** | Downloaded once |

### Output Files (Generated per batch)

For **each batch** (0-10000 indices), you get:
- **11 prediction files** (`pi_*_age_offset_0-10_*.pt`)
- **11 model checkpoint files** (`model_*_age_offset_0-10_*.pt`)
- Total: **22 files per batch**

**For 10 batches:**
- Total output files: **10 batches × 22 files = 220 files**

#### Output File Size Estimates

**Per batch output size:**
- Each prediction file: ~10-50 MB (depending on batch size)
- Each model checkpoint: ~50-200 MB (contains model state)
- Per batch total: ~1.3-5.5 GB

**10 batches total output:**
- **~13-55 GB** for all output files

**Important:** Output files are uploaded to S3 as they're created, so they don't all accumulate if uploads work properly. However, you should have enough space for at least 1-2 batches worth of outputs while they're being uploaded.

### Log Files

- **Log files:** ~10-100 MB per batch
- **10 batches:** ~100 MB - 1 GB total

### System and Software

- **Ubuntu OS:** ~3-5 GB
- **Conda/Python environment:** ~2-5 GB
- **Code files:** ~100-500 MB
- **System overhead:** ~5-10 GB
- **Total system:** ~10-20 GB

## Total Storage Needed

### Minimum Storage (Conservative Estimate)

| Component | Size |
|-----------|------|
| Input files | ~32 GB |
| Output files (peak, while uploading) | ~10 GB |
| Log files | ~1 GB |
| System/OS | ~20 GB |
| **Total Minimum** | **~63 GB** |

### Recommended Storage

**For 10 batches, I recommend: 100-200 GB**

Why:
- **Safe buffer** for large input files
- **Room for 2-3 batches** of outputs while uploading to S3
- **System overhead** and temporary files
- **Extra space** in case downloads/uploads are delayed

## EC2 Configuration

### During Launch

In the EC2 Launch Instance page:

1. **Step 6: Configure Storage**
2. **Root volume:**
   - Size: **100-200 GB** (recommended: **150 GB**)
   - Volume type: `gp3` (General Purpose SSD)
   - IOPS: Default (3000)
   - Throughput: Default (125 MB/s)

3. **Optional:** Add additional EBS volume if you want more
   - But 100-200 GB should be plenty

### Cost of Storage

- **gp3 storage:** ~$0.08-0.10 per GB per month
- **150 GB for ~1 day:** ~$0.005 (basically free)
- **150 GB for 1 month:** ~$12-15

**Storage cost is minimal** compared to instance compute cost ($6/hour).

## Storage Management Tips

### 1. Clean Up After Each Batch

The script uploads results to S3 automatically. You can optionally clean up after upload:

```bash
# After each batch completes, remove local output files (they're in S3)
rm -f output/*.pt
```

**Note:** The script already uploads to S3, so you can delete local files after verification.

### 2. Monitor Disk Space

```bash
# Check disk usage
df -h

# Check size of specific directories
du -sh data_for_running/
du -sh output/
du -sh logs/

# Watch disk space in real-time
watch -n 5 df -h
```

### 3. If Running Out of Space

If you're running low on space:

```bash
# Remove old output files (if already uploaded to S3)
rm -f output/*.pt

# Remove old log files (optional)
rm -f logs/run_*.log

# Remove input files (if you're done with all batches)
rm -rf data_for_running/*.pt
```

### 4. Upload Results Immediately

The `run_aws.sh` script uploads results after completion. Make sure uploads are working:

```bash
# Check if files are in S3
aws s3 ls s3://sarah-research-aladynoulli/results/ --recursive
```

## Recommendations Summary

### For 10 Batches on c7i.24xlarge:

✅ **Recommended:** **150 GB** root volume
- Plenty of space for inputs (~32 GB)
- Room for outputs while uploading (~10-20 GB)
- System overhead (~20 GB)
- Safety buffer

✅ **Minimum:** **100 GB** 
- Might be tight, but should work
- Need to clean up output files more aggressively

✅ **Optimal:** **200 GB**
- More comfortable
- Can keep files longer if needed

## Quick Storage Check

Before starting your batches:

```bash
# Check available space
df -h

# Should show something like:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/xvda1      150G   20G  130G  14% /
```

Make sure you have at least **50-100 GB free** before starting.

## Storage Cost vs Instance Cost

**Storage is CHEAP compared to compute:**
- **150 GB storage for 1 day:** ~$0.005 (negligible)
- **c7i.24xlarge for 1 hour:** ~$6.25
- **c7i.24xlarge for 20 hours:** ~$125

**Recommendation:** Don't skimp on storage to save pennies - it's worth having extra space to avoid issues.




