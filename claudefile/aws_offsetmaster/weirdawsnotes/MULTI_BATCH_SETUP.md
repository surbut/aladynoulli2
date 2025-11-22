# Multi-Batch Setup - Running 10 Batches

You're planning to run **10 batches** total on `c7i.24xlarge`.

## Batch Strategy

Each batch processes a subset of your data:
- **Batch 1:** indices 0-10000
- **Batch 2:** indices 10000-20000
- **Batch 3:** indices 20000-30000
- ... and so on
- **Batch 10:** indices 90000-100000

Each batch:
- Processes 11 age offsets (0-10 years)
- Runs on `c7i.24xlarge` (~30-60 minutes per batch)
- Produces 11 prediction files + 11 model files per batch

## Total Cost Estimates

### Your Instance: `c7i.24xlarge`

**Per Batch:**
- Runtime: ~30-60 minutes
- Cost: ~$3-6 per batch

**10 Batches Total:**
- Total Runtime: ~5-10 hours (or up to 20 hours if longer)
- Total Cost: **~$30-60** (or up to **~$120** if 20 hours total)

**Budget:** "A few hundred dollars" is plenty - you're well within budget!

## Running Multiple Batches

### Option 1: Sequential (One at a Time)

Run batches one after another:

```bash
# Batch 1: 0-10000
./run_aws.sh 0 10000 s3://sarah-research-aladynoulli 10

# Batch 2: 10000-20000
./run_aws.sh 10000 20000 s3://sarah-research-aladynoulli 10

# Batch 3: 20000-30000
./run_aws.sh 20000 30000 s3://sarah-research-aladynoulli 10

# ... and so on
```

**Pros:** Simple, easy to monitor
**Cons:** Takes longer (but with c7i.24xlarge, still fast!)

### Option 2: Automated Loop Script

Create a script to run all batches automatically:

```bash
#!/bin/bash
# run_all_batches.sh

BATCH_SIZE=10000
START_IDX=0
END_IDX=100000
S3_BUCKET="s3://sarah-research-aladynoulli"

for i in {0..9}; do
    BATCH_START=$((i * BATCH_SIZE))
    BATCH_END=$(((i + 1) * BATCH_SIZE))
    
    echo "=========================================="
    echo "Running Batch $((i+1)): $BATCH_START-$BATCH_END"
    echo "=========================================="
    
    ./run_aws.sh $BATCH_START $BATCH_END $S3_BUCKET 10
    
    echo "Batch $((i+1)) complete!"
    echo ""
    
    # Optional: Add a small delay between batches
    sleep 10
done

echo "All batches complete!"
```

### Option 3: Using nohup (Background Execution)

Run each batch in the background so you can disconnect:

```bash
# Run batch 1 in background
nohup ./run_aws.sh 0 10000 s3://sarah-research-aladynoulli 10 > batch_1.log 2>&1 &

# Run batch 2 in background
nohup ./run_aws.sh 10000 20000 s3://sarah-research-aladynoulli 10 > batch_2.log 2>&1 &

# Monitor progress
tail -f batch_1.log
```

**Note:** Be careful with background execution - make sure you have enough resources!

## Cost Optimization

With a budget of "a few hundred dollars":

1. **Run all batches on same instance** - Don't terminate between batches
   - Saves setup time
   - Single instance for all 10 batches
   - Total cost: ~$30-120 depending on total runtime

2. **Monitor progress closely**
   - Watch logs: `tail -f logs/run_*.log`
   - Check batch completion
   - Verify S3 uploads after each batch

3. **Terminate when ALL batches complete**
   - Don't leave the instance running after all batches are done
   - With c7i.24xlarge at $6/hour, unused time is expensive

## S3 Organization

Your results will be organized by run name (which includes timestamp and indices):

```
s3://sarah-research-aladynoulli/results/
├── age_offset_0_10000_20241225_120000/
│   ├── pi_enroll_fixedphi_age_offset_0_sex_0_10000_...pt
│   ├── pi_enroll_fixedphi_age_offset_1_sex_0_10000_...pt
│   └── ...
├── age_offset_10000_20000_20241225_140000/
│   └── ...
└── ...
```

Each batch creates its own folder with timestamp, so results are easy to identify.

## Monitoring Multiple Batches

### Check Batch Status

```bash
# List all batch logs
ls -lh logs/run_*.log

# Watch latest log
tail -f logs/run_*_$(date +%Y%m%d)*.log

# Check running processes
ps aux | grep forAWS_offsetmasterfix

# Check output files
ls -lh output/ | grep batch
```

### Check S3 Results

```bash
# List all batch results
aws s3 ls s3://sarah-research-aladynoulli/results/ --recursive

# Count files per batch
aws s3 ls s3://sarah-research-aladynoulli/results/ --recursive | wc -l
```

## After All Batches Complete

1. **Verify all results in S3**
   ```bash
   aws s3 ls s3://sarah-research-aladynoulli/results/ --recursive
   ```

2. **Download results locally** (if needed)
   ```bash
   aws s3 sync s3://sarah-research-aladynoulli/results/ ./local-results/
   ```

3. **TERMINATE THE INSTANCE** ⚠️
   - Don't forget!
   - At $6/hour, leaving it running is expensive

## Expected Timeline

With `c7i.24xlarge`:

- **Per batch:** ~30-60 minutes
- **10 batches:** ~5-10 hours (or up to 20 hours if longer)
- **Total cost:** ~$30-120

**Your budget:** "A few hundred dollars" covers this easily!

## Troubleshooting

### Batch fails mid-run
- Check the log file for that specific batch
- Re-run just that batch with same indices
- Other batches are unaffected

### Instance runs out of resources
- Unlikely with c7i.24xlarge (96 vCPUs, 192 GB RAM)
- If issues, reduce batch size or wait between batches

### Need to pause/resume
- You can stop the instance (not terminate) between batches
- Only pay for storage (~$0.10/month for 50 GB)
- Resume later and continue where you left off




