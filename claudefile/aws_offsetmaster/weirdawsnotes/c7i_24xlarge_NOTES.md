# Notes for c7i.24xlarge Instance

## Your Instance Configuration

**Instance Type:** `c7i.24xlarge`
- **vCPUs:** 96 cores
- **Memory:** 192 GiB RAM
- **Cost:** ~$6.00-6.50/hour
- **Type:** Compute Optimized (Intel Xeon)

## Performance Expectations

With 96 vCPUs and 192 GB RAM, this instance will:
- **Complete runs much faster** (~30-60 minutes vs 5-8 hours on smaller instances)
- **Handle larger batches** if you want to increase your batch size
- **Process multiple tasks in parallel** efficiently

## Cost Considerations

### Cost per Run
- **Full run (0-10000 indices, 11 age offsets):** ~$3-6
- **Per hour cost:** ~$6.25/hour

### Cost Savings Tips

1. **Terminate immediately after completion**
   - Don't leave the instance running
   - At $6+/hour, even 1 extra hour costs $6+
   - Set a reminder or use CloudWatch alarm to alert you when done

2. **Monitor the script closely**
   - Watch the log file: `tail -f logs/run_*.log`
   - Check progress regularly
   - Terminate as soon as you see "All done!"

3. **Use smaller instance for testing**
   - Test your script on `c7i.2xlarge` first ($0.33/hour)
   - Then run full job on `c7i.24xlarge` when confident

4. **Consider Spot Instances** (Advanced)
   - Can save up to 90% (~$0.60-0.65/hour)
   - Risk of interruption if AWS needs capacity
   - Good for non-critical runs

## What Makes This Instance Fast

1. **96 vCPUs** - Massive parallelization
   - Your PyTorch code can utilize many cores
   - Multiple age offset models can be processed faster

2. **192 GB RAM** - Large memory capacity
   - Can load larger batches
   - Less swapping to disk
   - Faster data access

3. **Compute Optimized** - Optimized for CPU workloads
   - Intel Xeon processors
   - High clock speeds
   - Designed for compute-intensive tasks

## Optimizing for This Instance

Since you have so many cores and memory:

1. **You could increase batch size**
   - Current: 0-10000 indices
   - Could try: 0-20000 or larger if needed
   - With 192 GB RAM, you have plenty of headroom

2. **Process multiple age offsets in parallel** (if you modify the script)
   - Current script processes sequentially
   - Could parallelize across multiple age offsets
   - Would need script modifications

3. **Monitor resource usage**
   - Check CPU usage: `htop` or `top`
   - Check memory: `free -h`
   - See if you're fully utilizing resources

## Setup Notes

The setup process is the same as for smaller instances:

1. **Follow EC2_SETUP_CONSOLE.md** - same steps
2. **Use Ubuntu Server 22.04 LTS** - recommended
3. **All commands are the same** - no special configuration needed

## Termination Reminder

**Set a reminder to terminate this instance!**

At $6+/hour:
- 1 hour = $6+
- 2 hours = $12+
- 1 day (if you forget) = $144+

**Always check:**
1. Is the script finished? Check the log file
2. Are results uploaded to S3? Verify in S3 console
3. **TERMINATE THE INSTANCE** - Don't wait!

## Quick Commands

```bash
# Check if script is running
ps aux | grep forAWS_offsetmasterfix

# Monitor log
tail -f logs/run_*.log

# Check resource usage
htop
free -h

# Verify results in S3
aws s3 ls s3://sarah-research-aladynoulli/results/

# TERMINATE INSTANCE (do this when done!)
# In EC2 Console: Instance → Instance state → Terminate instance
```

## Comparison to Smaller Instances

| Instance | vCPUs | RAM | Cost/hr | Runtime | Total Cost |
|----------|-------|-----|---------|---------|------------|
| **c7i.24xlarge** ⚡ | 96 | 192 GB | $6.25 | 30-60 min | **$3-6** |
| c7i.2xlarge | 8 | 16 GB | $0.33 | 5-8 hours | $1.60-2.64 |
| c7i.4xlarge | 16 | 32 GB | $0.66 | 4-6 hours | $2.64-3.96 |

**Your choice:** Fastest option, worth it if time is valuable!




