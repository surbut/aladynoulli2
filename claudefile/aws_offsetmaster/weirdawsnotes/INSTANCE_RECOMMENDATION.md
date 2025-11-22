# Instance Type Recommendation - CPU Only

## Your Code is CPU-Only

Since your PyTorch code runs on CPU (not GPU), you should use a **CPU-optimized instance**.

## Your Instance: **`c7i.24xlarge`** ⚡ (If Available)

- **vCPUs:** 96 cores
- **Memory:** 192 GiB RAM
- **Type:** Compute Optimized (Intel Xeon)
- **Cost:** ~$6.00-6.50/hour
- **Estimated runtime:** ~30-60 minutes for full run (0-10000 indices, 11 age offsets)
- **Total cost per run:** ~$3.00-6.50

**This is a powerful instance!**
- 96 vCPUs = much faster processing
- 192 GB RAM = can handle very large batches
- Will complete your runs much faster than smaller instances
- More expensive per hour, but finishes quickly

**⚠️ Availability Note:** Large instances like `c7i.24xlarge` may show "Insufficient capacity" errors. See `INSUFFICIENT_CAPACITY_FIX.md` for solutions.

### Alternative If c7i.24xlarge Not Available: **`c7i.16xlarge`** ⭐

- **vCPUs:** 64 cores
- **Memory:** 128 GiB RAM
- **Cost:** ~$4.00-4.50/hour
- **Estimated runtime:** ~40-80 minutes per batch
- **Total cost for 10 batches:** ~$40-90
- **More likely to be available** than c7i.24xlarge

## Alternative: Smaller Instance (if you want to save money)

### ⭐ **`c7i.2xlarge`** - Budget Option

- **vCPUs:** 8 cores
- **Memory:** 16 GiB RAM
- **Cost:** ~$0.32-0.34/hour
- **Estimated runtime:** 5-8 hours for full run
- **Total cost per run:** ~$1.60-2.72

**Why this one?**
- Much cheaper per hour
- Better for cost-conscious runs
- Takes longer but costs less overall

## Alternative Options

### If you need more memory:
- **`m7i.2xlarge`** - 8 vCPU, 32 GiB RAM (~$0.38-0.40/hour)
  - Use if you get out-of-memory errors with c7i.2xlarge

### If you need more compute:
- **`c7i.4xlarge`** - 16 vCPU, 32 GiB RAM (~$0.64-0.68/hour)
  - Faster, but ~2x the cost
  - Use if c7i.2xlarge is too slow

### Previous generation (slightly cheaper):
- **`c6i.2xlarge`** - 8 vCPU, 16 GiB RAM (~$0.32/hour)
  - Similar performance, slightly older hardware

## ❌ DO NOT USE GPU Instances

Since your code is CPU-only:
- ❌ Don't use `g4dn.xlarge`, `g5.xlarge`, or any GPU instances
- ❌ You'll pay extra for GPUs you can't use
- ✅ Stick with CPU-optimized instances like `c7i.2xlarge`

## How to Select in Console

1. In EC2 Launch Instance page
2. Go to "Instance type" section
3. Search for: **`c7i.2xlarge`**
4. Select it
5. Continue with launch

## Cost Optimization Tips

1. **Terminate immediately after completion** - Don't leave the instance running (CRITICAL with c7i.24xlarge!)
2. **Use Spot Instances** - Can save up to 90% (but can be interrupted)
3. **Stop (not terminate) during breaks** - Only pay for storage, not compute
4. **Monitor usage** - Use CloudWatch to see if you're using all resources

## Estimated Total Costs

For one full run (0-10000 indices, 11 age offsets):

### Your Instance:
- **c7i.24xlarge:** ~30-60 minutes × $6.25/hour = **~$3.13-6.25** ⚡ **FASTEST**

### Budget Options (if you want to save money):
- **c7i.2xlarge:** ~5-8 hours × $0.33/hour = **~$1.65-2.64** 💰 **CHEAPEST**
- **c7i.4xlarge:** ~4-6 hours × $0.66/hour = **~$2.64-3.96**
- **m7i.2xlarge:** ~5-8 hours × $0.39/hour = **~$1.95-3.12**

**Your choice (c7i.24xlarge):** Fastest option - finishes in under an hour, costs ~$3-6 per run.
**Budget option (c7i.2xlarge):** Cheapest - takes longer but saves ~$1-3 per run.

**⚠️ IMPORTANT:** With c7i.24xlarge, be extra careful to terminate immediately after completion - at $6+/hour, leaving it running for even an hour costs a lot!

