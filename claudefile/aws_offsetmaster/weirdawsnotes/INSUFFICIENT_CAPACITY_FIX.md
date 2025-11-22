# Fix: "Insufficient Capacity" Error

You're getting "Insufficient capacity" when trying to launch `c7i.24xlarge`. This is common with large instances.

## Quick Fixes (Try These First)

### Option 1: Try Different Availability Zones

1. **Click "Edit instance config"** (blue button on the error page)
2. Scroll to **"Network settings"** section
3. **Uncheck "Auto-assign public IP"** if checked (or leave as is)
4. Under **"Subnet"** or **"Availability Zone"**, try a different AZ:
   - If you're in `us-east-1a`, try `us-east-1b` or `us-east-1c`
   - Try each AZ in your region one by one

5. Click **"Launch instance"** again

**Why this works:** Different AZs have different instance availability.

### Option 2: Try Different Region

1. **Click "Cancel"** to go back
2. In the AWS Console, **change your region** (top right)
3. Common regions with good capacity:
   - `us-east-1` (N. Virginia) - Usually has best availability
   - `us-west-2` (Oregon)
   - `eu-west-1` (Ireland)
   
4. **Re-launch with same settings**

**Note:** If you change regions, make sure your S3 bucket is accessible from that region (S3 is global, so this should be fine).

### Option 3: Try Multiple Availability Zones at Once

When configuring the instance:

1. Go to **"Network settings"**
2. Click **"Edit"** 
3. **Select multiple Availability Zones** or use **"No preference"**
4. This lets AWS choose an AZ with capacity

## Alternative Solutions

### Option 4: Use a Slightly Smaller Instance (Still Very Fast) ⭐ **RECOMMENDED**

If `c7i.24xlarge` isn't available, try these alternatives - **they're actually a better choice**:

**Best alternatives:**
- **`c7i.16xlarge`** - 64 vCPU, 128 GB RAM (~$4.00-4.50/hour) ⭐ **GREAT CHOICE**
  - Still very fast: ~40-80 min per batch
  - More likely to be available
  - Total for 10 batches: ~$40-90
  - **Recommended if you want speed but save some money**

- **`c7i.12xlarge`** - 48 vCPU, 96 GB RAM (~$3.00-3.50/hour) ⭐ **BEST VALUE**
  - Good speed: ~50-100 min per batch
  - Even more likely to be available
  - Total for 10 batches: ~$30-70
  - **Best balance of speed and cost**

- **`c7i.8xlarge`** - 32 vCPU, 64 GB RAM (~$2.00-2.50/hour)
  - Still good: ~60-120 min per batch
  - Very likely to be available
  - Total for 10 batches: ~$20-50
  - **Cheapest option, still reasonable speed**

**Performance comparison:**
- `c7i.24xlarge` (96 vCPU): ~30-60 min per batch
- `c7i.16xlarge` (64 vCPU): ~40-80 min per batch
- `c7i.12xlarge` (48 vCPU): ~50-100 min per batch
- `c7i.8xlarge` (32 vCPU): ~60-120 min per batch

**Recommendation:** Try `c7i.16xlarge` - still very powerful and more likely to be available!

### Option 5: Use Spot Instances (Cheaper, But Can Be Interrupted)

**Pros:**
- Much cheaper (~$0.60-1.50/hour instead of $6/hour)
- Can save up to 90%

**Cons:**
- Can be interrupted if AWS needs capacity
- Need to handle interruptions in your script

**How to use Spot:**
1. In "Configure instance" → **"Purchasing option"**
2. Check **"Request Spot Instances"**
3. Set max price (recommend: ~$2-3/hour - below on-demand price)
4. For interruption behavior: Choose **"Stop"** or **"Terminate"**

**For your use case:** Spot might work if interruptions are acceptable (can re-run a batch if interrupted).

### Option 6: Use Capacity Reservations (Advanced)

If you need guaranteed capacity:
1. Go to EC2 → **"Capacity Reservations"**
2. Create a reservation for `c7i.24xlarge`
3. Then launch in that reservation

**Note:** This requires planning ahead and may have minimum commitments.

### Option 7: Wait and Retry Later

Sometimes capacity becomes available:
- Try again in a few hours
- Weekends/off-hours often have more capacity
- Check during off-peak times (late night, early morning in your timezone)

## Recommended Approach

### Step-by-Step:

1. **First, try different AZs** (Option 1) - Easiest fix
2. **If that fails, try `us-east-1` region** (Option 2) - Usually best availability
3. **If still fails, try `c7i.16xlarge`** (Option 4) - Still very fast, more available

### My Recommendation Right Now:

**Try `c7i.16xlarge` instead:**
- 64 vCPUs (vs 96) - Still very powerful
- 128 GB RAM (vs 192) - Still plenty
- More likely to be available
- ~$4-4.50/hour instead of $6/hour
- Runtime: ~40-80 min per batch (vs 30-60 min)

**Total cost for 10 batches:** ~$40-90 instead of $30-120
**Total time:** ~7-14 hours instead of 5-10 hours

## Checking Instance Availability

You can check which instances are available:

1. Go to EC2 → **"Instance Types"**
2. Search for `c7i`
3. Look for instances marked as available

Or use AWS CLI:
```bash
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=c7i.24xlarge \
  --region us-east-1
```

## Quick Decision Tree

```
Insufficient Capacity?
│
├─ Try different AZ? → Yes → Usually fixes it!
│
├─ Still failing? → Try different region (us-east-1)
│
├─ Still failing? → Try c7i.16xlarge (64 vCPU, still very fast)
│
├─ Still failing? → Try c7i.12xlarge (48 vCPU)
│
└─ Need guaranteed capacity? → Use Spot Instances or wait
```

## After You Get It Working

Once you successfully launch an instance:

1. **Note which AZ/region worked** - Use same for future launches
2. **Consider creating an AMI** - Save configured instance as image
3. **Use Launch Templates** - Save your configuration for easy re-launch

## Summary

**Best immediate fix:**
1. Try different Availability Zones first (easiest)
2. Try `us-east-1` region if AZ change doesn't work
3. Consider `c7i.16xlarge` (64 vCPU) - still very powerful and more available

**Don't worry** - this is a common issue with large instances. One of these solutions will work!

