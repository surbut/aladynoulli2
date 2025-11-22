# EC2 Setup Guide - Using AWS Console

This guide walks you through launching an EC2 instance from the AWS Console.

## Step 1: Launch an EC2 Instance

### From the EC2 Console:

1. **Navigate to EC2 Dashboard**
   - In AWS Console, go to "EC2" service
   - Click "Instances" in the left sidebar
   - Click "Launch instance" button

2. **Name Your Instance**
   - Give it a name like: `aladyn-age-offset-predictions`

## Step 2: Choose an Instance Type

**Your code runs on CPU only**, so you need a CPU-optimized instance.

### Recommended CPU Instance Types

**Best choices:**
- **`c7i.2xlarge`** - 8 vCPU, 16 GiB RAM (Compute Optimized) ⭐ **RECOMMENDED**
  - Best for: CPU-intensive workloads, fast training
  - Cost: ~$0.32-0.34/hour
  - **This is your best option!**

- **`c7i.4xlarge`** - 16 vCPU, 32 GiB RAM (if you need more memory)
  - Best for: Larger batches or if you run out of memory
  - Cost: ~$0.64-0.68/hour

- **`m7i.2xlarge`** - 8 vCPU, 32 GiB RAM (General Purpose - more memory)
  - Best for: If you need more RAM than c7i.2xlarge
  - Cost: ~$0.38-0.40/hour

**Other options:**
- **`c6i.2xlarge`** - 8 vCPU, 16 GiB RAM (previous generation, slightly cheaper)
  - Cost: ~$0.32/hour

### Recommendation

**Primary Choice: `c7i.24xlarge`** ⚡ (If available)

This is a very powerful instance:
- **96 vCPUs** - Excellent parallelization
- **192 GiB RAM** - Can handle very large batches
- **Fast runtime** - Completes in ~30-60 minutes
- **Higher cost** - ~$6+/hour, so ~$3-6 per run

**⚠️ If you get "Insufficient capacity" error:**
- Try different Availability Zones (see `INSUFFICIENT_CAPACITY_FIX.md`)
- Try different region (us-east-1 usually has best availability)
- Use alternative: **`c7i.16xlarge`** (64 vCPU, 128 GB RAM, ~$4-4.50/hour) - Still very fast!

**Alternative budget options:**
- **`c7i.2xlarge`** - 8 vCPU, 16 GiB RAM, ~$0.33/hour (5-8 hours runtime, cheaper overall)
- **`c7i.4xlarge`** - 16 vCPU, 32 GiB RAM, ~$0.66/hour (4-6 hours runtime)

**Note:** Since your code is CPU-only, **don't use GPU instances** - you'll pay more for GPUs you won't use!

**⚠️ IMPORTANT:** With c7i.24xlarge, always terminate the instance immediately after your script completes - it costs ~$6/hour, so leaving it running is expensive!

## Step 3: Select Amazon Machine Image (AMI)

### Recommended: Ubuntu Server 22.04 LTS ⭐

1. **Choose:** "Ubuntu Server 22.04 LTS" (64-bit (x86))
   - **Why Ubuntu?** 
     - Easier Python/conda setup
     - Better PyTorch support
     - More tutorials/examples for scientific computing
     - Simpler package management with `apt-get`

2. **Architecture:** x86_64 (required for `c7i.2xlarge` instances)

### Alternative: Amazon Linux 2023

**Choose this if:** You prefer AWS-optimized Linux distributions
- Uses `dnf` package manager (instead of `apt-get`)
- Default user is `ec2-user` (instead of `ubuntu`)
- Requires slightly different setup commands

**Our recommendation:** Stick with **Ubuntu Server 22.04 LTS** - it's easier to set up for Python/PyTorch workloads.

**Note:** If you choose Amazon Linux instead, see `AMAZON_LINUX_SETUP.md` for the equivalent setup commands (uses `dnf` instead of `apt-get`, user is `ec2-user` instead of `ubuntu`).

## Step 4: Configure Key Pair

1. **Create or Select Key Pair**
   - If you don't have one, click "Create new key pair"
   - Name it: `aladyn-ec2-key` (or whatever you prefer)
   - Choose: `RSA` or `ED25519`
   - Format: `.pem` for SSH
   - Click "Create key pair" - **Save the .pem file!** You'll need it to SSH.

2. **Select your key pair** from the dropdown

## Step 5: Network Settings

1. **VPC:** Use default VPC (or create one if needed)
2. **Security Group:**
   - Create new security group
   - Name: `aladyn-sg`
   - Description: `Security group for Aladyn predictions`
   - **Add rule:** 
     - Type: SSH
     - Port: 22
     - Source: My IP (or 0.0.0.0/0 for anywhere - less secure)

## Step 6: Configure Storage

For your large tensors and 10 batches, you'll need enough storage:

1. **Root volume:**
   - Size: **100-200 GB** ⭐ **RECOMMENDED: 150 GB**
   - Volume type: gp3 (General Purpose SSD)
   - IOPS: Default (3000)
   - Throughput: Default (125 MB/s)

   **Why 150 GB?**
   - Input files from S3: ~8-32 GB
   - Output files (while uploading): ~10-20 GB
   - System/OS: ~20 GB
   - Safety buffer: ~30-50 GB
   - **Total needed: ~100-150 GB**

2. **Optional:** Add an additional volume if you need more space (but 150 GB should be plenty)

**Storage cost:** ~$0.08-0.10 per GB per month - for 1 day of use, storage cost is negligible (~$0.01).

**See `STORAGE_REQUIREMENTS.md` for detailed storage breakdown.**

## Step 7: Advanced Details (Optional but Recommended)

Under "Advanced details":

1. **IAM role:** (Optional) Create/select a role with S3 access permissions
   - Allows the instance to access S3 without storing credentials
   - Or you can configure AWS credentials later

2. **User data:** (Optional) Paste this to auto-install dependencies:

```bash
#!/bin/bash
# Update system
apt-get update -y
apt-get upgrade -y

# Install basic tools
apt-get install -y wget git curl unzip python3 python3-pip

# Install Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ubuntu/miniconda3
chown -R ubuntu:ubuntu /home/ubuntu/miniconda3

# Install AWS CLI
apt-get install -y awscli

# Add conda to PATH
echo 'export PATH="/home/ubuntu/miniconda3/bin:$PATH"' >> /home/ubuntu/.bashrc
```

## Step 8: Launch Instance

1. Review your settings
2. Click "Launch instance"
3. Wait for instance to be in "running" state (green checkmark)

## Step 9: Connect to Your Instance

You have several options:

### Option A: EC2 Instance Connect (No SSH Key Needed)

1. Select your instance
2. Click "Connect"
3. Choose "EC2 Instance Connect" tab
4. Click "Connect" - opens browser-based terminal

### Option B: SSH (If you downloaded the .pem key)

1. Select your instance
2. Click "Connect"
3. Choose "SSH client" tab
4. Follow the instructions shown
5. Example:
   ```bash
   chmod 400 your-key.pem
   # For Ubuntu:
   ssh -i your-key.pem ubuntu@your-instance-ip
   # For Amazon Linux:
   ssh -i your-key.pem ec2-user@your-instance-ip
   ```

### Option C: AWS Systems Manager Session Manager

**Works with both Ubuntu and Amazon Linux** (SSM agent is pre-installed on Amazon Linux, needs to be installed on Ubuntu)

1. Select your instance
2. Click "Connect"
3. Choose "Session Manager" tab
4. Click "Connect"

**Note:** If using Ubuntu, you may need to install SSM agent first:
```bash
sudo snap install amazon-ssm-agent --classic
```

**Easier option:** Use EC2 Instance Connect (Option A) - works immediately on both Ubuntu and Amazon Linux!

## Step 10: Set Up the Environment

**These commands are for Ubuntu. If you chose Amazon Linux, see the note below.**

Once connected, run these commands:

### 1. Update System
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. Install AWS CLI (if not installed)
```bash
sudo apt-get install -y awscli
aws configure
```
Enter your AWS Access Key ID and Secret Access Key when prompted.

### 3. Install Miniconda (if not in user data)
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/etc/profile.d/conda.sh
```

### 4. Create Conda Environment
```bash
conda create -n aladyn python=3.9 -y
conda activate aladyn
pip install torch numpy pandas scipy scikit-learn matplotlib
```

**Note:** Since you're using CPU-only code, install the CPU version of PyTorch (no CUDA needed).

### 5. Clone/Copy Your Code

**Option A: From Git**
```bash
git clone <your-repo-url>
cd aladynoulli2/claudefile/aws_offsetmaster
```

**Option B: Upload Files Manually**
- Use EC2 Instance Connect file upload feature
- Or use `scp` from your local machine:
  ```bash
  scp -i your-key.pem -r claudefile/aws_offsetmaster ubuntu@your-instance-ip:~/
  ```

### 6. Make Scripts Executable
```bash
chmod +x *.sh
```

### 7. Copy Required Python Files

You'll need `utils.py` and `clust_huge_amp_fixedPhi.py` in the same directory:

```bash
# If they're in your repo, copy them:
cp ../../../pyScripts/utils.py .
cp ../../../pyScripts/clust_huge_amp_fixedPhi.py .
```

### 8. Run the Script

```bash
./run_aws.sh
```

Or with custom parameters:
```bash
./run_aws.sh 0 10000 s3://sarah-research-aladynoulli 10
```

## Step 11: Monitor Progress

### In the Terminal
```bash
# Watch the log
tail -f logs/run_*.log

# Check if script is running
ps aux | grep forAWS_offsetmasterfix

# Check output files
ls -lh output/
```

### From AWS Console
- Go to EC2 → Instances
- Check CloudWatch metrics for CPU/Memory usage
- View instance status and health checks

## Step 12: Download Results

Results are automatically uploaded to S3, but you can also download locally:

```bash
# Download results from S3 to local machine
aws s3 sync s3://sarah-research-aladynoulli/results/your-run-name ./local-results/
```

## Step 13: Clean Up (Important!)

**When you're done, stop or terminate the instance to avoid charges:**

1. Select your instance in EC2 Console
2. Click "Instance state" → "Stop instance" (to keep it) or "Terminate instance" (to delete it)

**Note:** Terminated instances cannot be recovered!

## Troubleshooting

### Instance won't start
- Check your security group allows SSH (port 22)
- Verify key pair is correct

### Can't connect via SSH
- Check security group rules
- Verify instance is in "running" state
- For EC2 Instance Connect, make sure it's enabled

### Out of memory errors
- Use a larger instance type
- Process smaller batches (reduce end_index - start_index)

### S3 access denied
- Configure AWS credentials: `aws configure`
- Or set up IAM role with S3 permissions

### Python import errors
- Make sure `utils.py` and `clust_huge_amp_fixedPhi.py` are in the same directory
- Check PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`

## Cost Estimates

For a typical run (0-10000 indices, 11 age offsets) with CPU-only code:

### Your Instance:
- **c7i.24xlarge (CPU):** ~30-60 minutes = **$3.13-6.25** ⚡ **YOUR CHOICE - FASTEST**

### Budget Options:
- **c7i.2xlarge (CPU):** ~5-8 hours = $1.60-2.72 💰 **CHEAPEST**
- **c7i.4xlarge (CPU):** ~4-6 hours = $2.56-4.08 (faster but more expensive)
- **m7i.2xlarge (CPU):** ~5-8 hours = $1.90-3.20 (more memory if needed)

**Your choice (c7i.24xlarge):** Fastest option - completes in under an hour, costs ~$3-6 per run.

**⚠️ CRITICAL:** With c7i.24xlarge at ~$6+/hour, **ALWAYS terminate the instance immediately after completion**. Even leaving it running for 1 hour costs $6+!

## Next Steps

After your first successful run, you might want to:
1. Create an AMI (image) of your configured instance for faster setup next time
2. Set up CloudWatch alarms for monitoring
3. Use Spot Instances for significant cost savings (with risk of interruption)

