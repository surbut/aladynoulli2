# Setup Commands for Amazon Linux 2023

If you chose **Amazon Linux 2023** instead of Ubuntu, use these commands:

## Differences from Ubuntu

- **Default user:** `ec2-user` (instead of `ubuntu`)
- **Package manager:** `dnf` (instead of `apt-get`)
- **Python:** Already installed (Python 3.9+)
- **AWS tools:** Already installed (AWS CLI, etc.)

## Setup Commands

### 1. Update System
```bash
sudo dnf update -y
```

### 2. Install Basic Tools
```bash
sudo dnf install -y wget git curl unzip gcc gcc-c++ make
```

### 3. Install AWS CLI (usually already installed, but update if needed)
```bash
# Check if installed
aws --version

# If not installed or need to update:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws configure
```

### 4. Install Miniconda
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/etc/profile.d/conda.sh
```

### 5. Create Conda Environment
```bash
conda create -n aladyn python=3.9 -y
conda activate aladyn
pip install torch numpy pandas scipy scikit-learn matplotlib
```

### 6. Clone/Copy Your Code

**Option A: From Git**
```bash
git clone <your-repo-url>
cd aladynoulli2/claudefile/aws_offsetmaster
```

**Option B: Upload Files Manually**
- Use EC2 Instance Connect file upload feature
- Or use `scp` from your local machine:
  ```bash
  scp -i your-key.pem -r claudefile/aws_offsetmaster ec2-user@your-instance-ip:~/
  ```

### 7. Make Scripts Executable
```bash
chmod +x *.sh
```

### 8. Copy Required Python Files
```bash
# If they're in your repo, copy them:
cp ../../../pyScripts/utils.py .
cp ../../../pyScripts/clust_huge_amp_fixedPhi.py .
```

### 9. Run the Script
```bash
./run_aws.sh
```

## Key Differences Summary

| Task | Ubuntu | Amazon Linux |
|------|--------|--------------|
| User | `ubuntu` | `ec2-user` |
| Update | `sudo apt-get update` | `sudo dnf update -y` |
| Install | `sudo apt-get install` | `sudo dnf install` |
| SSH | `ssh ubuntu@...` | `ssh ec2-user@...` |
| AWS CLI | Usually need to install | Usually pre-installed |
| Python | Usually need to install | Pre-installed (3.9+) |

## Recommendation

**Stick with Ubuntu** - it's simpler for Python/PyTorch workloads and has more community support. But Amazon Linux works fine too!




