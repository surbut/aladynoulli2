# Full-Mode Pi Computation and EC2 Recommendations

## 1. Computing Pi from Full-Mode Models

The full-mode models (joint lambda and phi estimation) are saved at:
```
Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_350000_360000.pt
```

### Formula
```
pi = kappa * sigmoid(phi) * softmax(lambda)
```

Where:
- `lambda`: Shape (N, K, T) - individual signature loadings
- `phi`: Shape (K, D, T) - signature-disease associations  
- `kappa`: Scalar - scaling parameter
- `pi`: Shape (N, D, T) - disease probabilities

### Usage

**Single batch:**
```bash
python compute_pi_from_fullmode_models.py \
    --single_batch \
    --model_path "/path/to/enrollment_model_W0.0001_batch_350000_360000.pt" \
    --output_dir "/path/to/output"
```

**All batches (0-40):**
```bash
python compute_pi_from_fullmode_models.py \
    --base_dir "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full" \
    --start_batch 0 \
    --end_batch 40 \
    --output_dir "/path/to/output"
```

This will:
1. Load each model checkpoint
2. Extract lambda, phi, and kappa
3. Compute pi using `calculate_pi_pred()` from `utils.py`
4. Save individual batch pi files: `pi_fullmode_batch_350000_360000.pt`
5. Concatenate all batches and save: `pi_fullmode_400k.pt`

### Implementation Details

The script uses the `calculate_pi_pred()` function from `utils.py`:
```python
from utils import calculate_pi_pred

# Load model
checkpoint = torch.load(model_path, weights_only=False)
lambda_ = checkpoint['model_state_dict']['lambda_']  # (N, K, T)
phi = checkpoint['model_state_dict']['phi']  # (K, D, T)
kappa = checkpoint['model_state_dict']['kappa']  # scalar

# Compute pi
pi = calculate_pi_pred(lambda_, phi, kappa)  # (N, D, T)
```

The `calculate_pi_pred` function implements:
```python
# 1. Softmax over signatures (normalize lambda)
all_thetas = softmax_by_k(lambda_params)  # (N, K, T)

# 2. Sigmoid of phi
phi_prob = sigmoid(phi)  # (K, D, T)

# 3. Einsum: nkt,kdt->ndt (sum over signatures)
pi = torch.einsum('nkt,kdt->ndt', all_thetas, phi_prob) * kappa
```

## 2. EC2 Instance Recommendations for Running Two Batches in Parallel

### Resource Requirements Analysis

Based on `run_aladyn_batch.py`:
- **Batch size**: 10,000 patients
- **Data tensors**: Y (N×D×T), E (N×D), G (N×P)
- **Model**: AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
- **Training**: 200 epochs (default)
- **Memory**: Need to hold full tensors + model + gradients

Estimated per-batch requirements:
- **Memory**: ~20-40 GB RAM
- **CPU**: Multi-core (benefits from parallelization)
- **GPU**: Optional but recommended for faster training
- **Storage**: ~10-20 GB for data + model checkpoints

### Recommended EC2 Instances (for 2 parallel batches)

#### Option 1: GPU Instances (RECOMMENDED for fastest training)

**p4d.24xlarge** (Best performance, highest cost)
- **vCPUs**: 96
- **GPUs**: 8× NVIDIA A100 (40GB each)
- **Memory**: 1,152 GB
- **Network**: 400 Gbps
- **Cost**: ~$32/hour
- **Why**: Can easily run 2 batches in parallel on separate GPUs, fastest training

**p3.16xlarge** (Good balance)
- **vCPUs**: 64
- **GPUs**: 8× NVIDIA V100 (16GB each)
- **Memory**: 488 GB
- **Network**: 25 Gbps
- **Cost**: ~$24/hour
- **Why**: Good GPU memory, can run 2 batches on separate GPUs

**g5.12xlarge** (Cost-effective GPU option)
- **vCPUs**: 48
- **GPUs**: 4× NVIDIA A10G (24GB each)
- **Memory**: 192 GB
- **Network**: 25 Gbps
- **Cost**: ~$5/hour
- **Why**: Modern GPUs, good price/performance, can run 2 batches

#### Option 2: CPU-Only Instances (If GPU not needed)

**c7i.24xlarge** (Latest generation, best CPU performance)
- **vCPUs**: 96 (4th Gen Intel Xeon)
- **Memory**: 192 GB
- **Network**: 50 Gbps
- **Cost**: ~$4/hour
- **Why**: Latest CPU architecture, high clock speed, good for CPU-bound workloads

**c6i.24xlarge** (Previous generation, still excellent)
- **vCPUs**: 96 (3rd Gen Intel Xeon)
- **Memory**: 192 GB
- **Network**: 50 Gbps
- **Cost**: ~$4/hour
- **Why**: Proven performance, good value

**r7i.24xlarge** (If you need more memory)
- **vCPUs**: 96
- **Memory**: 768 GB
- **Network**: 50 Gbps
- **Cost**: ~$6.72/hour
- **Why**: More memory if data loading is memory-intensive

### Running Two Batches in Parallel

**Method 1: Separate processes (recommended)**
```bash
# Terminal 1
python run_aladyn_batch.py --start_index 0 --end_index 10000 --data_dir /data --output_dir /results/batch0 &

# Terminal 2  
python run_aladyn_batch.py --start_index 10000 --end_index 20000 --data_dir /data --output_dir /results/batch1 &

# Wait for both
wait
```

**Method 2: Using GNU parallel**
```bash
parallel -j 2 python run_aladyn_batch.py \
    --start_index {1} --end_index {2} \
    --data_dir /data --output_dir /results/batch{3} \
    ::: 0 10000 \
    ::: 10000 20000 \
    ::: 0 1
```

**Method 3: Python multiprocessing**
```python
from multiprocessing import Process

def run_batch(start, end, batch_id):
    import subprocess
    subprocess.run([
        'python', 'run_aladyn_batch.py',
        '--start_index', str(start),
        '--end_index', str(end),
        '--data_dir', '/data',
        '--output_dir', f'/results/batch{batch_id}'
    ])

# Run two batches in parallel
p1 = Process(target=run_batch, args=(0, 10000, 0))
p2 = Process(target=run_batch, args=(10000, 20000, 1))
p1.start()
p2.start()
p1.join()
p2.join()
```

### Cost Optimization Tips

1. **Use Spot Instances**: Can save 50-90% on GPU instances
   ```bash
   aws ec2 request-spot-instances --instance-count 1 \
       --launch-specification file://spot-spec.json
   ```

2. **Use Reserved Instances**: If running for extended periods (1-3 year commitment)

3. **Monitor with CloudWatch**: Track actual resource usage to right-size instances

4. **Auto-shutdown**: Set up Lambda to stop instances when jobs complete

### Recommended Setup

**For fastest completion (money no object):**
- **Instance**: `p4d.24xlarge` (8× A100 GPUs)
- **Strategy**: Run 2 batches on separate GPUs (GPU 0-3 for batch 1, GPU 4-7 for batch 2)
- **Expected time**: ~2-4 hours per batch (depending on convergence)

**For best value:**
- **Instance**: `g5.12xlarge` (4× A10G GPUs)  
- **Strategy**: Run 2 batches on separate GPUs (GPU 0-1 for batch 1, GPU 2-3 for batch 2)
- **Expected time**: ~4-8 hours per batch

**For CPU-only (if GPU not available):**
- **Instance**: `c7i.24xlarge` (96 vCPUs)
- **Strategy**: Run 2 batches in parallel processes
- **Expected time**: ~8-16 hours per batch

### Monitoring Commands

```bash
# Check GPU usage
nvidia-smi

# Check CPU/memory usage
htop

# Check disk I/O
iostat -x 1

# Monitor both batches
watch -n 1 'ps aux | grep run_aladyn_batch'
```



