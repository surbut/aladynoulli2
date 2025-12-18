#!/usr/bin/env python3
"""
Pool phi values from all AOU batch checkpoints and save to aou_model_initialized.pt
This creates the checkpoint file used by Figure2_Population_Level_Patterns.ipynb
"""

import torch
import numpy as np
from pathlib import Path
import glob

# Configuration
AOU_BATCH_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/aou_batches')
OUTPUT_CHECKPOINT = Path('/Users/sarahurbut/aladynoulli2/aou_model_master_correctedE.pt')
OLD_CHECKPOINT_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt'

print("="*80)
print("POOLING PHI FROM AOU BATCHES")
print("="*80)
print(f"Batch directory: {AOU_BATCH_DIR}")
print(f"Output checkpoint: {OUTPUT_CHECKPOINT}")
print("="*80)

# Find all batch files
batch_pattern = str(AOU_BATCH_DIR / 'aou_model_batch_*_*_*.pt')
batch_files = sorted(glob.glob(batch_pattern))

if len(batch_files) == 0:
    raise FileNotFoundError(f"No batch files found in {AOU_BATCH_DIR}")

print(f"\nFound {len(batch_files)} batch files")

# Extract batch index for proper sorting
def extract_batch_idx(filename):
    """Extract batch index from filename like 'aou_model_batch_5_50000_60000.pt'"""
    return int(Path(filename).name.split('_')[3])

batch_files = sorted(batch_files, key=extract_batch_idx)

print(f"Will pool phi from {len(batch_files)} batches...\n")

# Load phi from all batches
all_phis = []
batch_info_list = []

for i, batch_file in enumerate(batch_files):
    batch_idx = extract_batch_idx(batch_file)
    print(f"Loading batch {batch_idx+1}/{len(batch_files)}: {Path(batch_file).name}")
    
    try:
        ckpt = torch.load(batch_file, map_location='cpu', weights_only=False)
        
        # Extract phi
        phi = None
        if 'model_state_dict' in ckpt and 'phi' in ckpt['model_state_dict']:
            phi = ckpt['model_state_dict']['phi']
        elif 'phi' in ckpt:
            phi = ckpt['phi']
        else:
            print(f"  ⚠️  Warning: No phi found in batch {batch_idx+1}")
            continue
        
        # Convert to numpy for averaging
        if torch.is_tensor(phi):
            phi_np = phi.detach().cpu().numpy()
        else:
            phi_np = np.array(phi)
        
        all_phis.append(phi_np)
        batch_info_list.append({
            'batch_idx': batch_idx,
            'filename': Path(batch_file).name,
            'shape': phi_np.shape
        })
        
        print(f"  ✓ Phi shape: {phi_np.shape}")
        print(f"  ✓ Phi range: [{phi_np.min():.6f}, {phi_np.max():.6f}]")
        print(f"  ✓ Phi mean: {phi_np.mean():.6f}, std: {phi_np.std():.6f}")
        
    except Exception as e:
        print(f"  ✗ Error loading batch {batch_idx+1}: {e}")
        continue

if len(all_phis) == 0:
    raise ValueError("No phi values found in any batch!")

# Verify all phis have same shape
shapes = [phi.shape for phi in all_phis]
if not all(s == shapes[0] for s in shapes):
    raise ValueError(f"Phi shapes differ between batches! Shapes: {set(shapes)}")

print(f"\n{'='*80}")
print("POOLING PHI")
print(f"{'='*80}")

# Stack and compute mean
phi_stack = np.stack(all_phis, axis=0)  # (n_batches, K, D, T)
phi_mean = np.mean(phi_stack, axis=0)
phi_std = np.std(phi_stack, axis=0)

print(f"Phi stack shape: {phi_stack.shape}")
print(f"Pooled phi shape: {phi_mean.shape}")
print(f"Pooled phi range: [{phi_mean.min():.6f}, {phi_mean.max():.6f}]")
print(f"Pooled phi mean: {phi_mean.mean():.6f}")
print(f"Std across batches (min): {phi_std.min():.6e}, (max): {phi_std.max():.6e}")

# Load old checkpoint to get other required fields
print(f"\n{'='*80}")
print("LOADING OLD CHECKPOINT FOR OTHER FIELDS")
print(f"{'='*80}")
print(f"Loading: {OLD_CHECKPOINT_PATH}")

old_ckpt = torch.load(OLD_CHECKPOINT_PATH, map_location='cpu', weights_only=False)

# Create new checkpoint with pooled phi
print(f"\n{'='*80}")
print("CREATING NEW CHECKPOINT")
print(f"{'='*80}")

# Convert phi_mean back to torch tensor
phi_tensor = torch.FloatTensor(phi_mean)

# Create model_state_dict with pooled phi
model_state_dict = old_ckpt.get('model_state_dict', {})
model_state_dict['phi'] = phi_tensor

# Create new checkpoint
new_checkpoint = {
    'model_state_dict': model_state_dict,
    'clusters': old_ckpt.get('clusters'),
    'signature_refs': old_ckpt.get('signature_refs'),
    'prevalence_t': old_ckpt.get('prevalence_t'),
    'disease_names': old_ckpt.get('disease_names'),
    # Keep other fields from old checkpoint if they exist
}

# Add metadata about pooling
new_checkpoint['phi_pooling_info'] = {
    'n_batches': len(all_phis),
    'batch_files': [info['filename'] for info in batch_info_list],
    'phi_std_min': float(phi_std.min()),
    'phi_std_max': float(phi_std.max()),
    'phi_std_mean': float(phi_std.mean()),
}

print(f"New checkpoint keys: {list(new_checkpoint.keys())}")
print(f"Model state dict keys: {list(new_checkpoint['model_state_dict'].keys())[:10]}...")

# Save checkpoint
print(f"\n{'='*80}")
print("SAVING CHECKPOINT")
print(f"{'='*80}")
print(f"Saving to: {OUTPUT_CHECKPOINT}")

torch.save(new_checkpoint, OUTPUT_CHECKPOINT)
print(f"✓ Saved pooled phi checkpoint to: {OUTPUT_CHECKPOINT}")

# Verify it can be loaded
print(f"\n{'='*80}")
print("VERIFICATION")
print(f"{'='*80}")
verify_ckpt = torch.load(OUTPUT_CHECKPOINT, map_location='cpu', weights_only=False)

if 'model_state_dict' in verify_ckpt and 'phi' in verify_ckpt['model_state_dict']:
    verify_phi = verify_ckpt['model_state_dict']['phi']
    if torch.is_tensor(verify_phi):
        verify_phi = verify_phi.numpy()
    
    print(f"✓ Verification successful!")
    print(f"  Phi shape in checkpoint: {verify_phi.shape}")
    print(f"  Phi range: [{verify_phi.min():.6f}, {verify_phi.max():.6f}]")
    print(f"  Phi matches pooled: {np.allclose(verify_phi, phi_mean)}")
else:
    print(f"✗ Verification failed: phi not found in checkpoint")

print(f"\n{'='*80}")
print("POOLING COMPLETE")
print(f"{'='*80}")
print(f"Checkpoint saved to: {OUTPUT_CHECKPOINT}")
print(f"Pooled from {len(all_phis)} batches")
print(f"Figure2 notebook can now use: aou_checkpoint['model_state_dict']['phi']")

