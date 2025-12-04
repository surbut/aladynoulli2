#!/usr/bin/env python3
"""
Quick script to check the shape of phi and psi in master_for_fitting_pooled_all_data.pt
"""
import torch
import sys
import os

# Try to find the checkpoint
checkpoint_paths = [
    'master_for_fitting_pooled_all_data.pt',
    '../data_for_running/master_for_fitting_pooled_all_data.pt',
    './data_for_running/master_for_fitting_pooled_all_data.pt',
    '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_all_data.pt',
]

checkpoint_path = None
for path in checkpoint_paths:
    if os.path.exists(path):
        checkpoint_path = path
        break

if checkpoint_path is None:
    print("❌ Could not find master_for_fitting_pooled_all_data.pt")
    print("   Tried paths:")
    for path in checkpoint_paths:
        print(f"     - {path}")
    sys.exit(1)

print(f"✓ Found checkpoint: {checkpoint_path}")
print(f"\n{'='*60}")
print("Loading checkpoint...")
print(f"{'='*60}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract phi and psi
    if 'model_state_dict' in checkpoint:
        phi = checkpoint['model_state_dict']['phi']
        psi = checkpoint['model_state_dict']['psi']
    else:
        phi = checkpoint['phi']
        psi = checkpoint['psi']
    
    # Convert to numpy if tensor
    if torch.is_tensor(phi):
        phi_shape = phi.shape
        phi_np = phi.cpu().numpy()
    else:
        phi_shape = phi.shape
        phi_np = phi
    
    if torch.is_tensor(psi):
        psi_shape = psi.shape
        psi_np = psi.cpu().numpy()
    else:
        psi_shape = psi.shape
        psi_np = psi
    
    print(f"\nPhi shape: {phi_shape}")
    print(f"Psi shape: {psi_shape}")
    
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    
    # Check if phi has 21 entries (K=20 + healthy state)
    if phi_shape[0] == 21:
        print("✅ Phi has 21 entries (20 disease signatures + 1 healthy state)")
    elif phi_shape[0] == 20:
        print("⚠️  Phi has 20 entries (only disease signatures, no healthy state)")
        print("   This would cause IndexError with the buggy code!")
    else:
        print(f"⚠️  Phi has {phi_shape[0]} entries (unexpected)")
    
    # Check if psi has 21 entries
    if psi_shape[0] == 21:
        print("✅ Psi has 21 entries (20 disease signatures + 1 healthy state)")
    elif psi_shape[0] == 20:
        print("⚠️  Psi has 20 entries (only disease signatures, no healthy state)")
    else:
        print(f"⚠️  Psi has {psi_shape[0]} entries (unexpected)")
    
    # Print description if available
    if 'description' in checkpoint:
        print(f"\nDescription: {checkpoint['description']}")
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Phi: {phi_shape[0]} x {phi_shape[1]} x {phi_shape[2]}")
    print(f"Psi: {psi_shape[0]} x {psi_shape[1]}")
    
    if phi_shape[0] == 21:
        print("\n✅ CONFIRMED: Phi includes healthy state (21 entries)")
        print("   The buggy code (looping over K_total=21) would work!")
    else:
        print("\n⚠️  Phi does NOT include healthy state")
        print("   The buggy code would crash with IndexError!")
        
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

