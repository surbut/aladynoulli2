#!/usr/bin/env python3
"""
Export ALADYNOULLI Model Parameters for Public Release

This script exports population-level model parameters from the master checkpoint
for public release, enabling replication without individual-level data.

Exports:
- phi (phi_{k,d,t}): Disease-signature associations (K × D × T)
- psi (psi_{k,d}): Static signature-disease strength (K × D)
- mu (mu_d(t)): Disease baseline trajectories (D × T)
- Summary tables: CSV files for easy interpretation
- Metadata: JSON file with model information

Usage:
    python export_model_parameters.py \
        --checkpoint path/to/master_checkpoint.pt \
        --output_dir ./exported_parameters \
        [--prevalence path/to/prevalence_t.pt] \
        [--disease_names path/to/disease_names.pt]
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from scipy.special import logit
from typing import Optional, Dict, Any


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load master checkpoint and extract parameters."""
    print(f"\n{'='*80}")
    print("LOADING MASTER CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract phi and psi
    if 'model_state_dict' in checkpoint:
        phi = checkpoint['model_state_dict']['phi']
        psi = checkpoint['model_state_dict']['psi']
    else:
        # Fallback: check root level
        phi = checkpoint['phi']
        psi = checkpoint['psi']
    
    # Convert to numpy
    if torch.is_tensor(phi):
        phi = phi.detach().cpu().numpy()
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    print(f"✓ Loaded phi, shape: {phi.shape}")
    print(f"✓ Loaded psi, shape: {psi.shape}")
    
    # Extract metadata
    metadata = {
        'description': checkpoint.get('description', ''),
        'phi_shape': list(phi.shape),
        'psi_shape': list(psi.shape),
        'checkpoint_keys': list(checkpoint.keys()),
    }
    
    # Extract disease names if available
    disease_names = checkpoint.get('disease_names', None)
    if disease_names is not None:
        if isinstance(disease_names, torch.Tensor):
            disease_names = disease_names.tolist()
        elif isinstance(disease_names, np.ndarray):
            disease_names = disease_names.tolist()
        elif isinstance(disease_names, pd.Series):
            disease_names = disease_names.tolist()
        # Handle if disease_names are numeric (phecodes) - convert to strings
        if disease_names and isinstance(disease_names[0], (int, float)):
            disease_names = [str(d) for d in disease_names]
        metadata['disease_names'] = disease_names
        print(f"✓ Found {len(disease_names)} disease names in checkpoint")
    
    return {
        'phi': phi,
        'psi': psi,
        'checkpoint': checkpoint,
        'metadata': metadata,
    }


def compute_mu_from_prevalence(prevalence_path: Path) -> np.ndarray:
    """Compute mu (baseline trajectories) from prevalence data."""
    print(f"\n{'='*80}")
    print("COMPUTING MU FROM PREVALENCE")
    print(f"{'='*80}")
    print(f"Prevalence file: {prevalence_path}")
    
    if not prevalence_path.exists():
        raise FileNotFoundError(f"Prevalence file not found: {prevalence_path}")
    
    prevalence_data = torch.load(prevalence_path, map_location='cpu', weights_only=False)
    
    # Handle different prevalence file formats
    if isinstance(prevalence_data, dict):
        if 'prevalence_t' in prevalence_data:
            prevalence = prevalence_data['prevalence_t']
        elif 'prevalence' in prevalence_data:
            prevalence = prevalence_data['prevalence']
        else:
            # Assume the dict itself is the prevalence
            prevalence = prevalence_data
    else:
        prevalence = prevalence_data
    
    # Convert to numpy
    if torch.is_tensor(prevalence):
        prevalence = prevalence.detach().cpu().numpy()
    
    print(f"Prevalence shape: {prevalence.shape}")
    
    # Compute mu as logit of prevalence
    # Add small epsilon to avoid logit(0) or logit(1)
    epsilon = 1e-10
    prevalence_clipped = np.clip(prevalence, epsilon, 1 - epsilon)
    mu = logit(prevalence_clipped)
    
    print(f"✓ Computed mu, shape: {mu.shape}")
    print(f"  Mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    
    return mu


def load_mu_from_checkpoint(checkpoint: Dict[str, Any]) -> Optional[np.ndarray]:
    """Try to extract mu from checkpoint."""
    if 'mu' in checkpoint:
        mu = checkpoint['mu']
        if torch.is_tensor(mu):
            mu = mu.detach().cpu().numpy()
        return mu
    
    if 'model_state_dict' in checkpoint and 'mu' in checkpoint['model_state_dict']:
        mu = checkpoint['model_state_dict']['mu']
        if torch.is_tensor(mu):
            mu = mu.detach().cpu().numpy()
        return mu
    
    return None


def create_summary_tables(
    phi: np.ndarray,
    psi: np.ndarray,
    disease_names: Optional[list],
    output_dir: Path
) -> None:
    """Create human-readable summary tables."""
    print(f"\n{'='*80}")
    print("CREATING SUMMARY TABLES")
    print(f"{'='*80}")
    
    K, D, T = phi.shape
    
    # Create signature names (if not available, use indices)
    signature_names = [f"Signature {k}" for k in range(K)]
    
    # Create disease names (if not available, use indices)
    if disease_names is None:
        disease_names = [f"Disease {d}" for d in range(D)]
    
    # 1. Top diseases per signature (based on average PSI)
    print("Creating top_diseases_per_signature.csv...")
    top_diseases_data = []
    
    for k in range(K):
        # Get average PSI across time for this signature
        # If psi is 2D (K, D), use it directly; if 3D (K, D, T), average across time
        if psi.ndim == 2:
            avg_psi = psi[k, :]  # Shape: (D,)
        else:
            avg_psi = psi[k, :, :].mean(axis=-1)  # Shape: (D,)
        
        # Get top diseases by PSI
        top_indices = np.argsort(avg_psi)[::-1][:20]  # Top 20
        
        for rank, d in enumerate(top_indices, 1):
            top_diseases_data.append({
                'Signature': k,
                'Signature_Name': signature_names[k],
                'Disease_Idx': d,
                'Disease_Name': disease_names[d],
                'Avg_PSI': float(avg_psi[d]),
                'Rank': rank,
            })
    
    top_diseases_df = pd.DataFrame(top_diseases_data)
    top_diseases_path = output_dir / 'top_diseases_per_signature.csv'
    top_diseases_df.to_csv(top_diseases_path, index=False)
    print(f"✓ Saved: {top_diseases_path}")
    
    # 2. Top signature per disease (based on consistency)
    print("Creating top_signature_per_disease.csv...")
    top_signature_data = []
    
    for d in range(D):
        # Get PSI values for this disease across all signatures
        disease_psi = psi[:, d] if psi.ndim == 2 else psi[:, d, :].mean(axis=-1)
        
        # Find top signature
        top_sig_idx = np.argmax(disease_psi)
        top_psi_value = disease_psi[top_sig_idx]
        
        # Check consistency (how many signatures have similar PSI)
        threshold = top_psi_value * 0.8  # 80% of max
        n_consistent = np.sum(disease_psi >= threshold)
        
        top_signature_data.append({
            'Disease_Idx': d,
            'Disease_Name': disease_names[d],
            'Top_Signature': top_sig_idx,
            'Top_Signature_Name': signature_names[top_sig_idx],
            'PSI_Value': float(top_psi_value),
            'Consistency': 'Fully consistent' if n_consistent == 1 else f'{n_consistent} signatures',
        })
    
    top_signature_df = pd.DataFrame(top_signature_data)
    top_signature_path = output_dir / 'top_signature_per_disease.csv'
    top_signature_df.to_csv(top_signature_path, index=False)
    print(f"✓ Saved: {top_signature_path}")
    
    # 3. Signature summary statistics
    print("Creating signature_summary_statistics.csv...")
    sig_summary_data = []
    
    for k in range(K):
        sig_psi = psi[k, :] if psi.ndim == 2 else psi[k, :, :].mean(axis=-1)
        
        # Count diseases with positive PSI
        n_positive = np.sum(sig_psi > 0)
        
        sig_summary_data.append({
            'Signature': k,
            'Signature_Name': signature_names[k],
            'N_Diseases_Positive_PSI': int(n_positive),
            'Mean_PSI': float(sig_psi.mean()),
            'Max_PSI': float(sig_psi.max()),
            'Min_PSI': float(sig_psi.min()),
            'Std_PSI': float(sig_psi.std()),
        })
    
    sig_summary_df = pd.DataFrame(sig_summary_data)
    sig_summary_path = output_dir / 'signature_summary_statistics.csv'
    sig_summary_df.to_csv(sig_summary_path, index=False)
    print(f"✓ Saved: {sig_summary_path}")


def create_verification_plots(
    phi: np.ndarray,
    psi: np.ndarray,
    disease_names: Optional[list],
    output_dir: Path
) -> None:
    """Create verification plots to check export quality."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.special import expit as sigmoid
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
        
        K, D, T = phi.shape
        ages = np.arange(30, 30 + T)
        
        # Plot 1: PSI heatmap for top signatures
        print("Creating PSI heatmap...")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get PSI values (handle 2D or 3D)
        if psi.ndim == 2:
            psi_plot = psi
        else:
            psi_plot = psi.mean(axis=-1)  # Average across time
        
        # Plot top 10 signatures by max PSI
        max_psi_per_sig = psi_plot.max(axis=1)
        top_sigs = np.argsort(max_psi_per_sig)[::-1][:10]
        
        im = ax.imshow(psi_plot[top_sigs, :], aspect='auto', cmap='RdBu_r', 
                      vmin=-3, vmax=3, interpolation='nearest')
        ax.set_yticks(range(len(top_sigs)))
        ax.set_yticklabels([f'Sig {k}' for k in top_sigs])
        ax.set_xlabel('Disease Index')
        ax.set_title('PSI Values: Top 10 Signatures (verification)')
        plt.colorbar(im, ax=ax, label='PSI')
        plt.tight_layout()
        
        psi_plot_path = output_dir / 'verification_psi_heatmap.png'
        plt.savefig(psi_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {psi_plot_path}")
        
        # Plot 2: Sample phi trajectories for a few diseases
        print("Creating sample phi trajectories...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Select a few representative diseases
        sample_diseases = [0, D//4, D//2, 3*D//4] if D >= 4 else list(range(D))
        
        for idx, d in enumerate(sample_diseases[:4]):
            ax = axes[idx]
            
            # Find top signature for this disease
            if psi.ndim == 2:
                disease_psi = psi[:, d]
            else:
                disease_psi = psi[:, d, :].mean(axis=-1)
            top_sig = np.argmax(disease_psi)
            
            # Plot phi trajectory for this disease in its top signature
            phi_traj = phi[top_sig, d, :]
            prob_traj = sigmoid(phi_traj)
            
            ax.plot(ages, prob_traj, linewidth=2, label=f'Sig {top_sig}')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Probability')
            disease_name = disease_names[d] if disease_names and d < len(disease_names) else f'Disease {d}'
            ax.set_title(f'{disease_name[:40]}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Sample Disease Probability Trajectories (verification)', fontsize=14)
        plt.tight_layout()
        
        traj_plot_path = output_dir / 'verification_sample_trajectories.png'
        plt.savefig(traj_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {traj_plot_path}")
        
        print("\n💡 Tip: Run verify_exported_parameters.py with --clusters to create full signature plots")
        print("   matching the style of plot_ukb_sigs.py for complete verification.")
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not available - skipping plots")
    except Exception as e:
        print(f"⚠️  Error creating plots: {e}")


def export_parameters(
    checkpoint_path: Path,
    output_dir: Path,
    prevalence_path: Optional[Path] = None,
    disease_names_path: Optional[Path] = None,
) -> None:
    """Main export function."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ALADYNOULLI MODEL PARAMETER EXPORT")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    # Load checkpoint
    data = load_checkpoint(checkpoint_path)
    phi = data['phi']
    psi = data['psi']
    checkpoint = data['checkpoint']
    metadata = data['metadata']
    
    # Get disease names - try multiple sources
    disease_names = metadata.get('disease_names', None)
    
    # Detect cohort from checkpoint path or output directory
    checkpoint_str = str(checkpoint_path).lower()
    output_str = str(output_dir).lower()
    cohort = None
    if 'aou' in checkpoint_str or 'aou' in output_str:
        cohort = 'AOU'
    elif 'mgb' in checkpoint_str or 'mgb' in output_str:
        cohort = 'MGB'
    elif 'ukb' in checkpoint_str or 'ukb' in output_str or 'master_for_fitting' in checkpoint_str:
        cohort = 'UKB'
    
    # Try loading from CSV files if not in checkpoint
    if disease_names is None:
        # Try cohort-specific disease names files
        if cohort == 'AOU':
            # Try AOU-specific files
            aou_disease_file = Path('/Users/sarahurbut/aladynoulli2/aou_diagnames.csv')
            if aou_disease_file.exists():
                try:
                    disease_df = pd.read_csv(aou_disease_file)
                    if 'x' in disease_df.columns:
                        disease_names = disease_df['x'].tolist()
                        print(f"✓ Loaded {len(disease_names)} disease names from {aou_disease_file.name} (AOU)")
                except Exception as e:
                    print(f"⚠️  Could not load from {aou_disease_file.name}: {e}")
        elif cohort == 'MGB':
            # Try MGB-specific files
            mgb_disease_file = Path('/Users/sarahurbut/aladynoulli2/mgb_diagnames.csv')
            if mgb_disease_file.exists():
                try:
                    disease_df = pd.read_csv(mgb_disease_file)
                    if 'x' in disease_df.columns:
                        disease_names = disease_df['x'].tolist()
                        print(f"✓ Loaded {len(disease_names)} disease names from {mgb_disease_file.name} (MGB)")
                except Exception as e:
                    print(f"⚠️  Could not load from {mgb_disease_file.name}: {e}")
        else:
            # Try UKB disease names files
            ukb_disease_file = Path('/Users/sarahurbut/aladynoulli2/ukb_disease_names_phecode.csv')
            if ukb_disease_file.exists():
                try:
                    disease_df = pd.read_csv(ukb_disease_file)
                    if 'phenotype' in disease_df.columns:
                        disease_names = disease_df['phenotype'].tolist()
                        print(f"✓ Loaded {len(disease_names)} disease names from {ukb_disease_file.name} (UKB)")
                except Exception as e:
                    print(f"⚠️  Could not load from {ukb_disease_file.name}: {e}")
            
            # Try data_for_running disease names
            if disease_names is None:
                data_disease_file = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv')
                if data_disease_file.exists():
                    try:
                        disease_df = pd.read_csv(data_disease_file)
                        if 'x' in disease_df.columns:
                            disease_names = disease_df['x'].tolist()
                            print(f"✓ Loaded {len(disease_names)} disease names from {data_disease_file.name} (UKB)")
                    except Exception as e:
                        print(f"⚠️  Could not load from {data_disease_file.name}: {e}")
    
    # Try disease_names_path if provided
    if disease_names is None and disease_names_path:
        if disease_names_path.suffix == '.csv':
            try:
                disease_df = pd.read_csv(disease_names_path)
                if 'phenotype' in disease_df.columns:
                    disease_names = disease_df['phenotype'].tolist()
                elif 'x' in disease_df.columns:
                    disease_names = disease_df['x'].tolist()
                print(f"✓ Loaded {len(disease_names)} disease names from {disease_names_path.name}")
            except Exception as e:
                print(f"⚠️  Could not load CSV: {e}")
        else:
            # Try as torch file
            try:
                disease_names_data = torch.load(disease_names_path, map_location='cpu', weights_only=False)
                if isinstance(disease_names_data, list):
                    disease_names = disease_names_data
                elif isinstance(disease_names_data, dict) and 'disease_names' in disease_names_data:
                    disease_names = disease_names_data['disease_names']
                print(f"✓ Loaded {len(disease_names)} disease names from {disease_names_path.name}")
            except Exception as e:
                print(f"⚠️  Could not load torch file: {e}")
    
    # Get or compute mu
    print(f"\n{'='*80}")
    print("EXTRACTING MU PARAMETERS")
    print(f"{'='*80}")
    
    mu = load_mu_from_checkpoint(checkpoint)
    
    if mu is None:
        if prevalence_path:
            mu = compute_mu_from_prevalence(Path(prevalence_path))
        else:
            print("⚠️  Warning: mu not found in checkpoint and no prevalence file provided.")
            print("   Mu will not be exported. Provide --prevalence to compute mu.")
            mu = None
    else:
        print(f"✓ Loaded mu from checkpoint, shape: {mu.shape}")
    
    # Export full arrays as NPY
    print(f"\n{'='*80}")
    print("EXPORTING FULL PARAMETER ARRAYS")
    print(f"{'='*80}")
    
    phi_path = output_dir / 'phi_master_pooled.npy'
    np.save(phi_path, phi)
    print(f"✓ Saved: {phi_path} (shape: {phi.shape})")
    
    psi_path = output_dir / 'psi_master.npy'
    np.save(psi_path, psi)
    print(f"✓ Saved: {psi_path} (shape: {psi.shape})")
    
    if mu is not None:
        mu_path = output_dir / 'mu_baselines.npy'
        np.save(mu_path, mu)
        print(f"✓ Saved: {mu_path} (shape: {mu.shape})")
    
    # Export disease names as CSV
    if disease_names:
        print(f"\n{'='*80}")
        print("EXPORTING DISEASE NAMES")
        print(f"{'='*80}")
        
        # Create disease names dataframe
        disease_df_export = pd.DataFrame({
            'Disease_Idx': range(len(disease_names)),
            'Disease_Name': disease_names,
        })
        
        disease_names_path = output_dir / 'disease_names.csv'
        disease_df_export.to_csv(disease_names_path, index=False)
        print(f"✓ Saved: {disease_names_path} ({cohort if cohort else 'unknown'} cohort, {len(disease_names)} diseases)")
    
    # Create summary tables
    create_summary_tables(phi, psi, disease_names, output_dir)
    
    # Export metadata
    print(f"\n{'='*80}")
    print("EXPORTING METADATA")
    print(f"{'='*80}")
    
    # Add age bins
    K, D, T = phi.shape
    age_bins = np.arange(30, 30 + T).tolist()  # Ages 30 to 30+T-1
    
    export_metadata = {
        'model_name': 'ALADYNOULLI',
        'checkpoint_path': str(checkpoint_path),
        'export_date': pd.Timestamp.now().isoformat(),
        'parameters': {
            'phi': {
                'shape': list(phi.shape),
                'description': 'Disease-signature associations (phi_{k,d,t})',
                'dimensions': ['K (signatures)', 'D (diseases)', 'T (timepoints)'],
                'file': 'phi_master_pooled.npy',
            },
            'psi': {
                'shape': list(psi.shape),
                'description': 'Static signature-disease strength (psi_{k,d})',
                'dimensions': ['K (signatures)', 'D (diseases)'],
                'file': 'psi_master.npy',
            },
        },
        'model_structure': {
            'n_signatures': int(K),
            'n_diseases': int(D),
            'n_timepoints': int(T),
            'age_range': f'{age_bins[0]}-{age_bins[-1]} years',
            'age_bins': age_bins,
        },
    }
    
    if mu is not None:
        export_metadata['parameters']['mu'] = {
            'shape': list(mu.shape),
            'description': 'Disease baseline trajectories (mu_d(t))',
            'dimensions': ['D (diseases)', 'T (timepoints)'],
            'file': 'mu_baselines.npy',
        }
    
    if disease_names:
        export_metadata['disease_names'] = disease_names
        export_metadata['n_diseases_with_names'] = len(disease_names)
    
    # Add original checkpoint metadata
    export_metadata['original_checkpoint'] = {
        'description': metadata.get('description', ''),
        'keys': metadata.get('checkpoint_keys', []),
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    print(f"✓ Saved: {metadata_path}")
    
    # Create README
    print(f"\n{'='*80}")
    print("CREATING README")
    print(f"{'='*80}")
    
    readme_content = f"""# ALADYNOULLI Model Parameters

This directory contains population-level model parameters exported from the ALADYNOULLI master checkpoint.

## Files

### Parameter Arrays (NumPy format)
- **phi_master_pooled.npy**: Disease-signature associations ($\\phi_{{k,d,t}}$)
  - Shape: ({K}, {D}, {T})
  - Dimensions: [signatures, diseases, timepoints]
  - Description: Time-varying logit probabilities for each disease within each signature
  
- **psi_master.npy**: Static signature-disease strength ($\\psi_{{k,d}}$)
  - Shape: ({K}, {D})
  - Dimensions: [signatures, diseases]
  - Description: Overall strength of association between each signature and disease

"""
    
    if mu is not None:
        readme_content += f"""- **mu_baselines.npy**: Disease baseline trajectories ($\\mu_d(t)$)
  - Shape: ({D}, {T})
  - Dimensions: [diseases, timepoints]
  - Description: Logit-transformed population prevalence for each disease over time

"""
    
    readme_content += f"""### Summary Tables (CSV format)
- **top_diseases_per_signature.csv**: Top diseases for each signature ranked by PSI
- **top_signature_per_disease.csv**: Primary signature assignment for each disease
- **signature_summary_statistics.csv**: Summary statistics for each signature

### Metadata
- **model_metadata.json**: Complete model metadata including shapes, descriptions, and age bins

## Usage

### Loading Parameters

```python
import numpy as np

# Load parameters
phi = np.load('phi_master_pooled.npy')  # Shape: ({K}, {D}, {T})
psi = np.load('psi_master.npy')          # Shape: ({K}, {D})
mu = np.load('mu_baselines.npy')         # Shape: ({D}, {T})  # if available

# Access specific values
phi_kdt = phi[k, d, t]  # Disease d in signature k at time t
psi_kd = psi[k, d]      # Overall association of disease d with signature k
mu_dt = mu[d, t]        # Baseline logit prevalence of disease d at time t
```

### Computing Disease Probabilities

The disease probability for individual $i$, disease $d$, at time $t$ is:

$$\\pi_{{i,d,t}} = \\kappa \\cdot \\sum_{{k=1}}^{{K}} \\theta_{{i,k,t}} \\cdot \\text{{sigmoid}}(\\phi_{{k,d,t}})$$

where:
- $\\theta_{{i,k,t}}$ are individual signature loadings (not included in this export)
- $\\phi_{{k,d,t}}$ are the values in `phi_master_pooled.npy`
- $\\kappa$ is a global calibration parameter

## Model Structure

- **Signatures (K)**: {K} disease signatures (20 disease signatures + 1 health signature)
- **Diseases (D)**: {D} diseases defined by PheCodes
- **Timepoints (T)**: {T} timepoints representing ages {age_bins[0]}-{age_bins[-1]} years

## Notes

- These are **population-level** parameters only. Individual-level parameters ($\\lambda_{{i,k,t}}$) are not included due to data sharing restrictions.
- Parameters were pooled/averaged across training batches for robustness.
- See `model_metadata.json` for complete metadata and descriptions.

## Citation

If you use these parameters, please cite the ALADYNOULLI publication.

## Contact

For questions about these parameters or the model, please see the main repository:
https://github.com/surbut/aladynoulli2
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Saved: {readme_path}")
    
    # Create visualization plots to verify exports
    print(f"\n{'='*80}")
    print("CREATING VERIFICATION PLOTS")
    print(f"{'='*80}")
    try:
        create_verification_plots(phi, psi, disease_names, output_dir)
    except Exception as e:
        print(f"⚠️  Could not create verification plots: {e}")
        print("   (This is optional - exports are still complete)")
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"All files saved to: {output_dir}")
    print(f"\nExported files:")
    print(f"  - phi_master_pooled.npy")
    print(f"  - psi_master.npy")
    if mu is not None:
        print(f"  - mu_baselines.npy")
    if disease_names:
        print(f"  - disease_names.csv")
    print(f"  - top_diseases_per_signature.csv")
    print(f"  - top_signature_per_disease.csv")
    print(f"  - signature_summary_statistics.csv")
    print(f"  - model_metadata.json")
    print(f"  - README.md")


def main():
    parser = argparse.ArgumentParser(
        description='Export ALADYNOULLI model parameters for public release',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python export_model_parameters.py \\
      --checkpoint aou_model_master_correctedE.pt \\
      --output_dir ./exported_parameters

  # Export with prevalence to compute mu
  python export_model_parameters.py \\
      --checkpoint aou_model_master_correctedE.pt \\
      --output_dir ./exported_parameters \\
      --prevalence prevalence_t_corrected.pt

  # Export with disease names
  python export_model_parameters.py \\
      --checkpoint aou_model_master_correctedE.pt \\
      --output_dir ./exported_parameters \\
      --disease_names disease_names.pt
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to master checkpoint file (.pt)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for exported parameters'
    )
    
    parser.add_argument(
        '--prevalence',
        type=str,
        default=None,
        help='Path to prevalence file (.pt) to compute mu (optional)'
    )
    
    parser.add_argument(
        '--disease_names',
        type=str,
        default=None,
        help='Path to disease names file (.pt) (optional, will try to extract from checkpoint)'
    )
    
    args = parser.parse_args()
    
    export_parameters(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        prevalence_path=Path(args.prevalence) if args.prevalence else None,
        disease_names_path=Path(args.disease_names) if args.disease_names else None,
    )


if __name__ == '__main__':
    main()

