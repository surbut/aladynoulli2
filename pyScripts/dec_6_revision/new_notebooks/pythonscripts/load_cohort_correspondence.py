"""
Load cohort checkpoints and generate cross-cohort cluster correspondence visualization.

This script loads UKB, MGB, and AoU model checkpoints and generates heatmaps
showing cluster correspondence between biobanks using plot_disease_blocks.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add path to utils for plot_disease_blocks
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import plot_disease_blocks


def load_cohort_checkpoints(
    ukb_path: str = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam.pt',
    mgb_path: str = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt',
    aou_path: str = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_AOU.pt'
) -> Tuple[Dict, Dict, Dict]:
    """
    Load model checkpoints for UKB, MGB, and AoU cohorts.
    
    Args:
        ukb_path: Path to UKB checkpoint file
        mgb_path: Path to MGB checkpoint file
        aou_path: Path to AoU checkpoint file
        
    Returns:
        Tuple of (ukb_checkpoint, mgb_checkpoint, aou_checkpoint) dictionaries
    """
    print("Loading cohort checkpoints...")
    
    # Load UKB checkpoint
    print(f"  Loading UKB checkpoint from: {ukb_path}")
    ukb_checkpoint = torch.load(ukb_path, map_location='cpu')
    
    # Load MGB checkpoint
    print(f"  Loading MGB checkpoint from: {mgb_path}")
    mgb_checkpoint = torch.load(mgb_path, map_location='cpu')
    
    # Load AoU checkpoint
    print(f"  Loading AoU checkpoint from: {aou_path}")
    aou_checkpoint = torch.load(aou_path, map_location='cpu')
    
    # Print summary information
    print(f"\nUKB checkpoint keys: {list(ukb_checkpoint.keys())}")
    print(f"UKB - Number of diseases: {len(ukb_checkpoint['disease_names'])}")
    if hasattr(ukb_checkpoint['clusters'], 'max'):
        print(f"UKB - Number of signatures: {ukb_checkpoint['clusters'].max() + 1}")
    else:
        print(f"UKB - Clusters type: {type(ukb_checkpoint['clusters'])}")
    
    print(f"\nMGB - Number of diseases: {len(mgb_checkpoint['disease_names'])}")
    if hasattr(mgb_checkpoint['clusters'], 'max'):
        print(f"MGB - Number of signatures: {mgb_checkpoint['clusters'].max() + 1}")
    
    print(f"\nAoU - Number of diseases: {len(aou_checkpoint['disease_names'])}")
    if hasattr(aou_checkpoint['clusters'], 'max'):
        print(f"AoU - Number of signatures: {aou_checkpoint['clusters'].max() + 1}")
    
    return ukb_checkpoint, mgb_checkpoint, aou_checkpoint


def extract_clusters_and_diseases(checkpoint: Dict) -> Tuple[np.ndarray, list]:
    """
    Extract clusters and disease names from a checkpoint.
    
    Args:
        checkpoint: Model checkpoint dictionary
        
    Returns:
        Tuple of (clusters array, disease_names list)
    """
    clusters = checkpoint['clusters']
    disease_names = checkpoint['disease_names']
    
    # Convert clusters to numpy array if needed
    if isinstance(clusters, torch.Tensor):
        clusters = clusters.cpu().numpy()
    elif not isinstance(clusters, np.ndarray):
        clusters = np.array(clusters)
    
    # Convert disease_names to list if needed
    if isinstance(disease_names, (list, tuple)):
        disease_names = list(disease_names)
    elif hasattr(disease_names, 'values'):
        disease_names = disease_names.values.tolist()
    elif isinstance(disease_names, np.ndarray):
        disease_names = disease_names.tolist()
    
    return clusters, disease_names


def generate_correspondence_plot(
    ukb_checkpoint: Dict,
    mgb_checkpoint: Dict,
    aou_checkpoint: Dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> Dict:
    """
    Generate cross-cohort cluster correspondence visualization.
    
    This generates Supplementary Figure S5: Cross-cohort validation heatmaps.
    
    Args:
        ukb_checkpoint: UKB model checkpoint dictionary
        mgb_checkpoint: MGB model checkpoint dictionary
        aou_checkpoint: AoU model checkpoint dictionary
        output_path: Optional path to save the plot. If None, uses default S5 location.
        figsize: Figure size tuple (width, height)
        
    Returns:
        Dictionary with cross-tabulation results and best matches
    """
    if output_path is None:
        # Save to S5 location (Supplementary Figure S5)
        output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s5/S5.pdf'
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating cluster correspondence plot...")
    print(f"  Output path: {output_path}")
    
    # Verify we're saving to S5 location, not fig2
    if 'fig2' in output_path.lower() or 'disease_blocks' in output_path.lower():
        raise ValueError(f"ERROR: Output path should be for S5 (supp/s5/), not fig2! Got: {output_path}")
    
    # Call plot_disease_blocks function
    results = plot_disease_blocks(
        mgb_checkpoint=mgb_checkpoint,
        aou_checkpoint=aou_checkpoint,
        ukb_checkpoint=ukb_checkpoint,
        output_path=output_path,  # Explicitly pass the S5 path
        figsize=figsize
    )
    
    print(f"  ✓ Plot saved successfully to: {output_path}")
    
    return results


def main():
    """
    Main function to load checkpoints and generate correspondence plot.
    """
    # Load checkpoints
    ukb_checkpoint, mgb_checkpoint, aou_checkpoint = load_cohort_checkpoints()
    
    # Extract clusters and disease names for reference
    ukb_clusters, ukb_diseases = extract_clusters_and_diseases(ukb_checkpoint)
    mgb_clusters, mgb_diseases = extract_clusters_and_diseases(mgb_checkpoint)
    aou_clusters, aou_diseases = extract_clusters_and_diseases(aou_checkpoint)
    
    print(f"\nExtracted data:")
    print(f"  UKB: {len(ukb_diseases)} diseases, clusters shape: {ukb_clusters.shape}")
    print(f"  MGB: {len(mgb_diseases)} diseases, clusters shape: {mgb_clusters.shape}")
    print(f"  AoU: {len(aou_diseases)} diseases, clusters shape: {aou_clusters.shape}")
    
    # Generate correspondence plot (S5: Cross-cohort validation heatmaps)
    results = generate_correspondence_plot(
        ukb_checkpoint=ukb_checkpoint,
        mgb_checkpoint=mgb_checkpoint,
        aou_checkpoint=aou_checkpoint,
        output_path='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s5/S5.pdf'
    )
    
    return {
        'ukb_checkpoint': ukb_checkpoint,
        'mgb_checkpoint': mgb_checkpoint,
        'aou_checkpoint': aou_checkpoint,
        'ukb_clusters': ukb_clusters,
        'ukb_diseases': ukb_diseases,
        'mgb_clusters': mgb_clusters,
        'mgb_diseases': mgb_diseases,
        'aou_clusters': aou_clusters,
        'aou_diseases': aou_diseases,
        'correspondence_results': results
    }


if __name__ == '__main__':
    results = main()

