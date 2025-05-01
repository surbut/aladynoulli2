import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from typing import Optional
from fig3_utils import (
    plot_single_patient_dynamics_ax,
    visualize_disease_contribution_breakdown_ax,
    plot_disease_signature_clusters_all_batches_ax
)
import pandas as pd

def create_composite_figure(
    model_path: str,
    patient_idx: int,
    disease_idx: int,
    disease_name: str,
    sig_refs_path: str,
    output_path: Optional[str] = None
):
    """
    Create a composite figure combining multiple visualization types.
    
    Args:
        model_path: Path to the model.pt file
        patient_idx: Index of the patient to analyze
        disease_idx: Index of the disease to analyze
        disease_name: Name of the disease
        sig_refs_path: Path to signature references
        output_path: Optional path to save the figure
    """
    # Load model data
    model_data = torch.load(model_path, map_location='cpu')
    lambda_values = model_data['model_state_dict']['lambda_']
    phi_values = model_data['model_state_dict']['phi']  # This is the time-varying associations
    time_points = np.arange(model_data['Y'].shape[2])
    
    # Convert to numpy if needed and check shapes
    if torch.is_tensor(lambda_values):
        lambda_values = lambda_values.detach().cpu().numpy()
    if torch.is_tensor(phi_values):
        phi_values = phi_values.detach().cpu().numpy()
    
    print(f"lambda_values shape: {lambda_values.shape}")
    print(f"phi_values shape: {phi_values.shape}")
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # Create axes for each subplot
    ax1 = fig.add_subplot(gs[0, 0])  # Patient dynamics
    ax2 = fig.add_subplot(gs[0, 1])  # Disease contribution breakdown
    ax3 = fig.add_subplot(gs[1, :])  # Signature clusters
    ax4 = fig.add_subplot(gs[2, :])  # Additional analysis if needed
    
    # Plot 1: Single Patient Dynamics
    plot_single_patient_dynamics_ax(
        model_path=model_path,
        patient_idx=patient_idx,
        sig_refs_path=sig_refs_path,
        ax1=ax1,
        ax2=ax2,
        ax3=ax3
    )
    
    # Plot 2: Disease Contribution Breakdown
    if len(phi_values.shape) == 3:
        # For 3D phi_values (K, D, T)
        K = lambda_values.shape[1]  # Number of signatures
        D = phi_values.shape[1]  # Number of diseases
        
        if disease_idx >= D:
            raise ValueError(f"Invalid disease_idx {disease_idx} for D={D}")
            
        # Get the disease-specific phi values (K, T)
        phi_disease = phi_values[:, disease_idx, :]
        print(f"phi_disease shape: {phi_disease.shape}")
        
        # Get patient's lambda values (K, T)
        patient_lambda = lambda_values[patient_idx, :, :]
        print(f"patient_lambda shape: {patient_lambda.shape}")
        
        # Calculate contributions
        phi_probs_disease = 1 / (1 + np.exp(-phi_disease))
        contributions = patient_lambda * phi_probs_disease
        print(f"contributions shape: {contributions.shape}")
        
        # Get colors for signatures
        colors = get_signature_colors(K)
        
        # Plot the contributions over time
        for k in range(K):
            ax2.plot(time_points, contributions[k], 
                    color=colors[k], 
                    linewidth=2,
                    label=f'Signature {k}')
        
        # Add total contribution line
        total_contrib = np.sum(contributions, axis=0)
        ax2.plot(time_points, total_contrib, 
                'k--', 
                linewidth=2,
                label='Total')
        
        # Style the plot
        ax2.set_title(f'Disease Contribution Breakdown for {disease_name}', fontsize=14, pad=15)
        ax2.set_xlabel('Age', fontsize=12)
        ax2.set_ylabel('Contribution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    else:
        # For other shapes, use the original function
        visualize_disease_contribution_breakdown_ax(
            lambda_values_np=lambda_values,
            phi_values_np=phi_values,
            individual_idx=patient_idx,
            disease_idx=disease_idx,
            disease_name=disease_name,
            time_points=time_points,
            ax1=ax1,
            ax2=ax2,
            ax3=ax3
        )
    
    # Plot 3: Disease Signature Clusters
    plot_disease_signature_clusters_all_batches_ax(
        disease_idx=disease_idx,
        axes=[ax3, ax4]  # Pass both remaining axes
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved composite figure to {output_path}")
    
    return fig

def plot_prs_effects_by_cluster(genetic_results, n_top=5, figsize=(12, 8)):
    """
    Plot the top PRS effects for each cluster in a bar plot.
    
    Args:
        genetic_results: Dictionary containing genetic analysis results
        n_top: Number of top effects to show per cluster
        figsize: Figure size
    """
    # Debug print to see what's in the genetic results
    print("\nGenetic Results DataFrame:")
    print(genetic_results['genetic_df'].head())
    print("\nColumns in genetic results:", genetic_results['genetic_df'].columns)
    
    # Load PRS names from CSV
    prs_names_df = pd.read_csv('/Users/sarahurbut/Dropbox/prs_names.csv')
    prs_names = prs_names_df['x'].tolist()
    print("\nAvailable PRS names:", prs_names)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    n_clusters = len(genetic_results['genetic_df']['Cluster'].unique())
    
    for cluster in range(n_clusters):
        ax = axes[cluster]
        
        # Get top effects for this cluster
        cluster_results = genetic_results['genetic_df'][
            (genetic_results['genetic_df']['Cluster'] == cluster) & 
            (genetic_results['genetic_df']['Significant'])
        ].sort_values('Effect_Size', key=abs, ascending=False)
        
        if len(cluster_results) > 0:
            # Take top n effects
            top_effects = cluster_results.head(n_top)
            print(f"\nTop effects for cluster {cluster}:")
            print(top_effects[['Factor', 'Effect_Size', 'P_Value_Corrected']])
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_effects))
            effect_sizes = top_effects['Effect_Size']
            
            # Get PRS names directly from the Factor column
            factor_names = top_effects['Factor']
            
            # Color bars based on effect direction
            colors = ['red' if x < 0 else 'blue' for x in effect_sizes]
            
            ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(factor_names, fontsize=10)
            
            # Add significance stars
            for i, p_val in enumerate(top_effects['P_Value_Corrected']):
                if p_val < 0.001:
                    ax.text(effect_sizes.iloc[i], i, '***', va='center')
                elif p_val < 0.01:
                    ax.text(effect_sizes.iloc[i], i, '**', va='center')
                elif p_val < 0.05:
                    ax.text(effect_sizes.iloc[i], i, '*', va='center')
            
            # Add zero line
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Style
            ax.set_title(f'Cluster {cluster}\nTop {n_top} PRS Effects', pad=15)
            ax.set_xlabel('Effect Size')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add legend for effect direction
            if cluster == 0:
                ax.legend([plt.Rectangle((0,0),1,1,fc='blue'), 
                          plt.Rectangle((0,0),1,1,fc='red')],
                         ['Positive Effect', 'Negative Effect'],
                         loc='upper left')
        else:
            ax.text(0.5, 0.5, 'No significant effects', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cluster {cluster}')
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Example parameters from figure3.ipynb
    model_path = '/Users/sarahurbut/Dropbox/resultshighamp/results/output_40000_50000/model.pt'
    sig_refs_path = '/Users/sarahurbut/Dropbox/data_for_running/reference_trajectories.pt'
    patient_idx = 43672 - 40000  # Local index within the batch
    disease_idx = 1
    disease_name = "Myocardial Infarction"
    output_path = "composite_figure.png"
    prs_names_path = '/Users/sarahurbut/Dropbox/prs_names.csv'
    
    # Create and display the composite figure
    fig = create_composite_figure(
        model_path=model_path,
        patient_idx=patient_idx,
        disease_idx=disease_idx,
        disease_name=disease_name,
        sig_refs_path=sig_refs_path,
        output_path=output_path
    )
    
    # Analyze genetic data and plot PRS effects
    genetic_results = analyze_genetic_data_by_cluster(
        disease_idx=66,  # Myocardial Infarction
        n_clusters=3,
        prs_names_file=prs_names_path,
        heatmap_output_path="/Users/sarahurbut/aladynoulli2/pyScripts/figures_for_science/figure4/fig4_genetic_heatmap_traj_MDD.pdf"
    )
    
    # Plot PRS effects
    prs_fig = plot_prs_effects_by_cluster(genetic_results, n_top=5)
    plt.show() 