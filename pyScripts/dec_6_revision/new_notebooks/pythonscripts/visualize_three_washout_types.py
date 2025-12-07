"""
Visualization of Three Types of Washout Analyses

Creates a diagram showing:
1. Type 1: Fixed Prediction Timepoint with Variable Training Windows
2. Type 2: Same Model, Different Prediction Intervals
3. Type 3: Time Horizon with Washout (True Washout)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

def create_washout_types_figure(output_dir=None):
    """
    Create a figure showing the three types of washout analyses.
    
    Args:
        output_dir: Directory to save the figure. If None, saves to default location.
    """
    
    # Set style
    plt.style.use('default')
    sns_colors = ['#2c7fb8', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.4, top=0.95, bottom=0.05)
    
    # Common time points
    t0, t1, t2, t3, t10, t30 = 0, 1, 2, 3, 10, 30
    
    # ============================================================================
    # TYPE 1: Fixed Prediction Timepoint with Variable Training Windows
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-0.5, 2.5)
    ax1.axis('off')
    
    # Title
    ax1.text(1.5, 2.2, 'Type 1: Fixed Prediction Timepoint with Variable Training Windows', 
             fontsize=14, fontweight='bold', ha='center')
    
    # Timeline
    ax1.plot([0, 3.5], [1, 1], 'k-', linewidth=2, alpha=0.3)
    ax1.plot([0, 3.5], [0.3, 0.3], 'k-', linewidth=2, alpha=0.3)
    
    # Time labels
    for t, label in [(0, 't₀'), (1, 't₁'), (2, 't₂'), (3, 't₃')]:
        ax1.plot([t, t], [0.9, 1.1], 'k-', linewidth=1.5)
        ax1.text(t, 1.25, label, ha='center', fontsize=11, fontweight='bold')
        ax1.plot([t, t], [0.2, 0.4], 'k-', linewidth=1.5)
        ax1.text(t, 0.05, label, ha='center', fontsize=11, fontweight='bold')
    
    # Model 1: Trained up to t0, predicts t1-t2
    train1 = Rectangle((0, 0.5), 1, 0.4, facecolor=sns_colors[0], alpha=0.6, edgecolor='black', linewidth=1.5)
    pred1 = Rectangle((1, 0.5), 1, 0.4, facecolor=sns_colors[1], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    ax1.add_patch(train1)
    ax1.add_patch(pred1)
    ax1.text(0.5, 0.7, 'Train\n(t₀)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(1.5, 0.7, 'Predict\n(t₁-t₂)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(1.5, 0.3, 'Same\nInterval', ha='center', va='center', fontsize=9, fontstyle='italic', color='red')
    
    # Arrow from train to predict
    arrow1 = FancyArrowPatch((1, 0.7), (1, 0.7), 
                             arrowstyle='->', mutation_scale=20, 
                             color='black', linewidth=1.5)
    ax1.add_patch(arrow1)
    
    # Model 2: Trained up to t1, predicts t1-t2
    train2 = Rectangle((0, 0), 2, 0.4, facecolor=sns_colors[2], alpha=0.6, edgecolor='black', linewidth=1.5)
    pred2 = Rectangle((2, 0), 1, 0.4, facecolor=sns_colors[1], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    ax1.add_patch(train2)
    ax1.add_patch(pred2)
    ax1.text(1, 0.2, 'Train\n(t₀-t₁)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(2.5, 0.2, 'Predict\n(t₁-t₂)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow from train to predict
    arrow2 = FancyArrowPatch((2, 0.2), (2, 0.2), 
                             arrowstyle='->', mutation_scale=20, 
                             color='black', linewidth=1.5)
    ax1.add_patch(arrow2)
    
    # Washout period annotation
    washout1 = Rectangle((1, -0.3), 1, 0.15, facecolor='yellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
    ax1.add_patch(washout1)
    ax1.text(1.5, -0.225, 'Washout\n(excluded)', ha='center', va='center', fontsize=8, fontstyle='italic')
    
    # Legend
    legend_elements1 = [
        mpatches.Patch(facecolor=sns_colors[0], alpha=0.6, label='Training Window'),
        mpatches.Patch(facecolor=sns_colors[1], alpha=0.6, label='Prediction Interval'),
        mpatches.Patch(facecolor='yellow', alpha=0.4, label='Washout Period')
    ]
    ax1.legend(handles=legend_elements1, loc='upper right', fontsize=9, frameon=True)
    
    # ============================================================================
    # TYPE 2: Same Model, Different Prediction Intervals
    # ============================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-0.5, 2)
    ax2.axis('off')
    
    # Title
    ax2.text(1.5, 1.8, 'Type 2: Same Model, Different Prediction Intervals', 
             fontsize=14, fontweight='bold', ha='center')
    
    # Timeline
    ax2.plot([0, 3.5], [1.2, 1.2], 'k-', linewidth=2, alpha=0.3)
    
    # Time labels
    for t, label in [(0, 't₀'), (1, 't₁'), (2, 't₂'), (3, 't₃')]:
        ax2.plot([t, t], [1.1, 1.3], 'k-', linewidth=1.5)
        ax2.text(t, 1.4, label, ha='center', fontsize=11, fontweight='bold')
    
    # Single training window
    train_single = Rectangle((0, 0.8), 1, 0.3, facecolor=sns_colors[0], alpha=0.6, edgecolor='black', linewidth=1.5)
    ax2.add_patch(train_single)
    ax2.text(0.5, 0.95, 'Train\n(t₀)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Three prediction intervals
    pred_int1 = Rectangle((1, 0.5), 0.8, 0.3, facecolor=sns_colors[1], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    pred_int2 = Rectangle((1.8, 0.1), 0.8, 0.3, facecolor=sns_colors[2], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    pred_int3 = Rectangle((2.6, -0.3), 0.8, 0.3, facecolor=sns_colors[3], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    
    ax2.add_patch(pred_int1)
    ax2.add_patch(pred_int2)
    ax2.add_patch(pred_int3)
    
    ax2.text(1.4, 0.65, 'Predict\nt₀-t₁', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(2.2, 0.25, 'Predict\nt₁-t₂', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(3.0, -0.15, 'Predict\nt₂-t₃', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows from training to predictions
    arrow2a = FancyArrowPatch((1, 0.65), (1, 0.65), 
                              arrowstyle='->', mutation_scale=20, 
                              color='black', linewidth=1.5)
    arrow2b = FancyArrowPatch((1.8, 0.25), (1.8, 0.25), 
                              arrowstyle='->', mutation_scale=20, 
                              color='black', linewidth=1.5)
    arrow2c = FancyArrowPatch((2.6, -0.15), (2.6, -0.15), 
                              arrowstyle='->', mutation_scale=20, 
                              color='black', linewidth=1.5)
    ax2.add_patch(arrow2a)
    ax2.add_patch(arrow2b)
    ax2.add_patch(arrow2c)
    
    # Washout periods
    washout2a = Rectangle((1, 0.2), 0.8, 0.15, facecolor='yellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
    washout2b = Rectangle((1.8, -0.2), 0.8, 0.15, facecolor='yellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
    ax2.add_patch(washout2a)
    ax2.add_patch(washout2b)
    ax2.text(1.4, 0.275, 'Washout', ha='center', va='center', fontsize=7, fontstyle='italic')
    ax2.text(2.2, -0.125, 'Washout', ha='center', va='center', fontsize=7, fontstyle='italic')
    
    # Legend
    legend_elements2 = [
        mpatches.Patch(facecolor=sns_colors[0], alpha=0.6, label='Single Training Window'),
        mpatches.Patch(facecolor=sns_colors[1], alpha=0.6, label='Multiple Prediction Intervals'),
        mpatches.Patch(facecolor='yellow', alpha=0.4, label='Washout Periods')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=9, frameon=True)
    
    # ============================================================================
    # TYPE 3: Time Horizon with Washout (True Washout)
    # ============================================================================
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_xlim(-1, 32)
    ax3.set_ylim(-0.5, 2)
    ax3.axis('off')
    
    # Title
    ax3.text(15.5, 1.8, 'Type 3: Time Horizon with Washout (True Washout)', 
             fontsize=14, fontweight='bold', ha='center')
    
    # Timeline
    ax3.plot([0, 31], [1.2, 1.2], 'k-', linewidth=2, alpha=0.3)
    
    # Time labels (sparse for clarity)
    for t, label in [(0, 't₀'), (1, 't₁'), (10, 't₁₀'), (30, 't₃₀')]:
        ax3.plot([t, t], [1.1, 1.3], 'k-', linewidth=1.5)
        ax3.text(t, 1.4, label, ha='center', fontsize=11, fontweight='bold')
    
    # Training window
    train3 = Rectangle((0, 0.8), 1, 0.3, facecolor=sns_colors[0], alpha=0.6, edgecolor='black', linewidth=1.5)
    ax3.add_patch(train3)
    ax3.text(0.5, 0.95, 'Train\n(t₀)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 10-year prediction with washout
    washout3_10 = Rectangle((1, 0.5), 1, 0.3, facecolor='yellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
    pred3_10 = Rectangle((2, 0.5), 8, 0.3, facecolor=sns_colors[1], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    ax3.add_patch(washout3_10)
    ax3.add_patch(pred3_10)
    ax3.text(1.5, 0.65, 'Washout\nt₀-t₁', ha='center', va='center', fontsize=8, fontstyle='italic')
    ax3.text(6, 0.65, 'Predict\nt₁-t₁₀\n(10-year)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 30-year prediction with washout
    washout3_30 = Rectangle((1, 0.1), 1, 0.3, facecolor='yellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
    pred3_30 = Rectangle((2, 0.1), 28, 0.3, facecolor=sns_colors[2], alpha=0.6, edgecolor='black', linewidth=1.5, linestyle='--')
    ax3.add_patch(washout3_30)
    ax3.add_patch(pred3_30)
    ax3.text(1.5, 0.25, 'Washout\nt₀-t₁', ha='center', va='center', fontsize=8, fontstyle='italic')
    ax3.text(16, 0.25, 'Predict\nt₁-t₃₀\n(30-year)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow3a = FancyArrowPatch((1, 0.65), (1, 0.65), 
                              arrowstyle='->', mutation_scale=20, 
                              color='black', linewidth=1.5)
    arrow3b = FancyArrowPatch((1, 0.25), (1, 0.25), 
                              arrowstyle='->', mutation_scale=20, 
                              color='black', linewidth=1.5)
    ax3.add_patch(arrow3a)
    ax3.add_patch(arrow3b)
    
    # Legend
    legend_elements3 = [
        mpatches.Patch(facecolor=sns_colors[0], alpha=0.6, label='Training Window'),
        mpatches.Patch(facecolor='yellow', alpha=0.4, label='Washout Period (Excluded)'),
        mpatches.Patch(facecolor=sns_colors[1], alpha=0.6, label='10-Year Prediction'),
        mpatches.Patch(facecolor=sns_colors[2], alpha=0.6, label='30-Year Prediction')
    ]
    ax3.legend(handles=legend_elements3, loc='upper right', fontsize=9, frameon=True)
    
    # Overall title
    fig.suptitle('Three Complementary Washout Analysis Approaches', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    if output_dir is None:
        output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis/plots')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'three_washout_types_diagram.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved washout types diagram to: {output_path}")
    
    return fig

if __name__ == '__main__':
    fig = create_washout_types_figure()
    plt.show()

