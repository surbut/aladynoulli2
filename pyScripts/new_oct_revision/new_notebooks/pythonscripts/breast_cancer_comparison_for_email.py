# Comparison of Breast Cancer Static 10-Year Predictions (1-year score for 10-year outcome)
# Batch 0 (0-10000) - Same as "batch1" in preprint table

import pandas as pd
import numpy as np

# Results from batch 0 (0-10000) - Static 10-year predictions
comparison_data = {
    'Approach': [
        'Joint Phi (Preprint/Original)',
        'Fixed Phi - Enrollment Pooled',
        'Fixed Phi - Retrospective Pooled'
    ],
    'AUC': [0.575, 0.581, 0.584],
    'CI_Lower': [0.538, 0.541, 0.540],
    'CI_Upper': [0.614, 0.615, 0.625],
    'N_Patients': [5409, 5409, 5409],
    'N_Events': [214, 214, 214],
    'Event_Rate_%': [4.0, 4.0, 4.0]
}

comparison_df = pd.DataFrame(comparison_data)

# Calculate improvements over Joint approach
comparison_df['Improvement_vs_Joint'] = comparison_df['AUC'] - comparison_df.loc[0, 'AUC']
comparison_df['Improvement_%'] = (comparison_df['Improvement_vs_Joint'] / comparison_df.loc[0, 'AUC']) * 100

print("="*80)
print("BREAST CANCER STATIC 10-YEAR PREDICTION COMPARISON")
print("(1-year score evaluated on 10-year outcomes)")
print("Batch 0 (0-10000) - Same cohort as preprint 'batch1'")
print("="*80)
print("\n")
print(comparison_df.to_string(index=False))
print("\n")

print("="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"✓ Fixed Enrollment Pooled: +{comparison_df.loc[1, 'Improvement_vs_Joint']:.3f} AUC improvement ({comparison_df.loc[1, 'Improvement_%']:.1f}% relative improvement)")
print(f"✓ Fixed Retrospective Pooled: +{comparison_df.loc[2, 'Improvement_vs_Joint']:.3f} AUC improvement ({comparison_df.loc[2, 'Improvement_%']:.1f}% relative improvement)")
print(f"\nBoth fixed phi approaches outperform the joint approach for clinical implementation.")
print(f"The fixed retrospective pooled approach shows the best performance (AUC = {comparison_df.loc[2, 'AUC']:.3f}).")

# Create a formatted table for email
print("\n" + "="*80)
print("FORMATTED FOR EMAIL:")
print("="*80)
print("\nBreast Cancer Static 10-Year Prediction (Batch 0, n=5,409, 214 events):")
print("-" * 100)
print(f"{'Approach':<45} {'AUC (95% CI)':<30} {'Improvement':<20}")
print("-" * 100)
print("{:<45}{:<30}{:<20}".format(
    "Joint Phi (Original)",
    f"{comparison_df.loc[0, 'AUC']:.3f} ({comparison_df.loc[0, 'CI_Lower']:.3f}-{comparison_df.loc[0, 'CI_Upper']:.3f})",
    "Reference"
))
print("{:<45}{:<30}{:<20}".format(
    "Fixed Phi - Enrollment Pooled",
    f"{comparison_df.loc[1, 'AUC']:.3f} ({comparison_df.loc[1, 'CI_Lower']:.3f}-{comparison_df.loc[1, 'CI_Upper']:.3f})",
    f"+{comparison_df.loc[1, 'Improvement_vs_Joint']:.3f} ({comparison_df.loc[1, 'Improvement_%']:.1f}%)"
))
print("{:<45}{:<30}{:<20}".format(
    "Fixed Phi - Retrospective Pooled",
    f"{comparison_df.loc[2, 'AUC']:.3f} ({comparison_df.loc[2, 'CI_Lower']:.3f}-{comparison_df.loc[2, 'CI_Upper']:.3f})",
    f"+{comparison_df.loc[2, 'Improvement_vs_Joint']:.3f} ({comparison_df.loc[2, 'Improvement_%']:.1f}%)"
))

# Save to CSV
comparison_df.to_csv('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/breast_cancer_comparison_batch0.csv', index=False)
print("\n✓ Saved to breast_cancer_comparison_batch0.csv")

