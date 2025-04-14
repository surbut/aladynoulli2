import matplotlib.pyplot as plt
import numpy as np

# Results data
results = {
    'Cancer of bronchus; lung': {
        'n_patients': 90,
        'loading_ratio': 0.93,
        'risk_ratio': 5.32
    },
    'Breast cancer [female]': {
        'n_patients': 163,
        'loading_ratio': 0.96,
        'risk_ratio': 3.57
    },
    'Cancer of prostate': {
        'n_patients': 94,
        'loading_ratio': 0.93,
        'risk_ratio': 3.98
    },
    'Colon cancer': {
        'n_patients': 95,
        'loading_ratio': 0.92,
        'risk_ratio': 1.02
    },
    'Malignant neoplasm of rectum': {
        'n_patients': 54,
        'loading_ratio': 0.97,
        'risk_ratio': 1.81
    }
}

# Extract data
cancer_types = list(results.keys())
loading_ratios = [results[c]['loading_ratio'] for c in cancer_types]
risk_ratios = [results[c]['risk_ratio'] for c in cancer_types]
patient_counts = [results[c]['n_patients'] for c in cancer_types]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.2])

# Plot loading ratios
bars1 = ax1.bar(cancer_types, loading_ratios, color='skyblue', alpha=0.7)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
ax1.set_title('Signature 6 Loading Ratio by Primary Cancer Type', pad=20)
ax1.set_ylabel('Loading Ratio (Primary/Control)')
ax1.grid(True, alpha=0.3)

# Rotate x-axis labels
ax1.set_xticklabels([])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x', ha='center', va='bottom')

# Plot risk ratios
bars2 = ax2.bar(cancer_types, risk_ratios, color='lightgreen', alpha=0.7)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Secondary Cancer Risk Ratio by Primary Cancer Type', pad=20)
ax2.set_ylabel('Risk Ratio\n(P(Secondary|Primary) / P(Secondary|No Primary))')
ax2.grid(True, alpha=0.3)

# Rotate x-axis labels
ax2.set_xticklabels(cancer_types, rotation=45, ha='right')

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x', ha='center', va='bottom')

# Add patient counts as text below each bar
for i, count in enumerate(patient_counts):
    ax2.text(i, 0, f'n={count}', ha='center', va='top', rotation=90)

plt.tight_layout()
plt.savefig('cancer_analysis.pdf', bbox_inches='tight', dpi=300)
plt.close() 