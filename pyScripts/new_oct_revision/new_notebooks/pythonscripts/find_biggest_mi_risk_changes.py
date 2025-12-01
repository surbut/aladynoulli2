# Find patients with biggest MI risk changes between enrollment and year 9
# Simple, straightforward approach

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find MI disease index
mi_idx = None
for i, name in enumerate(disease_names):
    if 'myocardial infarction' in name.lower():
        mi_idx = i
        print(f"✓ Found MI at index {i}: {name}")
        break

if mi_idx is None:
    print("⚠️  MI not found, using index 113 as fallback")
    mi_idx = 113

print(f"\n{'='*80}")
print("FINDING PATIENTS WITH BIGGEST MI RISK CHANGES")
print(f"{'='*80}\n")

# Calculate MI risk at enrollment (year 0) and year 9 for all patients
print("Calculating MI risks for all patients...")
mi_risks_year0 = []
mi_risks_year9 = []
patient_indices = []

for p in range(len(Y_batch)):
    # Skip if patient has MI at enrollment
    if Y_batch[p, mi_idx, 0].item() > 0:
        continue
    
    # Get predictions at year 0 and year 9
    pi_0 = pi_batches[0][p]
    pi_9 = pi_batches[min(9, len(pi_batches)-1)][p]
    
    # Get MI risk (1-year risk at time 0)
    risk_0 = pi_0[mi_idx, 0].item()
    risk_9 = pi_9[mi_idx, 0].item()
    
    # Only consider patients with some baseline risk
    if risk_0 > 0:
        mi_risks_year0.append(risk_0)
        mi_risks_year9.append(risk_9)
        patient_indices.append(p)

mi_risks_year0 = np.array(mi_risks_year0)
mi_risks_year9 = np.array(mi_risks_year9)

# Calculate changes
absolute_changes = mi_risks_year9 - mi_risks_year0
relative_changes = mi_risks_year9 / mi_risks_year0  # Only valid where risk_0 > 0

# Find biggest absolute increase
biggest_abs_increase_idx = np.argmax(absolute_changes)
patient_abs = patient_indices[biggest_abs_increase_idx]

# Find biggest relative increase (among patients with meaningful baseline risk)
# Filter for patients with baseline risk > median
median_baseline = np.median(mi_risks_year0)
high_baseline_mask = mi_risks_year0 > median_baseline
if high_baseline_mask.sum() > 0:
    relative_changes_filtered = relative_changes[high_baseline_mask]
    patient_indices_filtered = [patient_indices[i] for i in range(len(patient_indices)) if high_baseline_mask[i]]
    biggest_rel_increase_idx = np.argmax(relative_changes_filtered)
    patient_rel = patient_indices_filtered[biggest_rel_increase_idx]
else:
    biggest_rel_increase_idx = np.argmax(relative_changes)
    patient_rel = patient_indices[biggest_rel_increase_idx]

print(f"Total patients analyzed: {len(patient_indices)}")
print(f"Median baseline MI risk: {median_baseline:.6f}")
print(f"\n{'='*80}")
print("BIGGEST ABSOLUTE INCREASE")
print(f"{'='*80}")
print(f"Patient #{patient_abs}:")
print(f"  Year 0 MI risk: {mi_risks_year0[biggest_abs_increase_idx]:.6f}")
print(f"  Year 9 MI risk: {mi_risks_year9[biggest_abs_increase_idx]:.6f}")
print(f"  Absolute change: {absolute_changes[biggest_abs_increase_idx]:.6f}")
print(f"  Relative change: {relative_changes[biggest_abs_increase_idx]:.2f}x")

print(f"\n{'='*80}")
print("BIGGEST RELATIVE INCREASE (High Baseline Risk)")
print(f"{'='*80}")
if high_baseline_mask.sum() > 0:
    rel_idx_in_filtered = biggest_rel_increase_idx
    rel_idx_in_full = patient_indices.index(patient_rel)
    print(f"Patient #{patient_rel}:")
    print(f"  Year 0 MI risk: {mi_risks_year0[rel_idx_in_full]:.6f}")
    print(f"  Year 9 MI risk: {mi_risks_year9[rel_idx_in_full]:.6f}")
    print(f"  Absolute change: {absolute_changes[rel_idx_in_full]:.6f}")
    print(f"  Relative change: {relative_changes[rel_idx_in_full]:.2f}x")
else:
    rel_idx_in_full = biggest_rel_increase_idx
    print(f"Patient #{patient_rel}:")
    print(f"  Year 0 MI risk: {mi_risks_year0[rel_idx_in_full]:.6f}")
    print(f"  Year 9 MI risk: {mi_risks_year9[rel_idx_in_full]:.6f}")
    print(f"  Absolute change: {absolute_changes[rel_idx_in_full]:.6f}")
    print(f"  Relative change: {relative_changes[rel_idx_in_full]:.2f}x")

# Use the patient with biggest relative increase (usually more meaningful)
patient_idx = patient_rel

# Get this patient's MI risks over all years
patient_mi_risks = []
for k in range(min(10, len(pi_batches))):
    pi_k = pi_batches[k][patient_idx]
    risk_k = pi_k[mi_idx, 0].item()
    patient_mi_risks.append(risk_k)

# Calculate population average MI risk over time (for comparison)
print("\nCalculating population average MI risk over time...")
pop_avg_mi_risks = []
for k in range(min(10, len(pi_batches))):
    pi_k = pi_batches[k]  # Shape: (10000, 348, 52)
    # Get MI risk for all patients (excluding those with MI at enrollment)
    patient_mi_risks_k = []
    for p in range(len(Y_batch)):
        if Y_batch[p, mi_idx, 0].item() == 0:  # No MI at enrollment
            risk_k = pi_k[p, mi_idx, 0].item()
            if risk_k > 0:
                patient_mi_risks_k.append(risk_k)
    if len(patient_mi_risks_k) > 0:
        pop_avg_mi_risks.append(np.mean(patient_mi_risks_k))
    else:
        pop_avg_mi_risks.append(0.0)

print(f"  Population average MI risk: {pop_avg_mi_risks[0]:.6f} (year 0) → {pop_avg_mi_risks[-1]:.6f} (year {len(pop_avg_mi_risks)-1})")
print(f"  Patient MI risk: {patient_mi_risks[0]:.6f} (year 0) → {patient_mi_risks[-1]:.6f} (year {len(patient_mi_risks)-1})")
print(f"  Patient vs population: {patient_mi_risks[0]/pop_avg_mi_risks[0]:.2f}x → {patient_mi_risks[-1]/pop_avg_mi_risks[-1]:.2f}x" if pop_avg_mi_risks[0] > 0 and pop_avg_mi_risks[-1] > 0 else "  (Cannot calculate ratio)")

# Get patient's new diagnoses
new_diagnoses_by_year = {}
for year in range(1, min(10, Y_batch.shape[2])):
    new_diags = []
    for d_idx in range(len(disease_names)):
        if Y_batch[patient_idx, d_idx, year].item() > 0 and Y_batch[patient_idx, d_idx, year-1].item() == 0:
            new_diags.append(disease_names[d_idx])
    if new_diags:
        new_diagnoses_by_year[year] = new_diags[:3]  # Limit to 3 per year

# Visualize
fig, ax = plt.subplots(figsize=(14, 8))
years = np.arange(len(patient_mi_risks))
ax.plot(years, patient_mi_risks, 'o-', linewidth=3, markersize=10, color='steelblue', label='Patient MI Risk')

# Add population average line
if len(pop_avg_mi_risks) > 0:
    pop_years = np.arange(len(pop_avg_mi_risks))
    ax.plot(pop_years, pop_avg_mi_risks, '--', linewidth=2, color='gray', alpha=0.7, label='Population Average MI Risk')

# Highlight years with new diagnoses
for year, diags in new_diagnoses_by_year.items():
    if year < len(patient_mi_risks):
        ax.axvline(x=year, color='red', linestyle='--', alpha=0.6, linewidth=2)
        ax.scatter(year, patient_mi_risks[year], s=300, color='red', zorder=5, 
                   edgecolors='black', linewidth=2, marker='*')
        # Add annotation
        diag_text = diags[0][:30] + '...' if len(diags[0]) > 30 else diags[0]
        risk_range = max(patient_mi_risks) - min(patient_mi_risks)
        offset = max(risk_range * 0.15, 0.000001)
        ax.annotate(f'New: {diag_text}', 
                   xy=(year, patient_mi_risks[year]),
                   xytext=(year, patient_mi_risks[year] + offset),
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.set_xlabel('Years After Enrollment', fontsize=14, fontweight='bold')
ax.set_ylabel('1-Year MI Risk', fontsize=14, fontweight='bold')
ax.set_title(f'Patient #{patient_idx}: MI Risk Over Time\n' +
             f'Change: {patient_mi_risks[0]:.6f} → {patient_mi_risks[-1]:.6f} ({relative_changes[rel_idx_in_full]:.2f}x)', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xticks(years)

plt.tight_layout()
plt.show()

# Get baseline diagnoses (at enrollment, year 0)
baseline_diagnoses = []
for d_idx in range(len(disease_names)):
    if Y_batch[patient_idx, d_idx, 0].item() > 0:
        baseline_diagnoses.append(disease_names[d_idx])

# Try to get genetic risk factors (PRS scores) if available
genetic_info = {}
try:
    # First try to load from essentials (most common)
    if 'essentials' in locals() or 'essentials' in globals():
        if 'G' in essentials:
            G_full = essentials['G']
            if hasattr(G_full, 'numpy'):
                G_full = G_full.numpy()
            elif hasattr(G_full, 'detach'):
                G_full = G_full.detach().numpy()
            G_patient = G_full[patient_idx]
            genetic_info['G_available'] = True
            genetic_info['G_values'] = G_patient
            if 'prs_names' in essentials:
                genetic_info['prs_names'] = essentials['prs_names']
    # Fallback: check if G is directly available
    elif 'G' in locals() or 'G' in globals():
        G_patient = G[patient_idx]
        if hasattr(G_patient, 'numpy'):
            G_patient = G_patient.numpy()
        elif hasattr(G_patient, 'detach'):
            G_patient = G_patient.detach().numpy()
        genetic_info['G_available'] = True
        genetic_info['G_values'] = G_patient
    else:
        genetic_info['G_available'] = False
except Exception as e:
    genetic_info['G_available'] = False
    genetic_info['error'] = str(e)

# Get baseline demographics
patient_row = pce_df_batch.iloc[patient_idx]
baseline_info = {}
for col in pce_df_batch.columns:
    if col not in ['eid', 'Enrollment_Date', 'Birthdate']:  # Skip ID/date columns
        baseline_info[col] = patient_row[col]

# Print summary
print(f"\n{'='*80}")
print(f"PATIENT #{patient_idx} SUMMARY")
print(f"{'='*80}")
print(f"Enrollment age: {pce_df_batch.iloc[patient_idx]['age']:.0f} years")
if 'sex' in baseline_info:
    sex_str = "Male" if baseline_info['sex'] == 1 else "Female"
    print(f"Sex: {sex_str}")
print(f"MI risk: {patient_mi_risks[0]:.6f} → {patient_mi_risks[-1]:.6f}")
print(f"Relative increase: {relative_changes[rel_idx_in_full]:.2f}x")
if len(pop_avg_mi_risks) > 0:
    print(f"\nPopulation average MI risk: {pop_avg_mi_risks[0]:.6f} → {pop_avg_mi_risks[-1]:.6f}")
    if pop_avg_mi_risks[0] > 0 and pop_avg_mi_risks[-1] > 0:
        print(f"Patient vs population: {patient_mi_risks[0]/pop_avg_mi_risks[0]:.2f}x → {patient_mi_risks[-1]/pop_avg_mi_risks[-1]:.2f}x")

# Print baseline diagnoses
print(f"\n{'='*80}")
print("BASELINE DIAGNOSES (At Enrollment)")
print(f"{'='*80}")
if baseline_diagnoses:
    print(f"Patient had {len(baseline_diagnoses)} diagnosis(es) at enrollment:")
    # Group by cardiovascular-related, metabolic, etc.
    cv_related = [d for d in baseline_diagnoses if any(term in d.lower() for term in ['heart', 'cardiac', 'coronary', 'cardiovascular', 'hypertension', 'cholesterol', 'lipid'])]
    metabolic = [d for d in baseline_diagnoses if any(term in d.lower() for term in ['diabetes', 'metabolic', 'obesity', 'glucose'])]
    other = [d for d in baseline_diagnoses if d not in cv_related and d not in metabolic]
    
    if cv_related:
        print(f"\n  Cardiovascular-related ({len(cv_related)}):")
        for d in cv_related[:10]:  # Limit to 10
            print(f"    - {d}")
    if metabolic:
        print(f"\n  Metabolic ({len(metabolic)}):")
        for d in metabolic[:10]:
            print(f"    - {d}")
    if other:
        print(f"\n  Other ({len(other)}):")
        for d in other[:10]:
            print(f"    - {d}")
    if len(baseline_diagnoses) > 30:
        print(f"\n  ... and {len(baseline_diagnoses) - 30} more diagnoses")
else:
    print("  No diagnoses at enrollment")

# Print genetic risk factors if available
if genetic_info.get('G_available', False):
    print(f"\n{'='*80}")
    print("GENETIC RISK FACTORS (PRS Scores)")
    print(f"{'='*80}")
    G_vals = genetic_info['G_values']
    if hasattr(G_vals, 'numpy'):
        G_vals = G_vals.numpy()
    elif hasattr(G_vals, 'detach'):
        G_vals = G_vals.detach().numpy()
    G_vals = np.array(G_vals).flatten()
    
    prs_names = genetic_info.get('prs_names', None)
    if prs_names is None:
        # Try to load from file
        try:
            prs_names_df = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/prs_names.csv', header=None)
            prs_names = prs_names_df.iloc[:, 0].tolist()
        except:
            prs_names = [f"PRS_{i}" for i in range(len(G_vals))]
    
    # Show top PRS scores (highest absolute values)
    prs_dict = {}
    for i, val in enumerate(G_vals):
        if i < len(prs_names):
            prs_dict[prs_names[i]] = val
        else:
            prs_dict[f"Factor_{i}"] = val
    
    # Sort by absolute value
    sorted_prs = sorted(prs_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Top genetic risk factors (by absolute value):")
    for name, val in sorted_prs[:15]:  # Top 15
        # Highlight cardiovascular-related PRS
        highlight = " ⭐" if any(term in name.lower() for term in ['cad', 'cvd', 'coronary', 'cardiac', 'heart', 'ascvd', 'mi', 'stroke']) else ""
        print(f"  {name}: {val:.4f}{highlight}")
else:
    print(f"\n{'='*80}")
    print("GENETIC RISK FACTORS")
    print(f"{'='*80}")
    print("  Genetic data (G matrix) not available in current context")

# Print other baseline characteristics
if baseline_info:
    print(f"\n{'='*80}")
    print("OTHER BASELINE CHARACTERISTICS")
    print(f"{'='*80}")
    for key, val in baseline_info.items():
        if key not in ['age', 'sex']:  # Already shown
            if isinstance(val, (int, float)):
                if abs(val) < 1000:  # Skip very large numbers (likely IDs)
                    print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
            elif isinstance(val, str):
                print(f"  {key}: {val}")

print(f"\nNew diagnoses:")
if new_diagnoses_by_year:
    for year, diags in sorted(new_diagnoses_by_year.items()):
        print(f"  Year {year}: {', '.join(diags[:2])}")
else:
    print("  None")
    print(f"\n{'='*80}")
    print("WHY DID RISK INCREASE WITHOUT NEW DIAGNOSES?")
    print(f"{'='*80}")
    print("Even without new diagnoses, MI risk can increase because:")
    print(f"1. **Age effect**: Patient is aging (age {pce_df_batch.iloc[patient_idx]['age']:.0f} → {pce_df_batch.iloc[patient_idx]['age'] + 9:.0f} years)")
    print("   - Age is a major risk factor for MI")
    print("   - Each year's prediction uses age-offset models that account for age")
    print("2. **Model learning/calibration**: Each year's prediction uses a model trained with")
    print("   data up to that point, so the model's understanding evolves")
    print("3. **Population trends**: The population average also changes over time")
    if len(pop_avg_mi_risks) > 0 and pop_avg_mi_risks[0] > 0:
        pop_change = pop_avg_mi_risks[-1] / pop_avg_mi_risks[0]
        print(f"   - Population average increased {pop_change:.2f}x")
    print("4. **Baseline risk factors**: Patient may have had risk factors at enrollment")
    print("   that the model learns to weight differently as it sees more outcomes")

