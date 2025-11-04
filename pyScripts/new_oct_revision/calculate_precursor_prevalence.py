"""
Calculate Average Precursor Prevalence by Pathway for MI
"""

# Data from the pathway analysis log
pathway_prevalences = {
    0: {
        'Coronary atherosclerosis': 86.3,
        'Hypercholesterolemia': 75.9,
        'Angina pectoris': 75.0,
        'Essential hypertension': 74.8,
        'Type 2 diabetes': 26.2,
        'Obesity': 11.8
    },
    1: {
        'Essential hypertension': 20.8,
        'Hypercholesterolemia': 9.9,
        'Coronary atherosclerosis': 8.3,
        'Angina pectoris': 7.4,
        'Type 2 diabetes': 6.8,
        'Obesity': 2.2
    },
    2: {
        'Essential hypertension': 65.2,
        'Hypercholesterolemia': 29.7,
        'Angina pectoris': 22.8,
        'Coronary atherosclerosis': 20.3,
        'Type 2 diabetes': 18.6,
        'Obesity': 15.8
    },
    3: {
        'Essential hypertension': 27.8,
        'Hypercholesterolemia': 16.8,
        'Coronary atherosclerosis': 15.9,
        'Type 2 diabetes': 13.3,
        'Angina pectoris': 12.3,
        'Obesity': 4.1
    }
}

print("="*80)
print("AVERAGE PRECURSOR PREVALENCE (π) BY PATHWAY")
print("="*80)

for pathway_id, prevalences in pathway_prevalences.items():
    avg_prevalence = sum(prevalences.values()) / len(prevalences)
    
    print(f"\nPathway {pathway_id}:")
    for disease, pct in sorted(prevalences.items(), key=lambda x: x[1], reverse=True):
        print(f"  {disease:40s}: {pct:5.1f}%")
    print(f"  {'AVERAGE (π)':40s}: {avg_prevalence:5.1f}%")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

for pathway_id, prevalences in pathway_prevalences.items():
    avg = sum(prevalences.values()) / len(prevalences)
    pathway_names = ['Progressive Ischemia', 'Hidden Risk', 'Multimorbid', 'Metabolic']
    print(f"Pathway {pathway_id} ({pathway_names[pathway_id]}): π = {avg:.1f}%")

# Also calculate in terms of Signature 5 deviations from the log
print("\n" + "="*80)
print("SIGNATURE 5 DEVIATIONS (for comparison):")
print("="*80)
# From the log file
sig5_deviations = {
    0: 0.179,
    1: 0.025,
    2: 0.048,
    3: 0.229
}

for pathway_id in range(4):
    avg_prev = sum(pathway_prevalences[pathway_id].values()) / len(pathway_prevalences[pathway_id])
    sig5_dev = sig5_deviations[pathway_id]
    print(f"Pathway {pathway_id}: π = {avg_prev:.1f}%, Sig5 deviation = {sig5_dev:+.3f}")






