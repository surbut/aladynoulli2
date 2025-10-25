# Actionable Next Steps for MI Pathway Analysis

## Immediate Actions (This Week)

### 1. Create Pathway Signature Heatmap
**Goal**: Visualize the "signature fingerprint" of each pathway

**Code to add to your analysis:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_pathway_signature_heatmap(pathway_data):
    """
    Create heatmap showing mean signature deviations for each pathway
    """
    # Extract pathway information
    patients = pathway_data['patients']
    K = pathway_data['trajectory_features'].shape[1] // 5  # Number of signatures
    
    # Calculate mean deviation per signature per pathway
    n_pathways = len(np.unique(pathway_data['pathway_labels']))
    mean_deviations = np.zeros((n_pathways, K))
    
    for pathway_id in range(n_pathways):
        # Get patients in this pathway
        pathway_mask = pathway_data['pathway_labels'] == pathway_id
        pathway_features = pathway_data['trajectory_features'][pathway_mask]
        
        # Average across patients and timepoints
        # Features are structured as: [sig0_t0, sig0_t1, ..., sig1_t0, sig1_t1, ...]
        for sig_idx in range(K):
            # Get all timepoints for this signature
            sig_features = pathway_features[:, sig_idx::K]  # Every K-th column
            mean_deviations[pathway_id, sig_idx] = np.mean(sig_features)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Signature names (from your project knowledge)
    sig_names = [
        '0: Cardiac Arrhy', '1: Musculoskel', '2: Upper GI', '3: Mixed Med',
        '4: Upper Resp', '5: Ischemic CV', '6: Metastatic', '7: Pain/Inflam',
        '8: Gynecologic', '9: Spinal', '10: Ophthalmic', '11: Cerebrovasc',
        '12: Renal/Uro', '13: Male Urogen', '14: Pulm/Smoke', '15: Diabetes',
        '16: Infect/Crit', '17: Lower GI', '18: Hepatobil', '19: Dermatologic',
        '20: Undefined'
    ]
    
    sns.heatmap(mean_deviations, 
                xticklabels=sig_names,
                yticklabels=[f'Pathway {i}' for i in range(n_pathways)],
                cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Mean Deviation from Population'},
                ax=ax)
    
    plt.title('Signature Fingerprint by MI Pathway', fontsize=16, fontweight='bold')
    plt.xlabel('Disease Signature', fontsize=12)
    plt.ylabel('Pathway', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('pathway_signature_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_deviations
```

---

### 2. Identify Key Discriminating Signatures
**Goal**: Statistically test which signatures most differentiate pathways

**Code to add:**
```python
from scipy.stats import f_oneway

def identify_discriminating_signatures(pathway_data):
    """
    Use ANOVA to identify signatures that most differentiate pathways
    """
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    trajectory_features = pathway_data['trajectory_features']
    K = trajectory_features.shape[1] // 5
    
    f_statistics = []
    p_values = []
    
    for sig_idx in range(K):
        # Get signature features across all timepoints
        sig_features = trajectory_features[:, sig_idx::K]
        
        # Average across timepoints for each patient
        sig_mean_per_patient = np.mean(sig_features, axis=1)
        
        # Group by pathway
        pathway_groups = []
        for pathway_id in range(4):
            pathway_mask = pathway_labels == pathway_id
            pathway_groups.append(sig_mean_per_patient[pathway_mask])
        
        # ANOVA
        f_stat, p_val = f_oneway(*pathway_groups)
        f_statistics.append(f_stat)
        p_values.append(p_val)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Signature': range(K),
        'F_statistic': f_statistics,
        'P_value': p_values,
        'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' 
                        for p in p_values]
    })
    
    results = results.sort_values('F_statistic', ascending=False)
    
    print("\nTop 10 Discriminating Signatures:")
    print(results.head(10).to_string(index=False))
    
    return results
```

---

### 3. Temporal Trajectory Visualization
**Goal**: Show when signatures diverge from population for each pathway

**Code to add:**
```python
def plot_temporal_trajectories(pathway_data, thetas, population_reference, top_sigs=5):
    """
    Plot temporal trajectories of top signatures for each pathway
    """
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    n_pathways = len(np.unique(pathway_labels))
    
    fig, axes = plt.subplots(n_pathways, 1, figsize=(14, 4*n_pathways))
    if n_pathways == 1:
        axes = [axes]
    
    # Get top discriminating signatures
    discriminating_results = identify_discriminating_signatures(pathway_data)
    top_sig_indices = discriminating_results['Signature'].values[:top_sigs]
    
    for pathway_id in range(n_pathways):
        ax = axes[pathway_id]
        
        # Get patients in this pathway
        pathway_mask = pathway_labels == pathway_id
        pathway_patients = [p for p, mask in zip(patients, pathway_mask) if mask]
        
        print(f"\nProcessing Pathway {pathway_id}: {len(pathway_patients)} patients")
        
        # Calculate mean trajectory for this pathway
        # Align all patients to 5 years before MI
        window_years = 5
        K, T = population_reference.shape
        
        pathway_trajectories = []
        
        for patient_info in pathway_patients:
            patient_id = patient_info['patient_id']
            age_at_disease = patient_info['age_at_disease']
            disease_time_idx = age_at_disease - 30
            
            # Get 5-year window
            start_idx = max(0, disease_time_idx - window_years)
            end_idx = disease_time_idx
            
            if end_idx > start_idx:
                patient_traj = thetas[patient_id, :, start_idx:end_idx]
                pathway_trajectories.append(patient_traj)
        
        # Align to common length
        if len(pathway_trajectories) > 0:
            min_length = min(traj.shape[1] for traj in pathway_trajectories)
            aligned = np.array([traj[:, -min_length:] for traj in pathway_trajectories 
                              if traj.shape[1] >= min_length])
            
            # Calculate mean
            mean_traj = np.mean(aligned, axis=0)
            
            # Get corresponding population reference
            ref_start = T - min_length
            ref_traj = population_reference[:, ref_start:]
            
            # Calculate deviations
            deviations = mean_traj - ref_traj
            
            # Plot top signatures
            time_points = np.arange(-min_length, 0)
            
            for sig_idx in top_sig_indices:
                ax.plot(time_points, deviations[sig_idx, :], 
                       linewidth=2, marker='o', markersize=4,
                       label=f'Sig {sig_idx}')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(f'Pathway {pathway_id} (n={len(pathway_patients)} patients)',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Years Before MI')
            ax.set_ylabel('Signature Deviation')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_trajectories_by_pathway.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Short-term Actions (Next 2 Weeks)

### 4. Age-Stratified Analysis
**Goal**: Test if pathways differ by age at MI onset

```python
def age_stratified_analysis(pathway_data):
    """
    Compare pathway distributions across age groups
    """
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    
    # Calculate ages
    ages = [p['age_at_disease'] for p in patients]
    
    # Define age groups
    age_bins = [0, 50, 60, 70, 100]
    age_labels = ['<50', '50-60', '60-70', '70+']
    age_groups = pd.cut(ages, bins=age_bins, labels=age_labels)
    
    # Create crosstab
    results = pd.crosstab(age_groups, pathway_labels, normalize='index') * 100
    
    print("\nPathway Distribution by Age at MI:")
    print(results.round(1))
    
    # Statistical test
    chi2, p_value = stats.chi2_contingency(pd.crosstab(age_groups, pathway_labels))[:2]
    print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.2e}")
    
    # Visualization
    results.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Pathway Distribution by Age at MI')
    plt.xlabel('Age at MI')
    plt.ylabel('Percentage')
    plt.legend(title='Pathway', labels=[f'Pathway {i}' for i in range(4)])
    plt.tight_layout()
    plt.savefig('pathway_by_age.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results
```

---

### 5. Clinical Characteristics Table
**Goal**: Create Table 1 for paper submission

```python
def create_clinical_characteristics_table(pathway_data, Y, disease_names):
    """
    Create comprehensive characteristics table by pathway
    """
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    
    # Key diseases to report
    key_diseases = [
        'Essential hypertension',
        'Hypercholesterolemia', 
        'Type 2 diabetes',
        'Coronary atherosclerosis',
        'Angina pectoris',
        'Atrial fibrillation',
        'Acute renal failure',
        'Heart failure',
        'Obesity',
        'Major depressive disorder'
    ]
    
    results = []
    
    for pathway_id in range(4):
        pathway_mask = pathway_labels == pathway_id
        pathway_patients_info = [p for p, mask in zip(patients, pathway_mask) if mask]
        
        # Age statistics
        ages = [p['age_at_disease'] for p in pathway_patients_info]
        
        row = {
            'Pathway': pathway_id,
            'N': len(pathway_patients_info),
            'Age (mean ± SD)': f"{np.mean(ages):.1f} ± {np.std(ages):.1f}",
            'Age (median [IQR])': f"{np.median(ages):.1f} [{np.percentile(ages, 25):.1f}-{np.percentile(ages, 75):.1f}]"
        }
        
        # Disease prevalences
        for disease_name in key_diseases:
            disease_idx = None
            for i, name in enumerate(disease_names):
                if disease_name.lower() in name.lower():
                    disease_idx = i
                    break
            
            if disease_idx is not None:
                # Count patients with this disease BEFORE MI
                count = 0
                for patient_info in pathway_patients_info:
                    patient_id = patient_info['patient_id']
                    age_at_disease = patient_info['age_at_disease']
                    disease_time_idx = age_at_disease - 30
                    
                    if disease_time_idx > 0 and Y[patient_id, disease_idx, :disease_time_idx].sum() > 0:
                        count += 1
                
                pct = count / len(pathway_patients_info) * 100
                row[disease_name] = f"{count} ({pct:.1f}%)"
        
        results.append(row)
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("CLINICAL CHARACTERISTICS BY PATHWAY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('pathway_clinical_characteristics.csv', index=False)
    print("\nSaved to pathway_clinical_characteristics.csv")
    
    return df
```

---

### 6. Pathway Stability Analysis
**Goal**: Test if pathways are consistent across different k values

```python
def test_pathway_stability(Y, thetas, disease_names, target_disease="myocardial infarction",
                          k_values=[3, 4, 5, 6]):
    """
    Test pathway discovery across different numbers of clusters
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    results = []
    
    for k in k_values:
        print(f"\nTesting k={k} clusters...")
        
        pathway_data = discover_disease_pathways(
            target_disease, Y, thetas, disease_names, 
            n_pathways=k, method='deviation_from_reference'
        )
        
        if pathway_data is not None:
            features = pathway_data['features_scaled']
            labels = pathway_data['pathway_labels']
            
            # Calculate clustering quality metrics
            silhouette = silhouette_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            
            results.append({
                'k': k,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'n_patients': len(labels)
            })
            
            print(f"  Silhouette Score: {silhouette:.3f}")
            print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Plot results
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(df['k'], df['silhouette_score'], marker='o', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score (higher is better)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['k'], df['davies_bouldin_score'], marker='o', linewidth=2, color='orange')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Davies-Bouldin Score')
    axes[1].set_title('Davies-Bouldin Score (lower is better)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pathway_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df
```

---

## Medium-term Actions (Next Month)

### 7. Link to Medication Data
**Goal**: Understand medication patterns by pathway

**Questions to answer:**
1. What medications are patients on in each pathway?
2. Do pathways differ in medication exposure?
3. Does medication use modify pathway membership?

**Required data:**
- Prescription records from UK Biobank
- Link via patient IDs (`processed_ids`)

**Analysis approach:**
```python
def analyze_medications_by_pathway(pathway_data, prescription_data, processed_ids):
    """
    Analyze medication patterns by pathway
    
    Parameters:
    -----------
    prescription_data : DataFrame with columns ['eid', 'drug_name', 'issue_date']
    """
    
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    
    # Key medication classes
    medication_classes = {
        'Statin': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
        'ACE Inhibitor': ['ramipril', 'lisinopril', 'enalapril'],
        'Beta Blocker': ['metoprolol', 'bisoprolol', 'atenolol'],
        'Aspirin': ['aspirin', 'acetylsalicylic'],
        'Metformin': ['metformin']
    }
    
    results = []
    
    for pathway_id in range(4):
        pathway_mask = pathway_labels == pathway_id
        pathway_patients_info = [p for p, mask in zip(patients, pathway_mask) if mask]
        
        # Get eids for these patients
        pathway_eids = [processed_ids[p['patient_id']] for p in pathway_patients_info]
        
        # Filter prescriptions
        pathway_prescriptions = prescription_data[prescription_data['eid'].isin(pathway_eids)]
        
        row = {'Pathway': pathway_id, 'N': len(pathway_patients_info)}
        
        for med_class, drug_names in medication_classes.items():
            # Count patients with any prescription in this class
            patients_with_med = set()
            for drug in drug_names:
                matches = pathway_prescriptions[
                    pathway_prescriptions['drug_name'].str.contains(drug, case=False, na=False)
                ]['eid'].unique()
                patients_with_med.update(matches)
            
            count = len(patients_with_med)
            pct = count / len(pathway_patients_info) * 100
            row[med_class] = f"{count} ({pct:.1f}%)"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("MEDICATION USE BY PATHWAY (PRE-MI)")
    print("="*80)
    print(df.to_string(index=False))
    
    return df
```

---

### 8. Outcome Analysis
**Goal**: Test if pathways predict post-MI outcomes

**Outcomes to evaluate:**
- 5-year mortality
- Recurrent MI
- Heart failure hospitalization
- Stroke

**Analysis code:**
```python
def analyze_outcomes_by_pathway(pathway_data, Y, disease_names, processed_ids, 
                               followup_years=5):
    """
    Compare outcomes by pathway
    """
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    
    # Outcome diseases
    outcome_diseases = {
        'Recurrent MI': 'myocardial infarction',
        'Heart Failure': 'heart failure',
        'Stroke': 'cerebral infarction'
    }
    
    results = []
    
    for pathway_id in range(4):
        pathway_mask = pathway_labels == pathway_id
        pathway_patients_info = [p for p, mask in zip(patients, pathway_mask) if mask]
        
        row = {'Pathway': pathway_id, 'N': len(pathway_patients_info)}
        
        for outcome_name, disease_keyword in outcome_diseases.items():
            # Find disease index
            disease_idx = None
            for i, name in enumerate(disease_names):
                if disease_keyword.lower() in name.lower():
                    disease_idx = i
                    break
            
            if disease_idx is not None:
                # Count patients with outcome in followup period
                count = 0
                for patient_info in pathway_patients_info:
                    patient_id = patient_info['patient_id']
                    age_at_mi = patient_info['age_at_disease']
                    mi_time_idx = age_at_mi - 30
                    
                    # Look for outcome in followup window
                    followup_start = mi_time_idx
                    followup_end = min(mi_time_idx + followup_years, Y.shape[2])
                    
                    if followup_end > followup_start:
                        if Y[patient_id, disease_idx, followup_start:followup_end].sum() > 0:
                            count += 1
                
                pct = count / len(pathway_patients_info) * 100
                row[outcome_name] = f"{count} ({pct:.1f}%)"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"OUTCOMES BY PATHWAY ({followup_years}-YEAR FOLLOW-UP)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Statistical test
    print("\nStatistical comparisons:")
    from scipy.stats import chi2_contingency
    
    for outcome_name in outcome_diseases.keys():
        # Create contingency table
        outcome_counts = []
        for pathway_id in range(4):
            pathway_mask = pathway_labels == pathway_id
            pathway_patients_info = [p for p, mask in zip(patients, pathway_mask) if mask]
            
            # Extract count from string "X (Y%)"
            outcome_str = df[df['Pathway'] == pathway_id][outcome_name].values[0]
            count = int(outcome_str.split(' (')[0])
            total = len(pathway_patients_info)
            
            outcome_counts.append([count, total - count])
        
        chi2, p_value = chi2_contingency(outcome_counts)[:2]
        print(f"{outcome_name}: χ² = {chi2:.2f}, p = {p_value:.3e}")
    
    return df
```

---

## Long-term Actions (Next 3 Months)

### 9. Build Pathway Prediction Model
**Goal**: Predict pathway membership from baseline data

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def build_pathway_predictor(pathway_data, Y, disease_names, processed_ids):
    """
    Build model to predict pathway membership from baseline characteristics
    """
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    
    # Create feature matrix from baseline (e.g., age 40) characteristics
    baseline_age = 40
    baseline_time_idx = baseline_age - 30
    
    X = []
    y = []
    
    for i, patient_info in enumerate(patients):
        patient_id = patient_info['patient_id']
        age_at_disease = patient_info['age_at_disease']
        
        # Only include if disease occurred after baseline
        if age_at_disease > baseline_age:
            # Extract baseline features
            # 1. Disease history up to baseline
            disease_counts = Y[patient_id, :, :baseline_time_idx].sum(axis=1)
            
            # 2. Add age at baseline
            features = list(disease_counts) + [baseline_age]
            
            X.append(features)
            y.append(pathway_labels[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training set: {len(X)} patients")
    print(f"Pathway distribution: {np.bincount(y)}")
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\nCross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Fit final model
    clf.fit(X, y)
    
    # Feature importance
    importances = clf.feature_importances_
    feature_names = disease_names + ['Age']
    
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    print("\nTop 20 Predictive Features:")
    print(top_features.to_string(index=False))
    
    return clf, top_features
```

---

### 10. Write Methods Section
**Goal**: Document your approach for paper submission

**Key elements to include:**

```markdown
## Pathway Discovery Methods

### Study Population
We analyzed 24,903 UK Biobank participants who developed myocardial infarction (MI) 
between ages 30-81 with at least 5 years of pre-MI electronic health record data.

### Signature-Based Trajectory Modeling
We used the ALADYNOULLI model to decompose each patient's disease history into 
21 disease signatures representing distinct biological processes (cardiac, 
metabolic, inflammatory, etc.). For each patient, we obtained time-varying 
signature loadings θ(t) representing the contribution of each signature at each 
age.

### Pathway Discovery Algorithm
1. **Feature extraction**: For each MI patient, we extracted signature trajectories 
   from the 5-year window preceding MI diagnosis.

2. **Population reference**: We calculated population-level average signature 
   trajectories across all 400,000 UK Biobank participants to establish age-specific 
   reference values.

3. **Deviation calculation**: For each patient, we computed deviations from the 
   population reference at each timepoint: Δθ(t) = θ_patient(t) - θ_population(t)

4. **Feature matrix**: We created a feature matrix with dimensions [n_patients × 
   (K signatures × 5 timepoints)], yielding 105 features per patient.

5. **Clustering**: We applied K-means clustering (k=4) to identify distinct 
   pathway groups, selecting k=4 based on silhouette score optimization.

### Statistical Analysis
We compared pathway groups using:
- ANOVA for continuous variables
- Chi-square tests for categorical variables
- Bonferroni correction for multiple testing (α = 0.05/21 signatures)

### Validation
We validated pathways through:
1. Clinical coherence (disease prevalence patterns)
2. Genetic architecture (polygenic risk score associations)
3. Reproducibility across clustering methods
```

---

## Summary: Priority Order

### This Week:
1. ✅ Create pathway signature heatmap
2. ✅ Identify discriminating signatures (ANOVA)
3. ✅ Plot temporal trajectories

### Next Week:
4. ⏳ Age-stratified analysis
5. ⏳ Clinical characteristics table
6. ⏳ Pathway stability analysis

### This Month:
7. 🔄 Link to medication data
8. 🔄 Outcome analysis
9. 🔄 Genetic validation

### Next 3 Months:
10. 📝 Build pathway prediction model
11. 📝 Write methods section
12. 📝 Prepare manuscript

---

## Key Deliverables for Paper

### Main Text Figures:
1. **Figure 1**: Study design & pathway overview
2. **Figure 2**: Pathway signature heatmap + temporal trajectories
3. **Figure 3**: Clinical characteristics & disease patterns
4. **Figure 4**: Outcomes & predictions

### Supplementary:
- S1: Clustering stability analysis
- S2: Age-stratified results
- S3: Medication patterns
- S4: Individual patient examples
- S5: Genetic validation

### Tables:
- Table 1: Clinical characteristics by pathway
- Table 2: Disease prevalence by pathway
- Table 3: Outcomes by pathway

---

## Code Integration

All the code above can be integrated into your existing scripts. I recommend creating:

```python
# pathway_analysis_complete.py

# Import all your existing functions
from pathway_discovery import load_full_data, discover_disease_pathways
from compare_specific_pathways import compare_specific_pathways

# Add all the new analysis functions
def run_complete_pathway_analysis():
    """
    Run complete pathway analysis pipeline
    """
    # 1. Load data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # 2. Discover pathways
    pathway_data = discover_disease_pathways(
        "myocardial infarction", Y, thetas, disease_names, 
        n_pathways=4, method='deviation_from_reference'
    )
    
    # 3. Create visualizations
    mean_devs = create_pathway_signature_heatmap(pathway_data)
    
    # 4. Statistical analysis
    discriminating_sigs = identify_discriminating_signatures(pathway_data)
    
    # 5. Temporal trajectories
    plot_temporal_trajectories(pathway_data, thetas, np.mean(thetas, axis=0))
    
    # 6. Clinical characteristics
    clinical_table = create_clinical_characteristics_table(
        pathway_data, Y, disease_names
    )
    
    # 7. Age stratification
    age_results = age_stratified_analysis(pathway_data)
    
    # 8. Stability analysis
    stability_results = test_pathway_stability(Y, thetas, disease_names)
    
    print("\n✅ Complete pathway analysis finished!")
    print("Check output folder for results")
    
    return {
        'pathway_data': pathway_data,
        'mean_deviations': mean_devs,
        'discriminating_signatures': discriminating_sigs,
        'clinical_characteristics': clinical_table,
        'age_stratified': age_results,
        'stability': stability_results
    }

if __name__ == "__main__":
    results = run_complete_pathway_analysis()
```

---

## Questions to Address in Paper

### Main Questions:
1. **How many pathways?** → 4 distinct pathways
2. **What defines them?** → Different signature patterns
3. **Are they clinically meaningful?** → Yes, different disease profiles
4. **Do they matter?** → Yes, different outcomes (need to show)

### Reviewer Questions (anticipate):
1. "Why 4 pathways?" → Silhouette score optimization
2. "Are these just severity stages?" → No, qualitatively different
3. "Can you validate?" → Yes, genetic + clinical coherence
4. "Can you predict?" → Yes, built prediction model
5. "Does it change treatment?" → Yes, pathway-specific interventions

---

Let me know which analysis you'd like to run first, and I can help you implement it!
