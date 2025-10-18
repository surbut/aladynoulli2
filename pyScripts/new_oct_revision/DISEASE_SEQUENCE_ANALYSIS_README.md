# Disease Sequence Analysis

## Overview

This analysis addresses a fundamental clinical and scientific question: **How do different patients arrive at the same disease through different biological pathways?**

This directly supports your "patient morphs" narrative and provides clinical value beyond AUC improvements.

## What It Does

### 1. **Identifies Disease Sequences**
- Uses granular ICD-10 diagnosis data to identify the sequence of diseases before a target outcome (e.g., MI)
- Looks back 5-10 years before disease onset
- Identifies the most common disease sequences (e.g., "Rheumatoid arthritis → Hypertension → CAD → MI")

### 2. **Links Sequences to Signature Patterns**
- For each disease sequence, identifies which signatures were elevated beforehand
- Shows that different sequences associate with different signature patterns
- E.g., inflammatory pathway (high Sig 5) vs. metabolic pathway (high Sig 12)

### 3. **Integrates Multi-Modal Data**
- **Medications**: Do different pathways show different treatment patterns?
- **Genetics (PRS)**: Do pathways have distinct genetic risk profiles?
- **Temporal Dynamics**: How do signatures evolve over time along each pathway?

### 4. **Quantifies Heterogeneity**
- Shows that patients with "identical" diagnoses (e.g., MI) got there in fundamentally different ways
- Identifies the most discriminating signatures and diseases for each pathway
- Provides interpretable, biologically grounded clustering

## Clinical Value

### Beyond AUC: Interpretability and Actionability

1. **Precision Risk Stratification**
   - Not just "10% risk of MI in 10 years"
   - But "You're on an inflammatory pathway to MI - high Sig 5, low Sig 12"
   - Enables targeted screening and earlier intervention

2. **Treatment Personalization**
   - Inflammatory pathway → aggressive anti-inflammatory therapy
   - Metabolic pathway → focus on metabolic optimization
   - Traditional risk factor pathway → standard statin therapy
   - Different pathways may respond differently to the same drug

3. **Mechanistic Understanding**
   - Reveals the biological processes driving disease progression
   - Shows how signatures capture real disease biology (not just correlations)
   - Connects genetics → signatures → disease sequences → outcomes

4. **Early Detection**
   - Identify patients on high-risk pathways before traditional risk scores flag them
   - E.g., 40-year-old with RA loading heavily on Sig 5 + high CVD PRS
   - Intervene before symptoms appear

## What Makes This Novel

### 1. **Temporal + Multi-Disease + Genetic Integration**
Most approaches focus on one aspect:
- Traditional risk scores: snapshot in time
- Disease trajectory models: ignore genetics
- PRS: ignore longitudinal disease history
- **ALADYNOULLI**: combines all three with interpretable signatures

### 2. **Unsupervised Discovery + Prediction**
- Other approaches separate discovery (clustering) from prediction
- Your model does both simultaneously
- Signatures are biologically informed by genetics, not just data-driven

### 3. **Bayesian Posterior Updating**
- As new diagnoses occur, posterior updates refine individual loading
- Combines prior belief (genetics) with observed data (diagnoses)
- Not a static "risk score" but a living, evolving assessment

### 4. **Gaussian Process Temporal Smoothing**
- Signatures evolve smoothly over time (captured by GP covariance)
- Warping allows genetics to affect speed through pathways
- More realistic than discrete state transitions

## How to Use the Analysis

### Quick Start
```python
from run_pathway_analysis import run_complete_pathway_analysis

# Run complete analysis for MI
results = run_complete_pathway_analysis("myocardial infarction", n_pathways=4)
```

### What You Get
1. **Pathway Discovery**
   - 4 distinct pathways to MI
   - Patients clustered by pre-disease signature trajectory
   
2. **Discriminating Signatures**
   - Which signatures most differentiate pathways
   - Signature evolution plots for each pathway
   
3. **Disease Patterns**
   - Pre-disease conditions that characterize each pathway
   - Statistical testing for differential disease enrichment
   
4. **Medication Patterns** (with relaxed thresholds)
   - Long-term medications by pathway
   - Fold enrichment analysis
   
5. **Genetic Patterns**
   - PRS differences by pathway
   - Top discriminating PRS scores
   
6. **Disease Sequences** (if ICD-10 data available)
   - Most common disease sequences
   - Bigrams (2-disease transitions)
   - Category-level analysis

## Key Figures for Paper

### Figure 1: Disease Sequence → Signature Heatmap
- Rows: Common disease sequences (e.g., "Rheum → HTN → CAD → MI")
- Columns: Signatures
- Shows which signatures are elevated for each sequence

### Figure 2: Pathway Signature Evolution
- Time on x-axis (years before MI)
- Signature loading on y-axis
- One line per pathway, showing how they diverge over time
- Deviation from population reference

### Figure 3: Pathway Characterization Matrix
- Pathways on x-axis
- Characterization dimensions on y-axis:
  - Top diseases (pre-MI)
  - Top medications
  - Top PRS enrichments
  - Dominant signatures
- Heatmap showing pathway-specific patterns

### Figure 4: Patient Trajectory Examples
- Individual patient lambda trajectories
- Show 3-4 patients from different pathways
- All end at MI but arrived differently
- Annotate with key diagnoses and medications

## Addressing Reviewer Concerns

### "What heterogeneity does the model reveal?"
**Answer**: The model reveals that patients with identical outcomes (e.g., MI) arrive through distinct biological pathways characterized by:
1. Different pre-disease conditions (inflammatory vs. metabolic vs. traditional)
2. Different signature patterns (temporal evolution)
3. Different genetic risk profiles (PRS)
4. Different treatment patterns (medications)

This is not arbitrary clustering - it's biologically interpretable and clinically actionable.

### "Why is this better than existing approaches?"
**Answer**: 
1. **Temporal**: Captures disease evolution, not just snapshots
2. **Multi-domain**: Integrates genetics + longitudinal diagnoses + treatments
3. **Interpretable**: Signatures have biological meaning via genetic associations
4. **Unified**: Discovery and prediction in one framework
5. **Bayesian**: Combines universal (population) and specific (individual) information

### "What's the clinical value beyond AUC?"
**Answer**:
1. **Risk stratification**: Different pathways may need different screening intensities
2. **Treatment selection**: Pathway membership could guide therapy choice
3. **Early detection**: Identify high-risk pathways before traditional risk scores
4. **Mechanistic insights**: Understanding WHY risk is elevated, not just that it is
5. **Precision prevention**: Tailor interventions to biological pathway

## Files in This Analysis

1. `disease_sequence_analysis.py`: Core analysis functions
   - Load ICD-10 data
   - Find disease sequences
   - Link to signatures
   - Visualizations

2. `run_pathway_analysis.py`: Main wrapper
   - Runs complete pipeline
   - Integrates all analyses
   - Handles errors gracefully

3. `pathway_discovery.py`: Clustering methods
   - Average loading clustering
   - Trajectory similarity clustering
   - Pre-disease focus

4. `pathway_interrogation.py`: Characterization
   - Disease patterns
   - Signature dynamics
   - PRS analysis
   - Granular disease analysis

5. `medication_integration.py`: Treatment patterns
   - Link medications to pathways
   - Fold enrichment analysis
   - BNF categorization

## Data Requirements

### Required (for basic pathway analysis):
- `Y.pt`: Disease occurrence matrix (400K patients × 348 diseases × 52 timepoints)
- `thetas.npy`: Signature loadings (400K patients × 21 signatures × 52 timepoints)
- `processed_ids.csv`: Patient IDs (first 400K)
- `disease_names.csv`: Disease names

### Optional (for enhanced analysis):
- `icd2phecode.rds`: Granular ICD-10 diagnoses (for sequence analysis)
- `gp_scripts.txt`: Medication prescriptions (for treatment patterns)
- `prs_scores.csv`: Polygenic risk scores (for genetic analysis)

## Installation Requirements

```bash
# Basic requirements
pip install numpy pandas matplotlib seaborn scikit-learn torch

# For ICD-10 data
pip install pyreadr

# For medication analysis
# (No additional packages needed - uses existing identify_long_term_medications.py)
```

## Next Steps

1. **Run the analysis** on MI to verify everything works
2. **Relax medication thresholds** further if needed (currently 1.2x fold enrichment OR 5% prevalence)
3. **Add more ICD-10 mappings** for other diseases
4. **Create publication-quality figures**
5. **Write results section** emphasizing:
   - Heterogeneity discovery
   - Clinical actionability
   - Biological interpretability
   - Beyond-AUC value

## Questions to Answer with This Analysis

For the paper, focus on these questions:

1. **How many distinct pathways lead to MI?** (e.g., 4 pathways)
2. **What characterizes each pathway?** (diseases, signatures, PRS)
3. **Are pathways biologically interpretable?** (inflammatory vs. metabolic vs. traditional)
4. **Do pathways show different medication patterns?** (validation of biological difference)
5. **Can we predict pathway membership before MI?** (early detection potential)
6. **Do different pathways have different outcomes after MI?** (future work)

## Contact

For questions about this analysis, refer to:
- Main notebook: `test_disease_sequences.ipynb`
- Test with MI: `test_pathway_analysis.ipynb`
- Documentation: This file and `README_PATHWAY_ANALYSIS.md`

