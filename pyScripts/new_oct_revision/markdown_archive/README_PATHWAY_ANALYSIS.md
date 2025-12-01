# Pathway Discovery and Analysis Pipeline

This directory contains scripts for discovering and analyzing different pathways patients take to reach the same disease outcome.

## Overview

The analysis consists of three main components:

1. **Pathway Discovery** (`pathway_discovery.py`): Discovers patient clusters/pathways
2. **Pathway Interrogation** (`pathway_interrogation.py`): Analyzes what distinguishes different pathways
3. **Medication Integration** (`medication_integration.py`): Links medications to discovered pathways

## Data Requirements

- **Y matrix**: Binary disease event matrix (407K patients, 348 diseases, 52 time points)
  - Location: `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt`
  - Note: Only the first 400K rows are used (matching thetas)
  - The first 400K rows correspond to the first 400K eids in `processed_ids.csv`

- **Thetas**: Signature loadings (400K patients, 21 signatures, 52 time points)
  - Location: `/Users/sarahurbut/aladynoulli2/pyScripts/thetas.npy`
  - Patient index i in thetas corresponds to row i in Y and eid i in processed_ids.csv

- **Processed IDs**: Ordered list of eids for the first 400K patients
  - Location: `/Users/sarahurbut/aladynoulli2/pyScripts/processed_ids.csv`
  - Patient index i (0-indexed) maps to processed_ids[i]

- **Disease Names**: List of 348 disease names
  - Location: `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv`

- **GP Scripts**: Medication prescription data (optional, for medication integration)
  - Location: `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/gp_scripts.txt`

## Quick Start

### Option 1: Use the Complete Pipeline

```python
from run_pathway_analysis import run_complete_pathway_analysis

# Analyze a single disease
results = run_complete_pathway_analysis("myocardial infarction", n_pathways=4)
```

### Option 2: Use Individual Components

```python
from pathway_discovery import load_full_data, discover_disease_pathways
from pathway_interrogation import interrogate_disease_pathways
from medication_integration import integrate_medications_with_pathways

# Load data
Y, thetas, disease_names, processed_ids = load_full_data()

# Discover pathways
pathway_data = discover_disease_pathways(
    "myocardial infarction", Y, thetas, disease_names, 
    n_pathways=4, method='average_loading'
)

# Interrogate pathways
results = interrogate_disease_pathways(pathway_data, Y, thetas, disease_names)

# Integrate medications
med_results = integrate_medications_with_pathways(
    pathway_data, Y, thetas, disease_names, processed_ids
)
```

### Option 3: Use the Notebook

Open `test_pathway_analysis.ipynb` for an interactive analysis.

## Clustering Methods

Two methods are available for discovering pathways:

### 1. Average Loading Method (Default)
- Clusters patients by their average signature loading over time
- Matches the R script methodology (`mean_thetas`)
- Best for finding patients with similar overall signature profiles

```python
pathway_data = discover_disease_pathways(
    disease_name, Y, thetas, disease_names, 
    method='average_loading'
)
```

### 2. Trajectory Similarity Method
- Clusters by pre/post-disease trajectory features
- Includes slopes, variance, and temporal dynamics
- Best for finding patients with similar disease progression patterns

```python
pathway_data = discover_disease_pathways(
    disease_name, Y, thetas, disease_names, 
    method='trajectory_similarity'
)
```

## Outputs

### Pathway Discovery
- Pathway labels for each patient
- Pathway size distribution
- Trajectory features used for clustering

### Pathway Interrogation
- **Signature Discrimination Scores**: Which signatures most distinguish pathways
- **Disease Patterns**: What other diseases characterize each pathway
- **Age at Onset**: Age distribution by pathway
- **Visualizations**:
  - Signature trajectory deviations from population reference
  - Pathway size distribution
  - Age at disease onset by pathway

### Medication Integration
- **Long-term Medications**: Drugs prescribed ≥5 times per patient
- **Medication Diversity**: Number of unique long-term medications per patient
- **Pathway-Specific Medications**: Top medications for each pathway
- **Visualizations**:
  - Medication coverage by pathway
  - Medication diversity by pathway
  - Top medications heatmap across pathways

## Key Concepts

### Patient Morphs
The analysis explores how patients dynamically transition between different signature states over time, revealing heterogeneity in disease progression.

### Signature Deviations
Rather than plotting absolute signature loadings, we plot deviations from a population reference to highlight differences between pathways.

### Pathway Characterization
Each pathway is characterized by:
- Distinct signature trajectory patterns
- Specific co-occurring diseases
- Age at disease onset
- Medication patterns

## Example Use Cases

### 1. CAD Heterogeneity
```python
# Compare rheumatologic vs. metabolic pathways to CAD
results_cad = run_complete_pathway_analysis("myocardial infarction", n_pathways=4)
# Hypothesis: Patients who transitioned from rheumatologic diseases to CAD 
# will cluster separately from those with traditional metabolic risk factors
```

### 2. Radiation-Induced Breast Cancer
```python
# Identify breast cancer patients with prior cancer exposure
results_bc = run_complete_pathway_analysis("breast cancer", n_pathways=4)
# Look for pathways enriched for prior radiation or other cancers
```

### 3. Multi-Disease Analysis
```python
from run_pathway_analysis import run_multiple_diseases

diseases = ["myocardial infarction", "breast cancer", "diabetes", "stroke"]
all_results = run_multiple_diseases(diseases, n_pathways=4)
```

## Technical Details

### ID Mapping
- Patient IDs in the analysis are 0-indexed (0 to 399,999)
- Patient index i directly corresponds to:
  - Row i in Y matrix
  - Row i in thetas array
  - Entry i in processed_ids.csv (which contains the actual eid)
- For medication integration: patient_index → processed_ids[patient_index] = eid

### Memory Efficiency
- Y is subset to first 400K patients to match thetas
- Uses torch tensors for efficient computation
- Clustering performed on standardized features

### Visualization
- All plots show deviations from population reference (like R script)
- Multiple pathways shown in different colors
- Includes confidence bands (±1 SD)

## Troubleshooting

### "Index out of bounds" Error
- Ensure Y is subset to first 400K patients
- Check that thetas and Y have matching patient dimensions

### "No medication data found"
- Verify `processed_ids.csv` is loaded correctly
- Check that `gp_scripts.txt` contains the eids from processed_ids

### Empty Pathways
- Try reducing `n_pathways` if disease is rare
- Check that disease name matches entries in disease_names.csv

## Files in this Directory

- `pathway_discovery.py`: Core pathway discovery functions
- `pathway_interrogation.py`: Pathway analysis and visualization
- `medication_integration.py`: Medication pattern analysis
- `run_pathway_analysis.py`: Complete pipeline wrapper
- `test_pathway_analysis.ipynb`: Interactive notebook
- `README_PATHWAY_ANALYSIS.md`: This file

