# Running UKB-MGB Pathway Comparison

## Quick Start

### Option 1: Run Everything (Recommended)

```python
from run_full_ukb_mgb_comparison import main

# This will:
# 1. Load or run UKB analysis
# 2. Run MGB analysis  
# 3. Match pathways by disease patterns
# 4. Compare matched pathways
# 5. Generate summary

results = main()
```

### Option 2: Step-by-Step

```python
# Step 1: Load or run UKB analysis
from run_complete_pathway_analysis_deviation_only import run_deviation_only_analysis
import pickle

# Check if UKB results exist
if os.path.exists('output_10yr/complete_analysis_results.pkl'):
    with open('output_10yr/complete_analysis_results.pkl', 'rb') as f:
        ukb_results = pickle.load(f)
else:
    ukb_results = run_deviation_only_analysis(
        "myocardial infarction", 
        n_pathways=4, 
        output_dir='output_10yr',
        lookback_years=10
    )

# Step 2: Run MGB analysis
from run_mgb_deviation_analysis_and_compare import run_deviation_analysis_mgb

mgb_results = run_deviation_analysis_mgb(
    target_disease="myocardial infarction",
    n_pathways=4,
    lookback_years=10,
    output_dir='mgb_deviation_analysis_output'
)

# Step 3: Match pathways by disease patterns
from match_pathways_by_disease_patterns import match_pathways_between_cohorts
from pathway_discovery import load_full_data
from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
import torch

# Load data
Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
Y_mgb, thetas_mgb, disease_names_mgb, _ = load_mgb_data_from_model()

# Convert to torch
if isinstance(Y_ukb, np.ndarray):
    Y_ukb = torch.from_numpy(Y_ukb)
if isinstance(Y_mgb, np.ndarray):
    Y_mgb = torch.from_numpy(Y_mgb)

# Match pathways
pathway_matching = match_pathways_between_cohorts(
    ukb_results['pathway_data_dev'], Y_ukb, disease_names_ukb,
    mgb_results['pathway_data'], Y_mgb, disease_names_mgb,
    top_n_diseases=20
)

# Step 4: Compare matched pathways
from match_pathways_by_disease_patterns import compare_matched_pathways

matched_comparison = compare_matched_pathways(
    ukb_results['pathway_data_dev'],
    mgb_results['pathway_data'],
    pathway_matching,
    ukb_results,
    mgb_results
)
```

## What It Does

1. **UKB Analysis**: Runs deviation-based pathway discovery on UKB (or loads existing results)
2. **MGB Analysis**: Runs the same deviation-based pathway discovery on MGB
3. **Pathway Matching**: Matches pathways between cohorts by disease enrichment patterns (not index numbers!)
4. **Comparison**: Compares matched pathways (sizes, proportions, disease patterns)

## Key Points

- **Pathway labels are arbitrary**: UKB Pathway 0 might match MGB Pathway 2
- **Matching is by biological content**: Pathways are matched based on which diseases are enriched, not by index numbers
- **Disease names differ**: The matching function handles different disease naming conventions between cohorts

## Output Files

- `pathway_matching_results.pkl`: Pathway matching results
- `ukb_mgb_comparison_summary.txt`: Text summary of findings
- `mgb_deviation_analysis_output/`: MGB analysis results
- `output_10yr/`: UKB analysis results (if run)

## Interpreting Results

The pathway matching will show:
- Which UKB pathway corresponds to which MGB pathway
- Disease enrichment patterns for each matched pair
- Similarity scores between matched pathways

Example output:
```
UKB Pathway 0 ↔ MGB Pathway 2 (similarity: 0.85)
  Top matching diseases:
    UKB: coronary atherosclerosis (enrichment: 11.2)
    MGB: coronary_artery_disease (enrichment: 10.8)
```

This means UKB Pathway 0 and MGB Pathway 2 have similar disease patterns (85% similarity) and both are enriched for coronary atherosclerosis.

