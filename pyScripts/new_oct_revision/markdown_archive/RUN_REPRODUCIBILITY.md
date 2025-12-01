# Run Pathway Reproducibility Analysis

## Quick Start (Notebook Cell)

```python
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from show_pathway_reproducibility import main

# Run reproducibility analysis
# This will:
# 1. Get pathway matches between UKB and MGB
# 2. Create reproducibility figure (sizes, proportions, correlations)
# 3. Create signature deviation plots showing ALL signatures side-by-side
# 4. Create heatmaps of signature deviations

results = main(force_rerun_mgb=False)  # Use existing MGB results
# OR
results = main(force_rerun_mgb=True)   # Force re-run MGB analysis
```

## What It Generates

1. **`pathway_reproducibility_ukb_mgb.png`** - 6-panel figure showing:
   - Pathway sizes comparison
   - Pathway proportions correlation
   - Similarity scores
   - Age at onset comparison
   - Summary statistics

2. **`signature_deviation_trajectories_all_sigs_ukb_mgb.png`** - All signatures plotted side-by-side:
   - Each matched pathway pair (UKB ↔ MGB)
   - All 20-21 signatures overlaid on each plot
   - Shows full trajectory (age 30-80)
   - ⚠️ Note: Signature indices are arbitrary - compare patterns/shapes, not index numbers

3. **`signature_deviation_heatmaps_all_sigs_ukb_mgb.png`** - Heatmap view:
   - 2D visualization of all signatures
   - Signatures (rows) vs time (columns)
   - Easier to compare overall patterns

## Key Points

- **Pathways are matched by disease patterns** (not signature indices)
- **Signature indices are arbitrary** - UKB Sig 5 ≠ MGB Sig 5 necessarily
- **Compare overall patterns/shapes** - similar biological pathways should show similar deviation patterns
- **Colors match by index** but biological correspondence may differ

## Expected Output

The signature plots should show:
- Similar overall **shapes/patterns** for matched pathways
- Different **colors** (because indices don't correspond)
- Similar **magnitudes** of deviation for corresponding biological processes
- Similar **temporal patterns** (which signatures are elevated/depressed at which ages)

