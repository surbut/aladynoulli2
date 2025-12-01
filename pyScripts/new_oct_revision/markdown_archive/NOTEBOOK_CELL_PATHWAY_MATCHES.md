# Notebook Cell: Show Pathway Matches

Paste this into your notebook to see which pathways match:

```python
from show_pathway_matches import show_pathway_matches

# This will:
# 1. Load UKB results (or run if not found)
# 2. Run MGB analysis (always re-runs)
# 3. Match pathways by disease patterns
# 4. Show simple matching table

results = show_pathway_matches(force_rerun_mgb=True)

# Show just the matches
print("\n" + "="*60)
print("PATHWAY MATCHES")
print("="*60)
print(f"{'UKB Pathway':<15} {'MGB Pathway':<15} {'Similarity':<15}")
print("-" * 60)

pathway_matching = results['pathway_matching']
best_matches = pathway_matching['best_matches']
similarities = pathway_matching['pathway_similarities']

for ukb_id in sorted(best_matches.keys()):
    mgb_id = best_matches[ukb_id]
    similarity = similarities[(ukb_id, mgb_id)]
    print(f"Pathway {ukb_id:<12} Pathway {mgb_id:<12} {similarity:<15.3f}")
```

