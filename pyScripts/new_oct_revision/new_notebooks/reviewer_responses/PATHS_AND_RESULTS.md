# Paths and Results Directory

## ✅ Moving Notebooks is Safe

**Good news**: Moving notebooks to `reviewer_responses/notebooks/` will **NOT** break results files because:

1. **Notebooks use absolute paths** for reading results:
   - `results_base = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results')`
   - These paths work regardless of where the notebook is located

2. **Scripts write results** (not notebooks):
   - Scripts like `generate_washout_predictions.py` write to `results/`
   - Notebooks only **read** from `results/`
   - Moving notebooks doesn't affect where scripts write

3. **Results directory stays in place**:
   - `results/` remains at: `/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/`
   - All notebooks (moved or not) reference this same absolute path

## 📁 Current Structure

```
new_notebooks/
├── results/                          # ← Results stay here (unchanged)
│   ├── washout/
│   ├── age_offset/
│   └── comparisons/
├── reviewer_responses/
│   └── notebooks/                    # ← Notebooks moved here
│       ├── R1_Q1_Selection_Bias.ipynb
│       └── ...
└── (other notebooks)
```

## 🔍 How It Works

**Notebooks read from** (absolute paths):
- `/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/`
- Works from any location

**Scripts write to** (relative or absolute):
- `results/washout/` (relative to script location)
- Or absolute paths specified in scripts

**No conflicts**: Notebooks read, scripts write, both use the same `results/` directory.

## ✅ Verification

All notebooks use absolute paths like:
```python
results_base = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results')
```

This means:
- ✅ Moving notebooks won't break result loading
- ✅ Results directory stays in one place
- ✅ All notebooks reference the same results

## 📝 If You Need Relative Paths

If you want notebooks to use relative paths (for portability), you can update them to:

```python
# Relative to notebook location
import os
notebook_dir = Path(__file__).parent if '__file__' in globals() else Path(os.getcwd())
results_base = notebook_dir.parent.parent / 'results'
```

But **absolute paths are fine** and more explicit for this use case.

