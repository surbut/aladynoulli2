# Reviewer Response Materials

This directory contains all materials for responding to reviewer comments.

## Structure

```
reviewer_responses/
├── README.md                          # This file
├── REVIEWER_RESPONSE_MASTER.Rmd       # Master R Markdown (knit to PDF/HTML)
├── notebooks/                          # Interactive Jupyter notebooks
│   ├── REVIEWER_QUESTIONS_INDEX.ipynb  # Master index
│   ├── R1_Q1_Selection_Bias.ipynb
│   ├── R1_Q3_Clinical_Meaning.ipynb
│   ├── R1_Q7_Heritability.ipynb
│   ├── R2_Temporal_Leakage.ipynb
│   └── R3_Q8_Heterogeneity.ipynb
└── docs/                               # Documentation files
    ├── REVIEWER_RESPONSE_ORGANIZATION.md
    ├── README_FOR_REVIEWERS.md
    ├── WHAT_NEEDS_RERUN.md
    └── DISCOVERY_VS_PREDICTION_MODES.md
```

## How to Use

### For Reviewers

1. **Start here**: Open `notebooks/REVIEWER_QUESTIONS_INDEX.ipynb`
2. **Navigate**: Click on any question to go to its analysis notebook
3. **Run analyses**: Each notebook is self-contained and can be run independently

### For Authors (Response Letter)

**Option 1: Generate HTML Master Document** (Python - Recommended)
```bash
python generate_master_document.py
# Opens REVIEWER_RESPONSE_MASTER.html
# Can print to PDF from browser
```

**Option 2: Convert Notebooks to HTML/PDF** (Python nbconvert)
```bash
./knit_notebooks.sh
# Converts all notebooks to HTML (and optionally PDF)
# Output in knitted_output/ directory
```

**Option 3: Use R Markdown** (If you have R installed)
```r
# In R
rmarkdown::render("REVIEWER_RESPONSE_MASTER.Rmd", output_format = "html_document")
# or
rmarkdown::render("REVIEWER_RESPONSE_MASTER.Rmd", output_format = "pdf_document")
```

**Note**: The R Markdown file (`REVIEWER_RESPONSE_MASTER.Rmd`) is provided as a template/outline. The Python notebooks are the source of truth.

### When to Re-Knit

**Re-run the knitting script** (`./knit_notebooks.sh`) after:
- ✅ AWS age offset predictions complete (new `pi_enroll_fixedphi_age_offset_*` files)
- ✅ Any notebook results are updated
- ✅ Before final submission (to ensure all results are current)

The knitted HTML/PDF outputs will reflect the latest results loaded by the notebooks.

## Quick Reference

| Reviewer | Question | Notebook | Status |
|----------|----------|----------|--------|
| R1 | Q1: Selection bias | `R1_Q1_Selection_Bias.ipynb` | ✅ Complete |
| R1 | Q3: Clinical meaning | `R1_Q3_Clinical_Meaning.ipynb` | ✅ Complete |
| R1 | Q7: Heritability | `R1_Q7_Heritability.ipynb` | ✅ Complete |
| R2 | Temporal leakage | `R2_Temporal_Leakage.ipynb` | ✅ Complete |
| R3 | Q8: Heterogeneity | `R3_Q8_Heterogeneity.ipynb` | ✅ Complete |

## Notes

- All notebooks use relative paths from this directory
- Results are saved in `../results/` directory
- Data paths may need adjustment based on your setup

