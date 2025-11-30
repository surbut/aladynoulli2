# Reviewer Response Analyses

This section contains all analyses addressing reviewer questions and concerns.

## 🎯 Start Here

**Navigate to**: [`notebooks/index.ipynb`](notebooks/index.ipynb)

The index notebook provides links to all reviewer response analyses, organized by reviewer.

## 📁 Structure

```
reviewer_responses/
├── README.md                      # This file
├── notebooks/
│   ├── index.ipynb                # Start here - navigation hub
│   ├── R1/                        # Referee #1 analyses
│   ├── R2/                        # Referee #2 analyses
│   ├── R3/                        # Referee #3 analyses
│   └── framework/                 # Framework overview
├── preprocessing/                 # Data preprocessing (shared)
└── scripts/                       # Reviewer-specific scripts
```

## 📊 Notebooks by Reviewer

### Referee #1: Human Genetics, Disease Risk
- Selection bias, lifetime risk, clinical meaning, heritability, AUC comparisons, age-specific analyses
- See [`notebooks/index.ipynb`](notebooks/index.ipynb) for complete list

### Referee #2: EHRs
- Temporal leakage, model validity
- See [`notebooks/index.ipynb`](notebooks/index.ipynb) for complete list

### Referee #3: Statistical Genetics, PRS
- Competing risks, heterogeneity, population stratification, model comparisons
- See [`notebooks/index.ipynb`](notebooks/index.ipynb) for complete list

## 🔧 Technical Notes

- **Results**: Stored in `../results/` (one level up from `reviewer_responses/`)
- **Source Code**: Shared code is in `pyScripts_forPublish/` (not duplicated here)
- **Paths**: Notebooks use absolute paths for reliability
- **Data**: All notebooks are self-contained and can be run independently

## 📝 Framework Overview

For an overview of the discovery and prediction framework, see:
[`notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb`](notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb)
