# ALADYNOULLI: A Bayesian Voyage Through the Genome and the EHR

## Book Outline

**Working Title**: *Dynamic Disease Modeling: A Bayesian Framework for Genomic Discovery and Clinical Prediction*

**Tagline**: From Electronic Health Records to Personalized Risk Trajectories

---

## Preface
- The shape of a story (Vonnegut reference)
- Why this book? The gap between genomics and clinical care
- Who is this for? Statisticians, geneticists, clinicians, data scientists

---

## Part I: The Clinical Problem

### Chapter 1: Everyone is Moving
- The ED case: 34yo with chest pain
- Why static risk scores fail
- The Cox proportional hazards assumption and its limitations
- Dynamic hazard: different roles, same end

### Chapter 2: The Promise of Longitudinal Data
- Electronic health records as a window into disease progression
- UK Biobank, Mass General Brigham, All of Us
- What 52 years of follow-up tells us
- The challenge: 348 diseases, millions of patients

### Chapter 3: Genetics Matters
- Polygenic risk scores and early disease
- The "when do you ask the question?" problem
- Genetics impacts not just early disease, but all disease early

---

## Part II: The ALADYNOULLI Model

### Chapter 4: Everyone is a Bayesian
- Thomas Bayes and the history of life insurance
- Prior, likelihood, posterior: a framework for updating beliefs
- Joint consideration: discovery AND prediction

### Chapter 5: The Mathematical Framework
- The model specification:
  - λ: Individual predilection to signatures
  - φ: Signature-disease associations
  - θ = softmax(λ): Signature proportions
  - π = Σ θ·φ: Disease hazard
- Gaussian processes for temporal smoothness
- The likelihood function and censoring

### Chapter 6: Genetic Effects on Disease Trajectories
- γ_level: Baseline genetic effects
- The GP prior as soft constraint
- Initialization from linear regression
- Identifiability considerations

### Chapter 7: Computational Implementation
- Vectorized likelihood evaluation
- PyTorch optimization
- From days to minutes: scaling to 400,000 patients
- Real-time patient refitting

---

## Part III: Discovering Disease Signatures

### Chapter 8: Latent Patterns of Disease
- What is a signature?
- Cardiovascular, neoplastic, metabolic, psychiatric signatures
- Visualization and interpretation
- The healthy signature (Signature 20)

### Chapter 9: Consistency Across Populations
- Cross-cohort validation: UKB, MGB, AoU
- Composition preservation probability index
- What stays the same, what adapts

### Chapter 10: Heterogeneity Within Disease
- Breast cancer subtypes
- Major depression clusters
- Myocardial infarction patterns
- Cohen's d effect sizes and clinical meaning

---

## Part IV: Genetic Validation

### Chapter 11: Common Variant Associations (GWAS)
- Signature-based GWAS methodology
- 151 genome-wide significant loci
- Novel cardiovascular associations not found in single-trait GWAS
- IL6R, SMAD3, ZPR1, SCARB1 and known biology

### Chapter 12: Rare Variant Associations (RVAS)
- Gene-based testing with REGENIE
- LDLR, LPA, APOB: established genes
- TTN, BRCA2, TET2: pleiotropic effects
- DEFB1 and the healthy signature

### Chapter 13: Heritability and Biological Relevance
- Observed-scale heritability estimation
- Signatures vs component diseases
- Why aggregation increases power
- Familial hypercholesterolemia carrier enrichment
- CHIP and cardiovascular signatures

---

## Part V: Clinical Prediction

### Chapter 14: Outperforming Established Risk Scores
- ASCVD: vs PCE, QRISK3, PREVENT
- Breast cancer: vs GAIL (1-year and 10-year)
- 28 diseases, systematic comparison
- Absolute vs relative AUC improvements

### Chapter 15: Dynamic Risk Updating
- Walking the timeline: from past to future
- Annual updates and prediction improvement
- The value of competing risks
- Leave-one-out validation

### Chapter 16: The Aladynoulli Web Application
- http://aladynoulli.hms.harvard.edu
- Interactive patient risk visualization
- Clinical deployment considerations

---

## Part VI: Robustness and Validation

### Chapter 17: Selection Bias and Inverse Probability Weighting
- UK Biobank's healthy volunteer bias
- The IPW-weighted likelihood
- Simulation validation: 90% women dropped
- Preserving biological signal while correcting bias

### Chapter 18: Population Stratification
- Ancestry principal components
- Cross-ancestry stability of signatures
- What φ captures vs what λ captures

### Chapter 19: Temporal Leakage and Washout
- The reverse causation concern
- 1, 3, 6 month washout windows
- Performance at multiple prediction horizons
- Robustness to data availability

### Chapter 20: Model Diagnostics
- Calibration analysis (MSE = 4.67 × 10⁻⁷)
- Gradient diagnostics and convergence
- Sensitivity to hyperparameters

---

## Part VII: Advanced Topics

### Chapter 21: Competing Risks and Multi-State Models
- Why competing risks improve prediction
- The MSGene comparison
- Absorbing states and transitions

### Chapter 22: Linear vs Nonlinear Mixing
- The softmax transformation
- Alternative architectures considered
- Why interpretability matters

### Chapter 23: Connections to Other Methods
- Comparison to Delphi-2M (transformer-based)
- Topic models and latent factor analysis
- The spectrum from black-box to interpretable

### Chapter 24: Future Directions
- Genetic effects on progression speed (γ_slope)
- Treatment effect estimation
- Causal inference from observational data
- Multi-modal data integration

---

## Appendices

### Appendix A: Mathematical Details
- Full model specification
- GP kernel functions
- Optimization algorithms

### Appendix B: Complete Validation Analyses
- B1: Selection Bias (R1_Q1)
- B2: Lifetime Risk (R1_Q2)
- B3: Clinical Meaning (R1_Q3)
- B4: ICD vs PheCode Comparison (R1_Q3)
- B5: Heritability (R1_Q7)
- B6: AUC Comparisons (R1_Q9)
- B7: Age-Specific Analysis (R1_Q10)
- B8: Biological Plausibility - CHIP (R1)
- B9: Clinical Utility - Dynamic Updating (R1)
- B10: GWAS Validation (R1)
- B11: RVAS Validation (R1)
- B12: Competing Risks (R1, R3)
- B13: LOO Validation (R1)
- B14: Model Validity & Learning (R2, R3)
- B15: Temporal Leakage (R2)
- B16: Washout Analysis (R2)
- B17: Cross-Cohort Similarity (R3)
- B18: Fixed vs Joint Phi (R3)
- B19: Full E vs Reduced E (R3)
- B20: Linear vs Nonlinear (R3)
- B21: Population Stratification (R3)
- B22: Heterogeneity Analysis (R3)
- B23: Discovery-Prediction Framework Overview

### Appendix C: Code and Reproducibility
- GitHub repository structure
- Running the model
- Reproducing all figures

### Appendix D: Supplementary Figures and Tables

---

## Estimated Length
- Main text: ~300 pages
- Appendices: ~150 pages
- Total: ~450 pages

## Target Audience
1. **Primary**: Biostatisticians, genetic epidemiologists, computational biologists
2. **Secondary**: Clinical researchers, health data scientists
3. **Tertiary**: Graduate students in biostatistics/bioinformatics

## Potential Publishers
1. Springer (Statistics for Biology and Health series)
2. Chapman & Hall/CRC (Biostatistics series)
3. Cambridge University Press
4. Open access: Bookdown (like R for Data Science)

---

## Timeline (if pursued)
1. **Phase 1**: Consolidate existing material (2-3 months)
2. **Phase 2**: Write connecting prose, expand methods (3-4 months)
3. **Phase 3**: Create unified figures, edit (2-3 months)
4. **Phase 4**: Publisher submission, review (6-12 months)

---

*Draft outline created: January 30, 2026*
