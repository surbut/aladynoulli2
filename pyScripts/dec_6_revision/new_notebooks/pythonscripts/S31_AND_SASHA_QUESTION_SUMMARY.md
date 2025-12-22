# S31 and Sasha's Question Summary

## S31: Population Stratification (PC Adjustments)

**Location**: `R3_Population_Stratification_Ancestry.ipynb`

### What S31 Should Include:

1. **Lambda Shift** (with/without PC adjustment)
   - Show how lambda (signature loadings) changes when PC adjustment is applied
   - Compare lambda trajectories between models with/without PCs

2. **Theta Deviation Line Plots** (deviation from reference)
   - Plot theta deviations from reference theta by ancestry
   - Show continuous ancestry effects (not binary)
   - Key signatures: Sig 5 (SAS), Sig 15 (EAS), plus most variable signatures

3. **Phis Being the Same** (with/without PC adjustment)
   - Show phi correlation >0.99 between WITH and WITHOUT PC models
   - Demonstrate that PC adjustment preserves biological signal (phi) while controlling for stratification

### Code References:

**Theta Deviations**:
- Cell 12: Calculate deviations from reference theta
- Cell 14: Identify signatures of interest
- Uses `reference_thetas.csv` as reference baseline

**Phi Comparison**:
- Cell 9a (around line 1449): Phi comparison code
- Loads phi from batches with/without PCs
- Calculates correlation and mean difference

**Lambda/Theta Loading**:
- Cell 2: `assemble_new_model_with_no_pcs()` - loads thetas without PCs
- Cell 3: `assemble_new_model_with_pcs()` - loads thetas with PCs

---

## Sasha's Question: Selection Bias with Women (IPW Correction)

**Question**: How does selection bias (e.g., dropping 90% of women) affect the model, and can IPW correct for it?

**Key Insight**: We should use the **same prevalence initialization** for both models (full population vs. biased sample) to ensure fair comparison, similar to what we learned from IPW analysis.

### What to Compare:

1. **Three Scenarios**:
   - **Full population** (baseline) - all 400K patients
   - **Biased sample** (90% women dropped) - no adjustment
   - **Biased sample with IPW** - reweighted to recover full population

2. **Key Comparisons**:
   - **Prevalence**: Show drop without adjustment, recovery with IPW
   - **Phi**: Should be nearly identical (correlation >0.99) if using same prevalence init
   - **Lambda/Pi**: Can shift - shows how model adapts to reweighted population
   - **Model predictions**: Compare pi from models trained on biased sample vs. full population

### Code Location:

**Script**: `demonstrate_ipw_correction.py`

**What it currently does**:
- Drops 90% of women (simulates selection bias)
- Computes prevalence for: full population, biased sample (no adjustment), biased sample (with IPW)
- Shows how IPW recovers prevalence patterns

**What to add for complete answer**:
- Train models on biased sample (with/without IPW weights)
- Compare phi, lambda, and pi from these models
- Use **same prevalence_t** for initialization in all models (full, biased, biased+IPW)
- This ensures phi stability while showing lambda/pi adaptability

### Expected Results:

- **Prevalence**: Drops substantially without adjustment, recovers with IPW
- **Phi correlation**: >0.99 (nearly identical) if using same prevalence init
- **Lambda/Pi**: Can show shifts - model adapts to reweighted population
- **Interpretation**: IPW corrects for selection bias without changing biological disease-signature associations (phi)

---

## Next Steps:

1. **Create S31 script** combining:
   - Lambda shift visualization
   - Theta deviation line plots
   - Phi comparison scatter plot

2. **For Sasha's question**: Ensure both models use same prevalence initialization when training/comparing

