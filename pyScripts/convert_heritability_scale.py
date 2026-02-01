"""
Convert observed-scale heritability to liability-scale heritability for binary traits.

Based on the formula from:
- Lee et al. 2011 (AJHG): https://doi.org/10.1016/j.ajhg.2011.08.014
- Grotzinger et al. 2023 (Nat Genet): https://pmc.ncbi.nlm.nih.gov/articles/PMC10066905/

Formula (Equation 2):
h²_l = h²_o * [K²(1-K)²] / [z² * P * (1-P)]

Where:
- h²_o = observed-scale heritability (from LDSC)
- h²_l = liability-scale heritability
- K = population prevalence
- P = sample prevalence (proportion of cases in the study)
- z = height of standard normal PDF at the threshold corresponding to K
      z = φ(Φ⁻¹(1-K)) where φ is the standard normal PDF and Φ⁻¹ is the inverse CDF

Reference: Neale Lab UKB LDSC results for coronary atherosclerosis:
- Observed h² = 0.0311
- Liability h² = 0.1614 (assuming population prevalence = sample prevalence = 0.0397)
https://nealelab.github.io/UKBB_ldsc/h2_summary_I9_CORATHER.html
"""

import numpy as np
from scipy import stats

def observed_to_liability_h2(h2_obs, sample_prevalence, population_prevalence=None):
    """
    Convert observed-scale heritability to liability-scale heritability.
    
    Parameters:
    -----------
    h2_obs : float
        Observed-scale heritability from LDSC
    sample_prevalence : float
        Proportion of cases in the study sample (P)
    population_prevalence : float, optional
        Population prevalence (K). If None, assumes equal to sample_prevalence.
    
    Returns:
    --------
    h2_liability : float
        Liability-scale heritability
    """
    if population_prevalence is None:
        population_prevalence = sample_prevalence
    
    K = population_prevalence  # population prevalence
    P = sample_prevalence       # sample prevalence
    
    # Calculate threshold on standard normal corresponding to population prevalence
    # If K is the proportion affected, threshold t is where Φ(t) = 1 - K
    threshold = stats.norm.ppf(1 - K)
    
    # Height of standard normal PDF at this threshold
    z = stats.norm.pdf(threshold)
    
    # Apply conversion formula
    h2_liability = h2_obs * (K**2 * (1 - K)**2) / (z**2 * P * (1 - P))
    
    return h2_liability


def liability_to_observed_h2(h2_liability, sample_prevalence, population_prevalence=None):
    """
    Convert liability-scale heritability to observed-scale heritability.
    (Inverse of the above function)
    """
    if population_prevalence is None:
        population_prevalence = sample_prevalence
    
    K = population_prevalence
    P = sample_prevalence
    
    threshold = stats.norm.ppf(1 - K)
    z = stats.norm.pdf(threshold)
    
    h2_obs = h2_liability * (z**2 * P * (1 - P)) / (K**2 * (1 - K)**2)
    
    return h2_obs


# ============================================================================
# Verify with Neale Lab CAD example
# ============================================================================
print("=" * 70)
print("VERIFICATION: Neale Lab Coronary Atherosclerosis (I9_CORATHER)")
print("=" * 70)

# From Neale Lab website:
# Cases: 14,334, Controls: 346,860, Total: 361,194
neale_cases = 14334
neale_total = 361194
neale_sample_prev = neale_cases / neale_total  # 0.0397

neale_h2_obs = 0.0311
neale_h2_liability_reported = 0.1614

# Convert using our formula
neale_h2_liability_calc = observed_to_liability_h2(
    h2_obs=neale_h2_obs,
    sample_prevalence=neale_sample_prev,
    population_prevalence=neale_sample_prev  # Neale assumes pop prev = sample prev
)

print(f"Sample prevalence: {neale_sample_prev:.4f}")
print(f"Observed h² (reported): {neale_h2_obs:.4f}")
print(f"Liability h² (reported): {neale_h2_liability_reported:.4f}")
print(f"Liability h² (calculated): {neale_h2_liability_calc:.4f}")
print(f"Match: {'YES' if abs(neale_h2_liability_calc - neale_h2_liability_reported) < 0.01 else 'NO'}")

# ============================================================================
# Convert component CVD traits from our analysis
# ============================================================================
print("\n" + "=" * 70)
print("COMPONENT CVD TRAITS: Observed to Liability Scale Conversion")
print("=" * 70)

# Component CVD traits with observed h² from our LDSC analysis
# Need to estimate prevalences from UK Biobank
# Using approximate prevalences from UK Biobank literature/our data

# Actual UK Biobank prevalences from case_control_sig5.tsv phenotypes
# Derived from E_matrix.pt where E < 51 indicates disease occurred
# Total N = 400,000 patients
component_traits = {
    'Angina Pectoris': {
        'h2_obs': 0.0340,
        'h2_se': 0.0024,
        'sample_prev': 28797 / 400000,  # 0.0720 (28,797 cases)
    },
    'Coronary Atherosclerosis': {
        'h2_obs': 0.0477,
        'h2_se': 0.0035,
        'sample_prev': 35487 / 400000,  # 0.0887 (35,487 cases)
    },
    'Hypercholesterolemia': {
        'h2_obs': 0.0444,
        'h2_se': 0.0032,
        'sample_prev': 71013 / 400000,  # 0.1775 (71,013 cases)
    },
    'Myocardial Infarction': {
        'h2_obs': 0.0316,
        'h2_se': 0.0024,
        'sample_prev': 24695 / 400000,  # 0.0617 (24,695 cases)
    },
    'Other Acute IHD': {
        'h2_obs': 0.0033,
        'h2_se': 0.0013,
        'sample_prev': 3667 / 400000,  # 0.0092 (3,667 cases)
    },
    'Other Chronic IHD': {
        'h2_obs': 0.0339,
        'h2_se': 0.0023,
        'sample_prev': 30630 / 400000,  # 0.0766 (30,630 cases)
    },
    'Unstable Angina': {
        'h2_obs': 0.0117,
        'h2_se': 0.0015,
        'sample_prev': 7984 / 400000,  # 0.0200 (7,984 cases)
    },
}

print(f"\n{'Trait':<30} {'h² (obs)':<12} {'Prev':<8} {'h² (liab)':<12} {'Ratio':<8}")
print("-" * 70)

for trait, data in component_traits.items():
    h2_liability = observed_to_liability_h2(
        h2_obs=data['h2_obs'],
        sample_prevalence=data['sample_prev'],
        population_prevalence=data['sample_prev']
    )
    ratio = h2_liability / data['h2_obs']
    
    print(f"{trait:<30} {data['h2_obs']:.4f}       {data['sample_prev']:.3f}    {h2_liability:.4f}       {ratio:.1f}x")

# ============================================================================
# Generate LaTeX table with liability scale column
# ============================================================================
print("\n" + "=" * 70)
print("LaTeX TABLE with Liability Scale Column")
print("=" * 70)

latex_table = r"""
\begin{table}[H]
\centering
\small
\caption{LDSC heritability estimates for component CVD traits on both observed and liability scales. 
Observed-scale heritabilities (h$^2_{\text{obs}}$ $\approx$ 0.03--0.05) are directly comparable to Signature 5 (h$^2$ = 0.0414). 
Liability-scale estimates are provided for comparison with literature values, which typically report heritability on this scale 
(e.g., Neale Lab reports coronary atherosclerosis h$^2_{\text{liability}}$ = 0.16, h$^2_{\text{observed}}$ = 0.03; 
\url{https://nealelab.github.io/UKBB_ldsc/h2_summary_I9_CORATHER.html}).
Conversion uses the formula from Lee et al. 2011: 
h$^2_l$ = h$^2_o \cdot K^2(1-K)^2 / [z^2 \cdot P(1-P)]$, 
where $K$ is population prevalence, $P$ is sample prevalence, and $z$ is the standard normal density at the liability threshold.}
\label{tab:component_cvd_heritabilities_liability}
\begin{tabular}{lccccc}
\toprule
\textbf{Component CVD Trait} & \textbf{h$^2_{\text{obs}}$ (SE)} & \textbf{Prevalence} & \textbf{h$^2_{\text{liability}}$} & \textbf{Intercept} \\
\midrule
"""

for trait, data in component_traits.items():
    h2_liability = observed_to_liability_h2(
        h2_obs=data['h2_obs'],
        sample_prevalence=data['sample_prev'],
        population_prevalence=data['sample_prev']
    )
    latex_table += f"{trait} & {data['h2_obs']:.4f} ({data['h2_se']:.4f}) & {data['sample_prev']:.3f} & {h2_liability:.3f} & -- \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

print(latex_table)

# ============================================================================
# Key message for reviewers
# ============================================================================
print("\n" + "=" * 70)
print("KEY MESSAGE FOR REVIEWERS")
print("=" * 70)
print("""
The seemingly low heritability estimates (~3-5%) are on the OBSERVED SCALE,
which is standard output from LDSC. This is directly comparable to:

1. Neale Lab UKB LDSC results for coronary atherosclerosis:
   - Observed h² = 0.031 (3.1%)
   - Liability h² = 0.161 (16.1%)
   Source: https://nealelab.github.io/UKBB_ldsc/h2_summary_I9_CORATHER.html

2. Our component CVD trait heritabilities are on the same observed scale
   (h² = 0.03-0.05), making them directly comparable to each other
   AND to our signature heritabilities.

3. Our signatures are CONTINUOUS phenotypes (integrated theta AUCs),
   so observed scale is the only applicable scale - there is no
   liability transformation for continuous traits.

4. When comparing observed-to-observed:
   - Signature 5 (cardiovascular): h² = 0.041
   - Component CVD traits: h² = 0.03-0.05
   These are comparable, demonstrating that signatures capture
   the genetic signal from component diseases.
""")
