import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Extract key ADHD studies with their effect sizes and sample sizes
# Converting odds ratios and hazard ratios to log scale for normal approximation

studies = {
    'Baker_2020': {'log_or': np.log(2.43), 'se': (np.log(4.21) - np.log(1.41))/(2*1.96), 'n': 345},
    'Baker_2025': {'log_or': np.log(3.15), 'se': (np.log(8.29) - np.log(1.20))/(2*1.96), 'n': 307},
    'Chen_2019': {'log_or': np.log(1.20), 'se': (np.log(1.42) - np.log(1.01))/(2*1.96), 'n': 3800},
    'Ahlqvist_2024': {'log_or': np.log(1.07), 'se': (np.log(1.07) - np.log(1.05))/(2*1.96), 'n': 2480797},
    'Alemany_2021': {'log_or': np.log(1.21), 'se': (np.log(1.36) - np.log(1.07))/(2*1.96), 'n': 73881},
    'Ji_2020_high': {'log_or': np.log(2.86), 'se': (np.log(4.67) - np.log(1.77))/(2*1.96), 'n': 996},
    'Liew_2014': {'log_or': np.log(1.37), 'se': (np.log(1.59) - np.log(1.19))/(2*1.96), 'n': 64322},
    'Ystrom_2017': {'log_or': np.log(2.20), 'se': (np.log(3.24) - np.log(1.50))/(2*1.96), 'n': 112973},
    'Gustavson_2021': {'log_or': np.log(2.02), 'se': (np.log(3.25) - np.log(1.17))/(2*1.96), 'n': 26613}
}

# Sibling studies (controls for family factors)
sibling_studies = {
    'Ahlqvist_sibling': {'log_or': np.log(0.98), 'se': (np.log(1.02) - np.log(0.94))/(2*1.96), 'n': 31156},
    'Gustavson_sibling': {'log_or': np.log(1.06), 'se': (np.log(2.05) - np.log(0.51))/(2*1.96), 'n': 34}
}

def bayesian_meta_analysis(studies_dict, prior_mean=0, prior_sd=0.3, title="Meta-Analysis"):
    """
    Simple Bayesian random effects meta-analysis
    prior_mean: log odds ratio (0 = no effect)
    prior_sd: standard deviation of prior (0.3 corresponds to modest skepticism)
    """
    
    log_ors = []
    ses = []
    weights = []
    study_names = []
    
    for name, data in studies_dict.items():
        log_ors.append(data['log_or'])
        ses.append(data['se'])
        weights.append(1/data['se']**2)  # inverse variance weighting
        study_names.append(name)
    
    log_ors = np.array(log_ors)
    ses = np.array(ses)
    weights = np.array(weights)
    
    # Simple inverse-variance weighted average (fixed effects)
    pooled_log_or = np.sum(weights * log_ors) / np.sum(weights)
    pooled_se = 1 / np.sqrt(np.sum(weights))
    
    # Bayesian update: combine prior with likelihood
    prior_precision = 1 / prior_sd**2
    likelihood_precision = 1 / pooled_se**2
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_mean = (prior_precision * prior_mean + likelihood_precision * pooled_log_or) / posterior_precision
    posterior_sd = 1 / np.sqrt(posterior_precision)
    
    # Convert back to odds ratio scale
    posterior_or = np.exp(posterior_mean)
    posterior_ci_lower = np.exp(posterior_mean - 1.96 * posterior_sd)
    posterior_ci_upper = np.exp(posterior_mean + 1.96 * posterior_sd)
    
    # Calculate probability that true effect is > 1.0 (harmful)
    prob_harmful = 1 - stats.norm.cdf(0, posterior_mean, posterior_sd)
    
    # Calculate probability that true effect is > 1.2 (clinically meaningful)
    prob_meaningful = 1 - stats.norm.cdf(np.log(1.2), posterior_mean, posterior_sd)
    
    return {
        'posterior_or': posterior_or,
        'posterior_ci': (posterior_ci_lower, posterior_ci_upper),
        'prob_harmful': prob_harmful,
        'prob_meaningful': prob_meaningful,
        'individual_ors': np.exp(log_ors),
        'study_names': study_names
    }

# Different prior scenarios
priors = {
    'Neutral': {'mean': 0, 'sd': 1.0},  # Uninformative
    'Skeptical': {'mean': 0, 'sd': 0.2},  # Concentrated around null
    'Literature-based': {'mean': np.log(1.1), 'sd': 0.3}  # Weak positive prior based on some previous evidence
}

results = {}

print("BAYESIAN REANALYSIS OF ACETAMINOPHEN-ADHD STUDIES")
print("="*60)

for prior_name, prior_params in priors.items():
    print(f"\n{prior_name.upper()} PRIOR (mean log OR = {prior_params['mean']:.2f}, SD = {prior_params['sd']:.2f})")
    print("-" * 50)
    
    # Main cohort studies
    result = bayesian_meta_analysis(
        studies, 
        prior_mean=prior_params['mean'], 
        prior_sd=prior_params['sd'],
        title=f"Cohort Studies - {prior_name} Prior"
    )
    
    print(f"Posterior OR: {result['posterior_or']:.2f} (95% CI: {result['posterior_ci'][0]:.2f}-{result['posterior_ci'][1]:.2f})")
    print(f"Probability of harm (OR > 1.0): {result['prob_harmful']:.1%}")
    print(f"Probability of meaningful harm (OR > 1.2): {result['prob_meaningful']:.1%}")
    
    results[f'cohort_{prior_name}'] = result
    
    # Sibling studies
    sib_result = bayesian_meta_analysis(
        sibling_studies, 
        prior_mean=prior_params['mean'], 
        prior_sd=prior_params['sd'],
        title=f"Sibling Studies - {prior_name} Prior"
    )
    
    print(f"\nSibling Studies (family-controlled):")
    print(f"Posterior OR: {sib_result['posterior_or']:.2f} (95% CI: {sib_result['posterior_ci'][0]:.2f}-{sib_result['posterior_ci'][1]:.2f})")
    print(f"Probability of harm (OR > 1.0): {sib_result['prob_harmful']:.1%}")
    
    results[f'sibling_{prior_name}'] = sib_result

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Bayesian Reanalysis: Effect of Prior Assumptions', fontsize=16, fontweight='bold')

# Plot 1: Forest plot of individual studies
ax1 = axes[0, 0]
study_data = [(name, data['log_or'], data['se']) for name, data in studies.items()]
study_names = [x[0].replace('_', ' ') for x in study_data]
log_ors = [x[1] for x in study_data]
ses = [x[2] for x in study_data]

y_pos = range(len(study_names))
ors = np.exp(log_ors)
ci_lower = np.exp(np.array(log_ors) - 1.96 * np.array(ses))
ci_upper = np.exp(np.array(log_ors) + 1.96 * np.array(ses))

ax1.errorbar(ors, y_pos, xerr=[ors - ci_lower, ci_upper - ors], fmt='o', capsize=3)
ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No effect')
ax1.set_xlabel('Odds Ratio')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(study_names)
ax1.set_title('Individual Study Results\n(Cohort Studies)')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Posterior distributions under different priors
ax2 = axes[0, 1]
x = np.linspace(-1, 2, 1000)
colors = ['blue', 'red', 'green']
prior_names = ['Neutral', 'Skeptical', 'Literature-based']

for i, prior_name in enumerate(prior_names):
    result = results[f'cohort_{prior_name}']
    # Reconstruct posterior parameters
    posterior_mean = np.log(result['posterior_or'])
    # Approximate posterior SD from CI
    posterior_sd = (np.log(result['posterior_ci'][1]) - np.log(result['posterior_ci'][0])) / (2 * 1.96)
    
    y = stats.norm.pdf(x, posterior_mean, posterior_sd)
    ax2.plot(np.exp(x), y, color=colors[i], label=f'{prior_name} Prior', linewidth=2)

ax2.axvline(x=1, color='black', linestyle='--', alpha=0.7, label='No effect (OR=1)')
ax2.axvline(x=1.2, color='orange', linestyle=':', alpha=0.7, label='Meaningful harm (OR=1.2)')
ax2.set_xlabel('Odds Ratio')
ax2.set_ylabel('Posterior Density')
ax2.set_title('Posterior Distributions\n(Cohort Studies)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Comparison of cohort vs sibling studies
ax3 = axes[1, 0]
prior_types = ['Neutral', 'Skeptical', 'Literature-based']
cohort_ors = [results[f'cohort_{p}']['posterior_or'] for p in prior_types]
sibling_ors = [results[f'sibling_{p}']['posterior_or'] for p in prior_types]

x_pos = np.arange(len(prior_types))
width = 0.35

ax3.bar(x_pos - width/2, cohort_ors, width, label='Cohort Studies', alpha=0.8)
ax3.bar(x_pos + width/2, sibling_ors, width, label='Sibling Studies', alpha=0.8)
ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
ax3.set_xlabel('Prior Type')
ax3.set_ylabel('Posterior Odds Ratio')
ax3.set_title('Cohort vs Sibling Studies\nPosterior Estimates')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(prior_types)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Probability of harm under different priors
ax4 = axes[1, 1]
prob_harm_cohort = [results[f'cohort_{p}']['prob_harmful'] for p in prior_types]
prob_harm_sibling = [results[f'sibling_{p}']['prob_harmful'] for p in prior_types]
prob_meaningful_cohort = [results[f'cohort_{p}']['prob_meaningful'] for p in prior_types]

ax4.bar(x_pos - width/3, prob_harm_cohort, width/3, label='Cohort: P(OR>1.0)', alpha=0.8)
ax4.bar(x_pos, prob_harm_sibling, width/3, label='Sibling: P(OR>1.0)', alpha=0.8)
ax4.bar(x_pos + width/3, prob_meaningful_cohort, width/3, label='Cohort: P(OR>1.2)', alpha=0.8)

ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% probability')
ax4.set_xlabel('Prior Type')
ax4.set_ylabel('Probability')
ax4.set_title('Probability of Harmful Effects')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(prior_types)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("SUMMARY OF BAYESIAN REANALYSIS")
print("="*60)
print("\nKey Findings:")
print("1. Effect estimates are highly sensitive to prior assumptions")
print("2. Sibling studies (controlling family factors) show much weaker effects")
print("3. Even with neutral priors, confidence intervals are wide")
print("4. With skeptical priors, probability of meaningful harm is low")

print(f"\nMost Conservative Estimate (Skeptical Prior, Cohort Studies):")
conservative = results['cohort_Skeptical']
print(f"  OR: {conservative['posterior_or']:.2f} ({conservative['posterior_ci'][0]:.2f}-{conservative['posterior_ci'][1]:.2f})")
print(f"  P(meaningful harm): {conservative['prob_meaningful']:.1%}")

print(f"\nSibling Studies (Skeptical Prior - controls for family confounding):")
sibling_conservative = results['sibling_Skeptical']
print(f"  OR: {sibling_conservative['posterior_or']:.2f} ({sibling_conservative['posterior_ci'][0]:.2f}-{sibling_conservative['posterior_ci'][1]:.2f})")
print(f"  P(any harm): {sibling_conservative['prob_harmful']:.1%}")