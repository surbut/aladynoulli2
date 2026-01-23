# Aladynoulli: A Bayesian Voyage Through the Genome and the EHR
## 15-Minute Talk Outline for Methods in Population Genomics

### **Slide 1: Title** (30 sec)
- Title: Aladynoulli: A Bayesian Voyage Through the Genome and the EHR
- Your name, affiliations
- Brief: "A hierarchical Bayesian model integrating genetics and longitudinal EHRs"

---

### **Slide 2: The Lifetime Problem** (1.5 min)
**Key Message**: Traditional models assume static risk, but disease trajectories evolve

**Content**:
- Hazard ratios are not constant over time (show Cox PH assumption limitation)
- The "when you ask matters" problem - risk changes with age/time
- Example: Young person with high PRS vs older person - different absolute risks
- Need: Dynamic risk assessment across the life course

**Visual**: Timeline showing how risk assessment at age 40 vs 60 gives different answers

---

### **Slide 3: Walking Through the Lifetime with Bayes** (2 min)
**Key Message**: We're all Bayesians - continuously updating beliefs

**Content**:
- Central Bayes equation: $P(\Pi | \text{Data}) \propto P(\text{Data} | \Pi) \cdot P(\Pi)$
- As time progresses and diagnoses accrue:
  - Individual likelihood (EHR data) updates
  - Prior knowledge (population signatures) informs
  - Posterior beliefs about risk evolve
- This is what happens naturally: we learn from data as it arrives

**Visual**: Diagram showing prior → likelihood → posterior updating over time
- Use the "Everyone is a Bayesian" concept from your ACC slides

---

### **Slide 4: The Data Structure** (1 min)
**Key Message**: Rich longitudinal EHR data with genetic information

**Content**:
- Individuals observed over time (ages 35-72, say)
- At each time point: ICD codes (diseases), age, demographics
- Genetic data: PRS for multiple diseases
- Enrollment matrix E: accounts for when patients enter system (censoring)
- Challenge: Heterogeneous follow-up, missingness, co-occurring diseases

**Visual**: Simplified EHR timeline showing diagnoses appearing over time

---

### **Slide 5: Model Structure - The Core Idea** (2.5 min)
**Key Message**: Mixture of latent disease signatures, informed by genetics

**Content**:
**Main equation:**
$$\pi_{i,d,t} = \kappa \cdot \sum_{k=1}^{K} \theta_{i,k,t} \cdot \text{sigmoid}(\phi_{k,d,t})$$

Where:
- $\pi_{i,d,t}$ = probability of disease $d$ for individual $i$ at time $t$
- $\theta_{i,k,t}$ = individual $i$'s loading on signature $k$ at time $t$ (softmax over $\lambda$)
- $\phi_{k,d,t}$ = signature $k$'s association with disease $d$ at time $t$
- $\kappa$ = global calibration parameter

**Key insight**: This is a **mixture of probabilities**, not probability of a mixture (unlike topic models)

**Visual**: Diagram showing how individual signatures combine to give disease risk

---

### **Slide 6: Individual Trajectories - The Genetic Connection** (2 min)
**Key Message**: Genetics modify the mean trajectory via Gaussian processes

**Content**:
**Individual signature loadings:**
$$\lambda_{i,k} \sim \mathcal{GP}(r_k + \mathbf{g}_i^{\top}\Gamma_k, K_{\lambda})$$

Where:
- $\lambda_{i,k}$ = latent trajectory for individual $i$ on signature $k$ (GP)
- $r_k$ = signature-specific baseline
- $\mathbf{g}_i$ = individual's genetic factors (PRS)
- $\Gamma_k$ = genetic effects on signature affinity (TO LEARN)
- $K_{\lambda}$ = temporal covariance kernel (smoothness)

**Signature-disease associations:**
$$\phi_{k,d} \sim \mathcal{GP}(\mu_d + \psi_{k,d}, K_{\phi})$$

Where:
- $\mu_d$ = disease baseline prevalence
- $\psi_{k,d}$ = signature-disease association strength
- $K_{\phi}$ = temporal covariance (allows time-varying associations)

**Visual**: Example trajectory showing how PRS shifts mean of signature loading over time

---

### **Slide 7: The Likelihood - Survival Framework** (1.5 min)
**Key Message**: Bernoulli discrete-time survival likelihood properly handles censoring

**Content**:
**Likelihood for individual $i$, disease $d$:**
$$\ell_{i,d} = \sum_{t < E_{i,d}} \log(1 - \pi_{i,d,t}) + Y_{i,d,t}\log(\pi_{i,d,t}) + (1-Y_{i,d,t})\log(1-\pi_{i,d,t})$$

At risk: $(1-\pi)$ contributions  
Event: $\log(\pi)$ if disease occurs  
Censored: $(1-\pi)$ at enrollment time

**Key innovation**: Unlike allocation models (topic models), this directly models probability of disease occurrence - enables prediction, not just description

**Visual**: Timeline showing contributions to likelihood before, at, and after disease onset

---

### **Slide 8: What We Learn - Signatures** (1.5 min)
**Key Message**: Discover interpretable disease signatures with genetic basis

**Content**:
- Signatures = latent patterns of disease co-occurrence
- Examples: Metabolic (T2D, HTN, CAD), Inflammatory (RA, IBD), Cancer, etc.
- Each signature has characteristic timing patterns (some early, some late)
- Genetics inform which signatures individuals are predisposed to

**Visual**: Heatmap of signature-disease associations ($\phi_{k,d}$) showing which diseases cluster

---

### **Slide 9: Genetic Discovery Results** (1.5 min)
**Key Message**: Signature-based GWAS reveals biology invisible to single-disease analyses

**Content**:
- PRS for cardiovascular, metabolic, psychiatric diseases map to corresponding signatures
- Strong genetic correlations between signatures and related traits
- Signature-based GWAS uncovers loci associated with shared disease processes
- Genetics provides biological interpretation to signatures

**Visual**: 
- Bar plot showing PRS associations with signatures
- Manhattan plot showing signature-GWAS hits

---

### **Slide 10: Prediction Performance** (1.5 min)
**Key Message**: State-of-the-art prediction while maintaining interpretability

**Content**:
**Comparison to benchmarks:**
- ASCVD 10-year: Aladynoulli (AUC 0.737) vs PCE (0.683) vs QRISK3 (0.702) → **+7.9% improvement**
- ASCVD 30-year: Aladynoulli (0.708) vs PREVENT (0.650) → **+9.0% improvement**
- Outperforms Delphi-2M for 15/28 diseases
- Large improvements over Cox baseline (age+sex): Parkinson's +35%, CKD +33%, Stroke +32%

**Dynamic risk**: Model updates as patients age and develop conditions

**Visual**: 
- Bar chart comparing AUCs across models
- Example showing how risk prediction updates over time for a patient

---

### **Slide 11: The Bayesian Update in Action** (1 min)
**Key Message**: Show how posterior beliefs evolve for a real patient

**Content**:
- Pick a patient with interesting trajectory
- Show signature loadings ($\theta$) evolving over time
- Show how disease probabilities ($\pi$) update as new diagnoses arrive
- Demonstrate the "walking through the lifetime" concept

**Visual**: 
- Timeline showing patient's diagnoses
- Overlaid signature trajectory evolution
- Disease probability curves updating

---

### **Slide 12: Conclusions & Future Directions** (1 min)
**Key Message**: Bayesian framework enables both discovery and prediction

**Content**:
- **Unified framework**: Simultaneous genomic discovery and clinical prediction
- **Interpretability**: Signatures provide biological meaning
- **Dynamic**: Properly models lifetime risk evolution
- **Scalable**: Works across diseases, biobanks

**Future**:
- Integration with imaging (CAC, CT-coronary)
- AI-enhanced feature extraction (ECG, TTE signals)
- Intervention modeling (digital twins)

**Visual**: Summary diagram of the framework

---

## **Timing Breakdown** (Total: 15 min)
1. Title: 0.5 min
2. Lifetime Problem: 1.5 min
3. Bayesian Framework: 2 min
4. Data Structure: 1 min
5. Model Structure: 2.5 min
6. Genetic Connection: 2 min
7. Likelihood: 1.5 min
8. Signatures: 1.5 min
9. Genetic Results: 1.5 min
10. Prediction: 1.5 min
11. Example: 1 min
12. Conclusions: 1 min

**Buffer**: ~2 minutes for questions/discussion or deeper dive on preferred topic

---

## **Key Talking Points for Statistical Genetics Audience**

1. **Emphasize Bayesian methodology**: They'll appreciate the principled framework
2. **Highlight genetic integration**: Show how PRS/Genetics inform the model structure
3. **Discuss identifiability**: Be ready to explain how signatures are identifiable
4. **Computational aspects**: Mention optimization, inference (if time)
5. **Validation strategy**: Temporal validation, cross-biobank replication
6. **Methodological innovations**: 
   - Bernoulli survival vs allocation models
   - Mixture of probabilities vs probability of mixture
   - Gaussian processes for temporal smoothness

---

## **Potential Questions to Anticipate**

1. **How do you choose K (number of signatures)?** → Mention model selection, cross-validation, interpretability
2. **Identifiability of signatures?** → Signatures identified up to permutation; genetic effects help anchor biological meaning
3. **Scalability?** → Trained on large biobanks; mention computational considerations
4. **Comparison to other methods?** → Delphi-2M, topic models, traditional survival models
5. **How do you handle missing genetic data?** → Only include individuals with genetic data (or could impute)
6. **Priors?** → Gaussian process priors; discuss hyperparameters

---

## **Visual Style Recommendations**

- **Clean, mathematical**: This audience appreciates equations and diagrams
- **Minimal text**: Focus on concepts, not bullet points
- **Show structure**: Flow diagrams of the Bayesian updating
- **Real examples**: Include actual patient trajectories, not just abstract concepts
- **Side-by-side comparisons**: When showing results, compare to benchmarks clearly

---

## **Opening Hook Suggestions**

Consider starting with one of these:
1. **"Imagine asking a 30-year-old and a 60-year-old the same question about their cardiovascular risk..."** → Lead into dynamic risk
2. **"Bayes' theorem is often taught as a static formula, but in medicine, we're continuously updating..."** → Natural segue to lifetime modeling
3. **"Electronic health records contain rich longitudinal information, but most models treat them as static snapshots..."** → Problem statement

---

Good luck with your presentation! The Bayesian framework narrative should resonate well with this audience.




