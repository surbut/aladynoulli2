
6th October 2025

*Please ensure you delete the link to your author home page in this e-mail if you wish to forward it to your co-authors. 

Dear Professor Parmigiani

Your manuscript, "Aladynoulli: A Bayesian approach to disease progression modeling for genomic discovery and clinical prediction", has now been seen by 3 referees, whose comments are attached below. In the light of their advice we have decided that we cannot offer to publish the manuscript in Nature.

While the referees find your work of some interest and appreciate the new method, they raise concerns about the strength of the novel conclusions that can be drawn at this stage and the robustness of the data. We feel that these reservations are sufficiently important as to preclude publication of the present study in Nature, and we are accordingly closing the file.

Having said this, should future experimental data and theoretical analysis allow you to address these concerns comprehensively we would be happy to look at a revised manuscript (unless, of course, something similar has by then been accepted at Nature or appeared elsewhere). At such a time, you can request that the manuscript file be reopened for resubmission by simply sending me an email to that effect (stating manuscript number). I should stress, however, that we would be reluctant to trouble our referees again unless we thought that their comments and any editorial issues had been addressed in full, and we would understand if you prefer instead to now pursue publication of the work elsewhere.

I am sorry that we cannot respond more positively, and we hope that you find our referees' comments helpful. 

Yours sincerely 

Michelle

Dr Michelle Trenkmann
Senior Editor
Nature

orcid.org/0000-0002-3332-6344
www.nature.com


Referee expertise:

Referee #1: human genetics, disease risk

Referee #2: EHRs

Referee #3: statistical genetics, PRS


Referees' comments: 
  
  Referee #1 (Remarks to the Author):

Urbut et al present a new method for summarizing and modelling longitudinal disease trajectories in a Bayesian setting. Using electronic health records they create a novel dynamic Bayesian framework called ALADYNOULLI that identifies latent disease signatures from longitudinal health records while modeling individual-specific trajectories. They apply the modelling framework to three EHR databases UKBB, Mass Genral biobank and AllOfUs dataset. They show associations with novel genetic loci and improvements in desease predictions in several diseases over clinical risk scores. Hey conclude that their method captures the complex interplay between genetic predisposition and time-varying disease patterns.

The paper is an interesting variation in the theme using rich EHR data to achieve better predictive power over traditional clinical prediction tools. What is common in these approaches is that while the computational and statistical approaches are interesting and make better use of the rich longitudinal and multivariate nature of the EHR data, the reports fall short in their ability to answer clinically relevant questions. This manuscript follows the pattern: the method is potentially interesting but the paper is mostly descriptive by nature. It is also very long.

The key questions related to the method are:
  1) EHR data coming from one heath care provider are typically highly biased in terms of the socio-economic background of the patients. Similarly, UKBB has a well-documented bias towards healthy upper socioecinomic participants. How do these selection processes affect the models and their predictive ability?
  2) For many diseases, lifetime risk is the key measure for preventive actions or for screening stretegies. How does the model behave when lifetime risks are modelled in comparison to the exisiting clinical risk models?
  3) The authors say in several places that the models describe clinically meaningful biological processes without giving any proof of the clinical and certainly not biological meaningfulness.
4) The authors write (136-139) that “This prospective approach simulates real-world clinical scenarios where physicians must predict future risk based solely on a patient’s history to date, ensuring our performance metrics reflect true predictive capability rather than retrospective explanation.” What this sentence unfortunately describes is the ignorance of the authors on the long tradition of predictive medical research.
5) The authors are using a rather difficult to interpret concept of disease signatures and signature loading as their primary metric. It would be much more interpretable to the field if these were translated to the tradional risk/hazard metrics.
6) The model focuses a lot on modeling the temporal, age-related patterns of the incidence. For many disease these are however well known, and it is difficult to see what is the additional benefit of the model over risk models allowing for different risk along the age scale, and with time-dependent covariates potentially modifying this baseline risk. 
7) The heritability estimates on lines 294-296 seem very low. How do they compare with direct CVD and other diagnoses?
  8) It is well documented that modelling correlated phenotypes jointly elevates power to detect genetic (and other) associations. But this comes with a cost of being non-specific and losing interpretability of the associations.
9) The AUC comparisons in lines 354-55 do not see plausible. For meaningful comparisons, please compare with AUCs of well-documented clinical risk scores for ASCVD, heart falure and (Type 2?) diabetes.
10) Lines 371-74: Please explain wjat you mean by age.specific discrimination.
11) In figures, please focus on individual figures highlighting the key messages of the manuscript and leave the individual trajectory comparisons to the supplementary material



Referee #2 (Remarks to the Author):

Thank you for the opportunity to review this interesting paper. Here the authors present an innovative method that uses longitudinal biobank data to characterize disease signatures with a temporal component. The authors list five factors that differentiate their work from what has been published before: (1) Interpretability, (2) temporal modeling, (3) genetic integration (4) Unified framework and (5) individuals specific trajectories. 

I agree that the paper is significant in that it goes beyond the cross-sectional studies run in biobanks and takes advantage of the significant and growing longitudinally of EHR records following their widespread adoption over the last two decades. The study is also strengthened by the use of three independent biobanks in their analysis. 

Major concerns
However, I have a couple of major reservations related to the interpretation of the results. First, while interpretability is listed as one of the key advantages of this method, I do no think this characterization is wholly justified. Indeed, on page 11 when describing some of the signatures, the authors themselves use descriptions "likely psychiatric", "likely inflammatory", etc. I commend the use of the term "likely" as it is appropriately cautious. When you create phenotypes, temporal or not, using complex methods on large and highly complex datasets, it can be very difficult to interpret what you are left with at the end of the process. I do not mean to suggest that complex methods are not useful or necessary in some cases, but simply that is typically a tradeoff between complexity and biological/clinical interpretability, and this limitation should be discussed in the paper.

As an example of why this is an issue: I was interested in the novel genetic associations referenced in the abstract. These GWAS results for 21 signatures are provided in Extended data S7-S27. However, i was unable to assess the novelty of the associations as they were not labeled as phenotypes, but rather signatures which are a complex amalgamation of numerous phenotypes. The handful of random SNPs from the extended data that I looked up in the GWAS catalog all had robust associations with various phenotypes from previous publications. But there was no straightforward way for me to relate these prior results to the signature associations.

The second major reservation relates to the temporal component of the model. The authors claim on pg 13 to use a "leakage-free validation strategy" by evaluating model performance at 30 timepoints. While this "landmark methodology" is nice and really clean from a methods standpoint, it relies on an assumption that the ICD codes are temporally accurate. This assumption is very shaky. Indeed, we know that the first date of diagnosis for an ICD code can be much later than the actual date of diagnosis, in part due to EHR fragmentation and/or missing information. Fragmentation is almost certainly going to affect the MGB and AoU datasets, since they are based in the US. For the UKB, it appears that the analysis was conducted using hospital ICD-10 codes. If true, an individual may have a diagnosis for years and receive treatment with their primary care doctor before ascertainment occurs in the hospital. 

There may be issues with the temporality of ICD diagnosis dates beyond what is caused by fragmentation and missing information. To my knowledge, this issue has not been systematically studied, but is very important for any prediction model. If a diagnosis actually occurred much earlier than the date of the first diagnosis code, the prediction model can borrow from post-diagnosis information, a form of leakage. This will likely improve the performance of the prediction model, potentially by a whole lot. Currently, there is no way to know how much temporal leakage boosted the performance of the model. 


Minor

I have some more minor concerns with the methods section which is missing a lot of information that is needed to interpret the results and/or reproduce the method. 

Cohort Definition: The authors do not describe how they defined their cohorts. Were there restrictions on the number of visits required for inclusion? Were any age restrictions applied?
  
  Genetic Analysis: For the GWAS results, how was genetic ancestry handled? Was the analysis restricted to individuals of European ancestry, or were other ancestries included?
  
  Phenotype Handling: There is insufficient information about how phenotypic data were managed in this study. A supplemental methods section that clearly explains how ICD codes were transformed and how key time points, such as prediction time, were defined is essential for others to replicate the approach. Some of this information may be available in the GitHub code, but unfortunately i got a 404 when i tried to look at the page.

Analytical Decisions: Some analytical choices, such as restricting the analysis to 348 diseases, lack justification. Providing a rationale for these decisions would enhance the clarity of the methods.

Referee #2 (Remarks on code availability):

The link didn't work.


Referee #3 (Remarks to the Author):

Urbut et al. propose a novel and impressive dynamic Bayesian framework that can identify latent disease signatures from longitudinal EHR while modeling individual disease risk trajectories. The statistical framework seems like a hybrid of two related methods developed by the authors, i.e. Urbut et al., (medRxiv 2024) and Jiang et al. (Nat Genet 2023). The proposed model is remarkable in that it allows for both inference of individual disease risk trajectories as well as inference of disease hazards over time. It also infers latent signatures, which seem remarkably stable when compared between cohorts. The method then also provides a prediction framework that the authors show is superior to some existing clinical prediction models. It’s a well written paper that is addressing a major challenge in health data science, namely how we integrate the many different datasets in a meaningful way to predict future health outcomes and better understand disease etiology. Overall I think this work is very interesting. However, I have several comments on the manuscript that focus on potential biases, how they measure accuracy, and other minor issues. 

0. I was unable to access the code using the github link. https://github.com/surbut/aladynoulli2. This obviously needs to be fixed!

1. Although the authors do mention selection bias in passing, they don’t seem to seriously consider it’s impact on the results. There are several types of selection bias that could possibly bias the results. E.g. A) participation bias, which is well described in the UKB, and which may be partially mitigated using inverse probability weights (see e.g. Schoeler et al. Nat Hum Behav 2023). B) Survival bias or left truncation, which is related to participation bias. Other relevant biases include information bias and possibly immortal time bias (although I don’t think it would be a serious issue here). It would be nice if the authors attempted to evaluate and address these (especially participation bias), or at least discuss these and what their consequences were.. 

2. It’s not very clear to me whether and how the authors include genetic ancestry and sex in the model. Genetic ancestry is a well appreciated confounder in genetic studies, and the authors do account for this in their GWAS. However, it’s not clear to me whether they examined whether their model is impacted by genetic ancestry, which could be evaluated by comparing their prediction model against a baseline model including genetic PCs, sex, age, (sex)*(age), etc. Also, it would be nice to see if the prediction model works equally well for individuals of different genetic ancestry.

3. When predicting it seems that the authors use all information up until the censoring time. However, in practice this can be risky as sometimes diagnostic procedures lead to clear patterns. E.g. a diagnosis A can lead to more tests, that usually are followed with a related diagnosis B. Therefore having A is almost a perfect predictor of B, but this is not real in the sense that a person with A usually has B as well. The way to fix this reverse causation problem is to introduce washout windows (e.g. 1-6 months, possibly depending on outcome). 

4. It’s unclear to me whether and how well the model actually accounts for competing risks, such as death, emigration, and other strong competitors. This can also be caused by diagnostic hierarchy. What scares me are the reported hazards (e.g. figures S6-8), which seem to decrease for very old individuals, which can be interpreted as decreased risks. This looks like a competing risk issue. 
4b. It would be nice to use these to estimate cumulative incidence rates and compare with publicly available population estimates. 

5. Another potential issue are cohort effects, i.e. changes in disease prevalences over time (calendar year). Can the model capture those? E.g., depression has become more significantly more prevalent in recent years compared to 2 decades ago. How do you account for this?

6. I would also appreciate more comparison with machine learning based approaches (e.g. Forrest et al., Lancet 2023; Graf et al., medRxiv 2025; Detrois et al., Nat Genet 2025; Shmatko et al., Nature 2025), or at least more discussion of these potentially competing approaches. 

7. There are a lot of interesting results in this paper, and it would be great if these were made more easily available. E.g., it would be great if the authors made summary statistics for the signature GWASs publicly available. Also, please release trained φ and code lists (phecode versions, mapping) so others can replicate. If possible, a website making the many interesting results available in an interactive manner would be fantastic, but that’s perhaps not necessary. 


Minor comments:

8. You talk about heterogeneity, both patient and biological heterogeneity. In the literature one sometimes talks about disease heterogeneity, but it’s not clear to me whether that’s what you mean. Please clarify what you mean by heterogeneity.

9. You talk about 20 or 21 signatures. It seems that 20 is the right number. This is also probably an issue in Fig 2B.

10. In Fig 2B, what is the x axis, and why are there 20 groups there? I guess this is clustering..

11. In Fig. 4D, why not cluster the GWAS on the Y axis, as is done in many other plots. Also, why did you only consider these outcomes when estimating genetic correlations with the signatures?

12. The heritabilities for the signatures were estimated using LDSC. I think it would be interesting to also estimate it using sparse Bayesian models, such as SBayesS or LDpred, as this could provide more accurate estimates as well as estimates of polygenicity. 

13. There seems to be something weird in line 526. geq1000?

14. In Fig 4B. It’s unclear to me how the clusters were identified. Also, it is not clear whether these differences are statistically significant, after accounting to multiple testing. 

15. You use AUC as a measurement of prediction accuracy. As these are time-to-event outcomes you could consider Harrell's C, or similar. 

16. Although AUC is usually considered invariant to changes in prevalence over time it is not invariant to changes in case-control ascertainment or selection biases over time. 

17. It would be nice to see information on how computationally intensive the model training is, as a function of sample size, and # of phecodes, etc. 



Referee #3 (Remarks on code availability):

The link doesn't work, see comment 0.





Although we cannot publish your paper, it may be appropriate for another journal in the Nature Portfolio. If you wish to explore the journals and transfer your manuscript please use our manuscript transfer portal. You will not have to re-supply manuscript metadata and files, unless you wish to make modifications. For more information, please see our manuscript transfer FAQ page.


This email has been sent through the Springer Nature Manuscript Tracking System NY-610A-SN&MTS

Confidentiality Statement:

This e-mail is confidential and subject to copyright. Any unauthorised use or disclosure of its contents is prohibited. If you have received this email in error please notify our Manuscript Tracking System Helpdesk team at http://platformsupport.nature.com .

Details of the confidentiality and pre-publicity policy may be found here http://www.nature.com/authors/policies/confidentiality.html

Privacy Policy | Update Profile