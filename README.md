# aladynoulli
Code for Solving Aladynoulli! 

In this repo, you'll find a guide to understanding, simulating, and solving aladynoulli.

More description of this work can be found [here](https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v1).

### The Model

![image](https://github.com/user-attachments/assets/adc66f5d-5107-47a3-a089-8bd677922605)

<img width="632" alt="image" src="https://github.com/user-attachments/assets/3792a90b-9432-4aa7-add4-fccd9b8566a9" />

Additional scripts related to useful functions ('utils'), model specific functions, initialization and sampling methods (for example, elliptical) are all in the utils directory.

The code for publication is available in `pyScripts_forPublish`. 

# Useful notebooks: 

[Here](pyScripts_forPublish/aladynoulli_fit_for_understanding_and_discovery.ipynb). Fitting the model for full discovery 
[Here](pyScripts_forPublish/aladynoulli_fit_for_prediction.ipynb) 

# Core scripts

The core model code is [here](pyScripts_forPublish/clust_huge_amp.py) and implemented using external (fixed) phi for prediction: [here](clust_huge_ampfixedPhi.py) 

* Streamlit app is [here](pyScripts_forPublish/patient_timeline_app)
* submission scripts for AWS with fixed phi over 30 years are [here](pyScripts_forPublish/submit_script_aws_fixedph_40_70.py)
* submission scripts for AWS basic are [here](pyScripts_forPublish/submit_script.py)

Please contact me at surbut@mgh.harvard.edu for qs!
