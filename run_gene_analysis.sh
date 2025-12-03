#!/bin/bash
# Wrapper script to run gene-based analysis in background

cd /Users/sarahurbut/aladynoulli2
source ~/.bash_profile
conda activate new_env_pyro2

python -u pyScripts/new_oct_revision/new_notebooks/pythonscripts/analyze_gene_based_associations.py \
    --results_dir ~/Desktop/SIG \
    --output_dir ~/Desktop/SIG/gene_based_analysis

