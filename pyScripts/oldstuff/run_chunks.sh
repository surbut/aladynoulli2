#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh  # or your conda path
conda new_env_pyro2

log_dir="logs"
results_dir="results"
mkdir -p $log_dir $results_dir

for start_idx in 0 10000 20000 30000 40000 50000 60000 70000 80000 90000 \
                100000 110000 120000 130000 140000 150000 160000 170000 180000 190000; do
    
    echo "Starting chunk ${start_idx}"
    
    # Print memory status before run
    echo "Memory status before chunk ${start_idx}:"
    free -h
    
    papermill template_notebook.ipynb \
        "${results_dir}/output_chunk_${start_idx}.ipynb" \
        -p start_idx ${start_idx} \
        -p output_path "${results_dir}/model_chunk_${start_idx}.pt" \
        > "${log_dir}/chunk_${start_idx}.log" 2>&1
    
    # Print memory status after run
    echo "Memory status after chunk ${start_idx}:"
    free -h
    
    # Add a delay to ensure memory cleanup
    sleep 30
    
    echo "Completed chunk ${start_idx}"
done
