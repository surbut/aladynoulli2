import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
import random
import glob
import traceback

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any 

def analyze_genetic_data_by_cluster(disease_idx, batch_size=10000, n_batches=10, n_clusters=3, prs_names_file=None, heatmap_output_path=None):
    """
    Analyze genetic/demographic data (G matrix/X data) across patient clusters
    for a specific disease
    
    Parameters:
    - disease_idx: Index of the disease to analyze
    - batch_size: Number of patients per batch
    - n_batches: Number of batches to process
    - n_clusters: Number of clusters to create
    - prs_names_file: Path to CSV file containing PRS names
    - heatmap_output_path: Path to save the heatmap PDF. If None, heatmap is shown but not saved.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    
    print(f"Starting genetic data analysis for disease {disease_idx}...")
    
    # Load PRS names if file is provided
    if prs_names_file:
        try:
            prs_df = pd.read_csv(prs_names_file)
            print(f"Loaded {len(prs_df)} PRS names from {prs_names_file}")
            # Create a dictionary to map indices to names
            prs_names_dict = {i: name for i, name in enumerate(prs_df.iloc[:, 0])} if len(prs_df.columns) > 0 else {}
        except Exception as e:
            print(f"Error loading PRS names: {str(e)}")
            prs_names_dict = {}
    else:
        prs_names_dict = {}
    
    # Load batch 1 model to get disease names and signature information
    model_path = '/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt'
    print("\nLoading batch 1 model for reference...")
    
    try:
        model1 = torch.load(model_path)
        disease_names = model1['disease_names'][0].tolist()
        K_total = model1['model_state_dict']['lambda_'].shape[1]  # Number of signatures
        time_points = model1['Y'].shape[2]  # Number of time points
        
        # Get X dimensions
        if 'G' in model1:
            X_dim = model1['G'].shape[1]
            # Use PRS names if available, otherwise use generic names
            genetic_factor_names = [prs_names_dict.get(i, f"G_{i}") for i in range(X_dim)]
        else:
            print("Warning: G matrix not found in first batch. Will try to determine dimensions from later batches.")
            genetic_factor_names = None
        
        disease_name = disease_names[disease_idx] if disease_idx < len(disease_names) else f"Disease {disease_idx}"
        print(f"Analyzing {disease_name}")
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Collect patients with this disease across all batches
    all_patients = []  # Will store (batch_idx, patient_idx) tuples
    all_features = []  # Will store signature proportions
    all_genetic_data = []  # Will store genetic/demographic data (X/G matrix)
    
    # Process each batch
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        
        model_path = f'/Users/sarahurbut/Dropbox/resultshighamp/results/output_{start_idx}_{end_idx}/model.pt'
        print(f"\nProcessing batch {batch+1}/{n_batches} (patients {start_idx}-{end_idx})")
        
        try:
            model = torch.load(model_path)
            lambda_values = model['model_state_dict']['lambda_']
            Y_batch = model['Y'].numpy()
            
            # Get genetic data (X/G matrix) for this batch
            if 'G' in model:
                X_batch = model['G'].numpy()  # Patient genetic/demographic data
                
                # If we don't have factor names yet, create them
                if genetic_factor_names is None:
                    X_dim = X_batch.shape[1]
                    genetic_factor_names = [prs_names_dict.get(i, f"G_{i}") for i in range(X_dim)]
            else:
                print(f"Warning: G matrix not found in batch {batch}")
                continue
            
            # Find patients with this disease
            for i in range(Y_batch.shape[0]):
                if np.any(Y_batch[i, disease_idx]):
                    # Get signature proportions
                    theta = torch.softmax(lambda_values[i], dim=0).detach().numpy()
                    mean_props = theta.mean(axis=1)
                    
                    # Store patient data
                    all_patients.append((batch, i))
                    all_features.append(mean_props)
                    all_genetic_data.append(X_batch[i])
            
            print(f"Found {len(all_patients) - (0 if batch == 0 else sum(1 for p in all_patients if p[0] < batch))} patients in batch {batch+1}")
            print("genetic_factor_names:", genetic_factor_names)
            print("First row of all_genetic_data:", all_genetic_data[0])
        except FileNotFoundError:
            print(f"Warning: Could not find model file for batch {batch}")
            break  # Assume we've reached the end of batches
        except Exception as e:
            print(f"Error processing batch {batch}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_patients) == 0:
        print(f"No patients found with disease {disease_idx}")
        return None
    
    print(f"\nTotal patients with {disease_name}: {len(all_patients)}")


    # --- Add debug print here ---
    bc_idx = None
    if 'BC' in genetic_factor_names:
        bc_idx = genetic_factor_names.index('BC')
        bc_prs = [x[bc_idx] for x in all_genetic_data]
        print(f"\n[DEBUG] Number of patients with {disease_name}: {len(all_patients)}")
        print(f"[DEBUG] Mean BC PRS for patients with {disease_name}: {np.mean(bc_prs):.4f}")
        print(f"[DEBUG] Min/Max BC PRS: {np.min(bc_prs):.4f} / {np.max(bc_prs):.4f}")
    else:
        print("[DEBUG] 'BC' not found in genetic_factor_names")

    print("PRS name and value for first patient:")
    for i, name in enumerate(genetic_factor_names):
        print(f"{name}: {all_genetic_data[0][i]}")
    
    # Convert to numpy arrays
    all_features = np.array(all_features)
    all_genetic_data = np.array(all_genetic_data)
    
    # Perform clustering based on signature profiles
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    patient_clusters = kmeans.fit_predict(all_features)
    bc_idx = genetic_factor_names.index('BC')
    overall_mean = np.mean(all_genetic_data[:, bc_idx])
    print(f"Overall mean BC PRS: {overall_mean:.4f}")
    for cluster in range(n_clusters):
        mask = patient_clusters == cluster
        mean_bc = np.mean(all_genetic_data[mask, bc_idx])
        print(f"Cluster {cluster}: mean BC PRS = {mean_bc:.4f} (n={np.sum(mask)})")
    means = []
    sizes = []
    for cluster in range(n_clusters):
        mask = patient_clusters == cluster
        mean_bc = np.mean(all_genetic_data[mask, bc_idx])
        size = np.sum(mask)
        means.append(mean_bc)
        sizes.append(size)
    weighted_mean = np.average(means, weights=sizes)
    print(f"Weighted mean of cluster means: {weighted_mean:.4f}")
    # Analyze genetic data by cluster
    genetic_by_cluster = {}
    for cluster in range(n_clusters):
        cluster_mask = patient_clusters == cluster
        genetic_by_cluster[cluster] = all_genetic_data[cluster_mask]

    # Calculate mean genetic values for each cluster
    mean_genetic = {cluster: np.mean(values, axis=0) for cluster, values in genetic_by_cluster.items()}

    print("\n[DEBUG] Per-cluster means for BC (in memory):")
    bc_idx = genetic_factor_names.index('BC')
    for c in range(n_clusters):
        print(f"Cluster {c}: {mean_genetic[c][bc_idx]:.6f}")
    # Create dataframe for visualization
    genetic_df = pd.DataFrame({
        'Factor': np.repeat(genetic_factor_names, n_clusters),
        'Cluster': np.tile(np.arange(n_clusters), len(genetic_factor_names)),
        'Mean_Value': np.concatenate([mean_genetic[c] for c in range(n_clusters)]),
        #'P_Value': p_values.flatten(),
        #'P_Value_Corrected': p_values_corrected.flatten(),
        #'Effect_Size': effect_sizes.flatten()
    })

    # Add significance indicator
    #genetic_df['Significant'] = genetic_df['P_Value_Corrected'] < 0.05

    # Save cluster scores by PRS to CSV
    # Create a pivot table to organize data by PRS and cluster
    pivot_table = genetic_df.pivot(
        index='Factor',
        columns='Cluster',
        values=['Mean_Value']
    )

    # Create a new DataFrame with properly named columns
    cluster_scores = pd.DataFrame(index=genetic_factor_names)

    # Manually create the columns with the desired naming format
    for c in range(n_clusters):
        for metric in ['Mean_Value']:#$, 'Effect_Size', 'P_Value_Corrected', 'Significant']:
            column_name = f"{metric}_Cluster{c}"
            cluster_scores[column_name] = pivot_table.loc[:, (metric, c)]

    # Add cluster sizes
    for c in range(n_clusters):
        cluster_scores[f'Cluster_Size_{c}'] = np.sum(patient_clusters == c)

    # Ensure all factors are included, in the correct order
    cluster_scores = cluster_scores.reindex(index=genetic_factor_names)

    # Set index name so reset_index creates 'Factor' column
    cluster_scores.index.name = 'Factor'
    cluster_scores = cluster_scores.reset_index()
    output_csv_path = f'cluster_scores_disease_{disease_idx}.csv'
    cluster_scores.to_csv(output_csv_path, index=False)
    print(f"\nSaved cluster scores to {output_csv_path}")

    # Debug check
    df = pd.read_csv(output_csv_path)
    print("\n[DEBUG] BC row from CSV:")
    print(df[df['Factor'] == 'BC'])

    









def analyze_genetic_data_by_population(batch_size=10000, n_batches=10, n_clusters=3, prs_names_file=None, heatmap_output_path=None):
    """
    Analyze genetic/demographic data (G matrix/X data) across patient clusters
    for a specific disease
    
    Parameters:
    - disease_idx: Index of the disease to analyze
    - batch_size: Number of patients per batch
    - n_batches: Number of batches to process
    - n_clusters: Number of clusters to create
    - prs_names_file: Path to CSV file containing PRS names
    - heatmap_output_path: Path to save the heatmap PDF. If None, heatmap is shown but not saved.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    
    print(f"Starting genetic data analysis for poulation")
    
    # Load PRS names if file is provided
    if prs_names_file:
        try:
            prs_df = pd.read_csv(prs_names_file)
            print(f"Loaded {len(prs_df)} PRS names from {prs_names_file}")
            # Create a dictionary to map indices to names
            prs_names_dict = {i: name for i, name in enumerate(prs_df.iloc[:, 0])} if len(prs_df.columns) > 0 else {}
        except Exception as e:
            print(f"Error loading PRS names: {str(e)}")
            prs_names_dict = {}
    else:
        prs_names_dict = {}
    
    # Load batch 1 model to get disease names and signature information
    model_path = '/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt'
    print("\nLoading batch 1 model for reference...")
    
    try:
        model1 = torch.load(model_path)
        disease_names = model1['disease_names'][0].tolist()
        K_total = model1['model_state_dict']['lambda_'].shape[1]  # Number of signatures
        time_points = model1['Y'].shape[2]  # Number of time points
        
        # Get X dimensions
        if 'G' in model1:
            X_dim = model1['G'].shape[1]
            # Use PRS names if available, otherwise use generic names
            genetic_factor_names = [prs_names_dict.get(i, f"G_{i}") for i in range(X_dim)]
        else:
            print("Warning: G matrix not found in first batch. Will try to determine dimensions from later batches.")
            genetic_factor_names = None
        
        disease_name = "Population"
        print(f"Analyzing {disease_name}")
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Collect patients with this disease across all batches
    all_patients = []  # Will store (batch_idx, patient_idx) tuples
    all_features = []  # Will store signature proportions
    all_genetic_data = []  # Will store genetic/demographic data (X/G matrix)
    
    # Process each batch
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        
        model_path = f'/Users/sarahurbut/Dropbox/resultshighamp/results/output_{start_idx}_{end_idx}/model.pt'
        print(f"\nProcessing batch {batch+1}/{n_batches} (patients {start_idx}-{end_idx})")
        
        try:
            model = torch.load(model_path)
            lambda_values = model['model_state_dict']['lambda_']
            Y_batch = model['Y'].numpy()
            
            # Get genetic data (X/G matrix) for this batch
            if 'G' in model:
                X_batch = model['G'].numpy()  # Patient genetic/demographic data
                
                # If we don't have factor names yet, create them
                if genetic_factor_names is None:
                    X_dim = X_batch.shape[1]
                    genetic_factor_names = [prs_names_dict.get(i, f"G_{i}") for i in range(X_dim)]
            else:
                print(f"Warning: G matrix not found in batch {batch}")
                continue
            
            # Find patients with this disease
            for i in range(Y_batch.shape[0]):
                #if np.any(Y_batch[i, disease_idx]):
                if True:
                    # Get signature proportions
                    theta = torch.softmax(lambda_values[i], dim=0).detach().numpy()
                    mean_props = theta.mean(axis=1)
                    
                    # Store patient data
                    all_patients.append((batch, i))
                    all_features.append(mean_props)
                    all_genetic_data.append(X_batch[i])
            
            print(f"Found {len(all_patients) - (0 if batch == 0 else sum(1 for p in all_patients if p[0] < batch))} patients in batch {batch+1}")
            
        except FileNotFoundError:
            print(f"Warning: Could not find model file for batch {batch}")
            break  # Assume we've reached the end of batches
        except Exception as e:
            print(f"Error processing batch {batch}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
  
    
    print(f"\nTotal patients with {disease_name}: {len(all_patients)}")
    
    # Convert to numpy arrays
    all_features = np.array(all_features)
    all_genetic_data = np.array(all_genetic_data)
    
    # Perform clustering based on signature profiles
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    patient_clusters = kmeans.fit_predict(all_features)
    
    # Analyze genetic data by cluster
    genetic_by_cluster = {}
    for cluster in range(n_clusters):
        cluster_mask = patient_clusters == cluster
        genetic_by_cluster[cluster] = all_genetic_data[cluster_mask]
    
    # Calculate mean genetic values for each cluster
    mean_genetic = {cluster: np.mean(values, axis=0) for cluster, values in genetic_by_cluster.items()}
    
    # Perform statistical testing
    p_values = np.ones((len(genetic_factor_names), n_clusters))
    effect_sizes = np.zeros((len(genetic_factor_names), n_clusters))
    
    for factor_idx in range(len(genetic_factor_names)):
        for cluster in range(n_clusters):
            # Compare this cluster to all others for this factor
            cluster_values = all_genetic_data[patient_clusters == cluster, factor_idx]
            other_values = all_genetic_data[patient_clusters != cluster, factor_idx]
            
            # t-test
            t_stat, p_val = stats.ttest_ind(cluster_values, other_values)
            p_values[factor_idx, cluster] = p_val
            
            # Effect size (Cohen's d)
            mean_diff = np.mean(cluster_values) - np.mean(other_values)
            pooled_std = np.sqrt(((len(cluster_values) - 1) * np.var(cluster_values) + 
                                 (len(other_values) - 1) * np.var(other_values)) / 
                                (len(cluster_values) + len(other_values) - 2))
            effect_sizes[factor_idx, cluster] = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Apply multiple testing correction
    from statsmodels.stats.multitest import multipletests
    _, p_values_corrected, _, _ = multipletests(p_values.flatten(), method='fdr_bh')
    p_values_corrected = p_values_corrected.reshape(p_values.shape)
    
    # Create dataframe for visualization
    genetic_df = pd.DataFrame({
        'Factor': np.repeat(genetic_factor_names, n_clusters),
        'Cluster': np.tile(np.arange(n_clusters), len(genetic_factor_names)),
        'Mean_Value': np.concatenate([mean_genetic[c] for c in range(n_clusters)]),
        'P_Value': p_values.flatten(),
        'P_Value_Corrected': p_values_corrected.flatten(),
        'Effect_Size': effect_sizes.flatten()
    })
    
    # Add significance indicator
    genetic_df['Significant'] = genetic_df['P_Value_Corrected'] < 0.05
    
    # Save cluster scores by PRS to CSV
    # Create a pivot table to organize data by PRS and cluster
    cluster_scores = genetic_df.pivot(
        index='Factor',
        columns='Cluster',
        values=['Mean_Value', 'Effect_Size', 'P_Value_Corrected', 'Significant']
    )
    
    # Flatten the multi-level columns
    cluster_scores.columns = [f'{col[0]}_Cluster{col[1]}' for col in cluster_scores.columns]
    
    # Add cluster sizes
    cluster_sizes = {f'Cluster_Size_{c}': np.sum(patient_clusters == c) for c in range(n_clusters)}
    for col, size in cluster_sizes.items():
        cluster_scores[col] = size
    
    # Save to CSV
    output_csv_path = f'cluster_scores_disease_population.csv'
    cluster_scores = cluster_scores.reset_index()
    cluster_scores.to_csv(output_csv_path)
    print(f"\nSaved cluster scores to {output_csv_path}")
    
    # Create heatmap of mean genetic values
    plt.figure(figsize=(16, 20))  # Increased figure size
    
    # Use all factors, in PRS CSV order
    top_factors = np.arange(len(genetic_factor_names))
    
    # Before creating the heatmap, get the data from the cluster_scores DataFrame
    heatmap_data = np.zeros((len(top_factors), n_clusters))
    p_value_annotations = np.empty((len(top_factors), n_clusters), dtype=object)
    
    # Calculate Bonferroni threshold
    bonferroni_threshold = 0.05 / (len(genetic_factor_names) * n_clusters)
    
    for i, factor in enumerate(genetic_factor_names):
        for c in range(n_clusters):
            mean_val = cluster_scores[f'Mean_Value_Cluster{c}'][factor]
            effect_size = cluster_scores[f'Effect_Size_Cluster{c}'][factor]
            p_val = cluster_scores[f'P_Value_Corrected_Cluster{c}'][factor]
            
            heatmap_data[i, c] = mean_val
            
            # Format annotation with mean value and effect size if significant
            mean_str = f"{mean_val:.3f}"
            if p_val < bonferroni_threshold:
                d_str = f"\nd={effect_size:.2f}"
                stars = "***"
                mean_str = f"\\mathbf{{{mean_str}}}"
            else:
                d_str = ""
                stars = ""
            p_value_annotations[i, c] = f"${mean_str}${d_str}{stars}"
    
    # Add these debug prints right before creating the heatmap
    print("\nVerifying CSV and heatmap values match:")
    print("\nFirst few values from cluster_scores (CSV):")
    for factor in genetic_factor_names[:3]:  # Show first 3 factors
        print(f"\n{factor}:")
        for c in range(n_clusters):
            mean_val = cluster_scores[f'Mean_Value_Cluster{c}'][factor]
            print(f"Cluster {c}: {mean_val:.3f}")

    print("\nFirst few values from heatmap_data:")
    for i in range(3):  # Show first 3 factors
        print(f"\n{genetic_factor_names[i]}:")
        for c in range(n_clusters):
            print(f"Cluster {c}: {heatmap_data[i,c]:.3f}")
    
    # Create heatmap with custom annotations
    ax = sns.heatmap(
        heatmap_data,
        annot=p_value_annotations,  # Use our custom annotations
        fmt="",
        xticklabels=[f"Cluster {i}\n(n={np.sum(patient_clusters == i)})" for i in range(n_clusters)],
        yticklabels=genetic_factor_names,
        cmap="RdBu_r",
        center=0,
        linewidths=1.0,
        linecolor='black',
        cbar_kws={'label': 'Mean Value'},
        annot_kws={'size': 10}
    )
    
    # Rotate y-axis labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid lines
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    
    plt.title(f"Top Genetic Factors by Cluster for {disease_name}\n"
              f"*** p < 0.05 (Bonferroni)", 
              pad=20,
              size=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save heatmap if output path is provided
    if heatmap_output_path:
        plt.savefig(heatmap_output_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved heatmap to {heatmap_output_path}")
        plt.close()
    else:
        plt.show()
    
    # Create bar plot for top significant factors
    plt.figure(figsize=(14, 12))
    
    # Find top significant factors using cluster_scores
    significant_factors = []
    for factor in genetic_factor_names:
        # Check if any cluster is significant for this factor
        is_significant = any(cluster_scores[f'P_Value_Corrected_Cluster{c}'][factor] < 0.05 
                            for c in range(n_clusters))
        if is_significant:
            significant_factors.append(factor)

    if len(significant_factors) > 0:
        # Sort by maximum absolute effect size
        max_effect_sizes = []
        for factor in significant_factors:
            effect_sizes_factor = [abs(cluster_scores[f'Effect_Size_Cluster{c}'][factor]) 
                                 for c in range(n_clusters)]
            max_effect_sizes.append(max(effect_sizes_factor))
        
        # Get top factors
        top_sig_factors = [f for _, f in sorted(zip(max_effect_sizes, significant_factors), 
                                              reverse=True)][:10]
        
        # Create subplot for each top factor
        for i, factor in enumerate(top_sig_factors):
            plt.subplot(5, 2, i+1)
            
            # Get values and errors for this factor
            factor_values = [cluster_scores[f'Mean_Value_Cluster{c}'][factor] 
                            for c in range(n_clusters)]
            
            # Calculate standard errors using original data
            cluster_sizes = [np.sum(patient_clusters == c) for c in range(n_clusters)]
            std_errors = [np.std(all_genetic_data[patient_clusters == c, 
                                                genetic_factor_names.index(factor)]) / 
                         np.sqrt(cluster_sizes[c]) for c in range(n_clusters)]
            
            # Create bar plot
            bars = plt.bar(range(n_clusters), factor_values, 
                         yerr=std_errors,
                         capsize=5,
                         color=['red' if cluster_scores[f'P_Value_Corrected_Cluster{c}'][factor] < 0.05 
                               else 'gray' for c in range(n_clusters)])
            
            # Add significance stars and p-values
            for c in range(n_clusters):
                y_pos = factor_values[c] + std_errors[c] + 0.1
                if p_values_corrected[genetic_factor_names.index(factor), c] < 0.001:
                    plt.text(c, y_pos, '***', ha='center', fontsize=10)
                elif p_values_corrected[genetic_factor_names.index(factor), c] < 0.01:
                    plt.text(c, y_pos, '**', ha='center', fontsize=10)
                elif p_values_corrected[genetic_factor_names.index(factor), c] < 0.05:
                    plt.text(c, y_pos, '*', ha='center', fontsize=10)
                
                # Add effect size below the bar
                plt.text(c, -0.1, f"d={effect_sizes[genetic_factor_names.index(factor), c]:.2f}", 
                        ha='center', va='top', fontsize=8)
            
            plt.title(f"{factor}", pad=10)
            plt.xticks(range(n_clusters), [f"Cluster {i}\n(n={cluster_sizes[i]})" for i in range(n_clusters)])
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Adjust y-axis to accommodate error bars and annotations
            y_min = min(0, min(factor_values) - max(std_errors) - 0.2)
            y_max = max(factor_values) + max(std_errors) + 0.3
            plt.ylim(y_min, y_max)
        
        plt.suptitle(f"Top Significant Genetic Factors for {disease_name} Clusters\n"
                    f"*** p < 0.001, ** p < 0.01, * p < 0.05", 
                    y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
    else:
        plt.text(0.5, 0.5, "No significant factors found after correction", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    # Return results for further analysis
    return {
        'genetic_by_cluster': genetic_by_cluster,
        'mean_genetic': mean_genetic,
        'p_values': p_values,
        'p_values_corrected': p_values_corrected,
        'effect_sizes': effect_sizes,
        'genetic_factor_names': genetic_factor_names,
        'patient_clusters': patient_clusters,
        'genetic_df': genetic_df,
        'all_patients': all_patients,
        'cluster_scores': cluster_scores  # Add the cluster scores to the return dict
    }


def plot_gamma_heatmap(model_path, output_path=None, figsize=(12, 10), 
                      cmap='RdBu_r', vmin=None, vmax=None, 
                      annotate=False, gene_names=None):
    """
    Create a heatmap visualization of the gamma parameter (genetic influences on signatures).
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file (.pt)
    output_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap to use for the heatmap
    vmin, vmax : float, optional
        Min and max values for color scaling. If None, automatically determined.
    annotate : bool, optional
        Whether to annotate each cell with its value
    gene_names : list, optional
        Names of genetic components (if available)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the heatmap
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torch
    import pandas as pd
    
    # Load the model data
    print(f"Loading model from {model_path}")
    try:
        model_data = torch.load(model_path, map_location='cpu')
        
        # Extract gamma
        if 'model_state_dict' in model_data and 'gamma' in model_data['model_state_dict']:
            gamma = model_data['model_state_dict']['gamma'].detach().cpu().numpy()
            print(f"Successfully loaded gamma with shape {gamma.shape}")
        else:
            # Try other possible locations
            if 'gamma' in model_data:
                gamma = model_data['gamma']
                if torch.is_tensor(gamma):
                    gamma = gamma.detach().cpu().numpy()
            else:
                raise KeyError("Could not find gamma parameter in the model file")
        
        # Get dimensions
        P, K = gamma.shape
        print(f"Gamma dimensions: {P} genetic components × {K} signatures")
        
        # Create row and column labels
        if gene_names is not None and len(gene_names) == P:
            row_labels = gene_names
        else:
            row_labels = [f"Genetic PC {i+1}" for i in range(P)]
        
        col_labels = [f"Signature {k}" for k in range(K)]
        
        # Create a pandas DataFrame for better visualization
        gamma_df = pd.DataFrame(gamma, index=row_labels, columns=col_labels)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the heatmap
        heatmap = sns.heatmap(gamma_df, cmap=cmap, vmin=vmin, vmax=vmax,
                             center=0, annot=annotate, fmt=".2f" if annotate else None,
                             linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Genetic Effect Size"})
        
        # Set title and labels
        plt.title("Genetic Influence on Signatures (Gamma Matrix)", fontsize=16, pad=20)
        plt.ylabel("Genetic Components", fontsize=14)
        plt.xlabel("Signatures", fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {output_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error creating gamma heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_gamma_heatmap_with_stats(base_dir, batch_size=10000, n_batches=10, output_path=None, 
                              figsize=(20, 16), cmap='RdBu_r', prs_names_file=None):
    """
    Create a heatmap of gamma (genetic influences on signatures) using data from all batches,
    with significance testing and PRS names.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the batch folders
    batch_size : int
        Size of each batch
    n_batches : int
        Number of batches to process
    output_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height) in inches
    cmap : str
        Colormap for the heatmap
    prs_names_file : str, optional
        Path to CSV file containing PRS names
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats
    
    # Load PRS names if file is provided
    if prs_names_file:
        try:
            prs_df = pd.read_csv(prs_names_file)
            print(f"Loaded {len(prs_df)} PRS names")
            prs_names = prs_df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error loading PRS names: {e}")
            prs_names = None
    else:
        prs_names = None
    
    # Storage for gamma values from each batch
    all_gammas = []
    
    # Process each batch
    print("Loading gamma values from all batches...")
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        model_path = f'{base_dir}/output_{start_idx}_{end_idx}/model.pt'
        
        try:
            model_data = torch.load(model_path, map_location='cpu')
            
            # Extract gamma
            if 'model_state_dict' in model_data and 'gamma' in model_data['model_state_dict']:
                gamma = model_data['model_state_dict']['gamma'].detach().cpu().numpy()
            elif 'gamma' in model_data:
                gamma = model_data['gamma']
                if torch.is_tensor(gamma):
                    gamma = gamma.detach().cpu().numpy()
            else:
                print(f"Warning: Could not find gamma in batch {batch}")
                continue
                
            all_gammas.append(gamma)
            print(f"Loaded gamma from batch {batch+1} with shape {gamma.shape}")
            
        except FileNotFoundError:
            print(f"Warning: Could not find model file for batch {batch}")
            break
        except Exception as e:
            print(f"Error processing batch {batch}: {e}")
            continue
    
    if not all_gammas:
        print("No gamma values found!")
        return None
    
    # Stack all gammas
    gamma_stack = np.stack(all_gammas)
    
    # Calculate mean and standard error
    gamma_mean = np.mean(gamma_stack, axis=0)
    gamma_sem = np.std(gamma_stack, axis=0) / np.sqrt(len(all_gammas))
    
    # Perform t-tests against null hypothesis (gamma = 0)
    t_stats = gamma_mean / gamma_sem
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(all_gammas)-1))
    
    # Bonferroni correction
    bonferroni_threshold = 0.05 / (36 * 21)  # Adjust based on total number of tests
    print(f"Using Bonferroni threshold: {bonferroni_threshold:.2e}")
    
    # Get dimensions and create labels
    P, K = gamma_mean.shape
    
    # Use provided PRS names or create generic ones
    if prs_names is not None and len(prs_names) == P:
        row_labels = prs_names
    else:
        row_labels = [f"PRS_{i+1}" for i in range(P)]
    
    col_labels = [f"Signature {k}" for k in range(K)]
    
    # Create annotations (just stars for significance)
    annotations = np.empty_like(gamma_mean, dtype=object)
    for i in range(P):
        for j in range(K):
            p_val = p_values[i, j]
            if p_val < bonferroni_threshold:
                annotations[i, j] = '*'
            else:
                annotations[i, j] = ''
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(gamma_mean, 
                    annot=annotations,
                    fmt='',
                    cmap=cmap,
                    center=0,
                    linewidths=0.5,
                    linecolor='gray',
                    xticklabels=col_labels,
                    yticklabels=row_labels,
                    cbar_kws={'label': 'Genetic Effect Size'},
                    annot_kws={'size': 12, 'weight': 'bold'})
    
    # Customize appearance
    plt.title("PRS-State Associations with Significance Annotations", 
             pad=20, size=14)
    
    # Rotate y-axis labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved heatmap to {output_path}")
        plt.close()
    else:
        plt.show()
    
    # After calculating gamma_mean, gamma_sem, p_values, and significant
    rows = []
    for i, prs in enumerate(row_labels):
        for j, sig in enumerate(col_labels):
            rows.append({
                "prs": prs,
                "signature": sig,
                "effect_mean": gamma_mean[i, j],
                "effect_se": gamma_sem[i, j],
                "p_value": p_values[i, j],
                "significant": p_values[i, j] < bonferroni_threshold
            })

    df = pd.DataFrame(rows)
    df.to_csv("gamma_associations.csv", index=False)
    print("Saved CSV to gamma_associations.csv")
    
    # Return the computed statistics
    return {
        'gamma_mean': gamma_mean,
        'gamma_sem': gamma_sem,
        'p_values': p_values,
        'significant': p_values < bonferroni_threshold
    }

"""
The batches are our independent samples/replicates that we're using to estimate the true gamma values and their uncertainty. They're not additional tests being performed.
The Bonferroni correction should account for the number of actual hypothesis tests we're doing:
We have 36 PRS scores
We have 21 signatures
So we're doing 36 * 21 = 756 tests total
Hence our current threshold of 0.05/(3621) is correct
Adding batches actually makes our estimates more precise (reduces standard error) because:
More batches = more independent samples
SE = standard deviation / sqrt(number of batches)
This is a good thing - it gives us more statistical power to detect true effects
The current Bonferroni threshold (0.05/(3621)) is appropriately controlling the family-wise error rate across all PRS-signature tests, regardless of how many batches we use to estimate each relationship.
"""

def analyze_signature_snp_associations(
    genotype_file_default="/Users/sarahurbut/Dropbox/genotype_raw/genotype_dosage_20250416.raw",
    genotype_file_sig5="/Users/sarahurbut/Dropbox/genotype_raw/genotype_dosage_20250415.raw",
    covariate_file="/Users/sarahurbut/Dropbox/for_regenie/ukbb_covariates_400k.txt",
    snp_list_dir="/Users/sarahurbut/Dropbox/snp_lists",
    phenotype_dir="/Users/sarahurbut/Dropbox/for_regenie/case_control_phenotypes",
    sig_stats_dir="/Users/sarahurbut/Dropbox/result326/10_loci",
    signatures_to_analyze=[4],  # List of signature numbers to analyze
    output_dir=None
):
    """
    Analyze SNPs that are significant for signatures but not individual phenotypes.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests
    import statsmodels.api as sm
    
    results_dict = {}
    
    for sig in signatures_to_analyze:
        print(f"\nAnalyzing Signature {sig}")
        
        # Read signature summary statistics
        sig_stats_file = f"{sig_stats_dir}/SIG{sig}_AUC_ukb_eur_regenie_af1.sig.lead.sumstats.txt"
        try:
            # First read a few lines to check the structure
            with open(sig_stats_file, 'r') as f:
                first_lines = [next(f) for _ in range(5)]
                print("First few lines of the file:")
                for line in first_lines:
                    print(line.strip())
            
            # Read the file with headers
            sig_stats = pd.read_csv(sig_stats_file, sep='\t', header=0)
            
            # Print column names for debugging
            print("\nAvailable columns:", sig_stats.columns.tolist())
            
            # Create mapping of SNP ID to signature z-stat
            # First, let's print the first few rows to verify the data
            print("\nFirst few rows of signature statistics:")
            print(sig_stats.head())
            
            # Create the mapping using the correct column names
            sig_z_stats = dict(zip(sig_stats.iloc[:, 12], sig_stats.iloc[:, 16]))  # Still using indices for now until we confirm column names
            print(f"\nLoaded signature statistics for {len(sig_z_stats)} SNPs")
            print("Example SNPs and their Z-stats:")
            for snp, z_stat in list(sig_z_stats.items())[:5]:
                print(f"{snp}: {z_stat:.2f}")
        except Exception as e:
            print(f"Error reading signature statistics: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Select appropriate genotype file based on signature
        genotype_file = genotype_file_sig5 if sig == 5 else genotype_file_default
        print(f"Using genotype file: {genotype_file}")
        
        # Read significant SNPs for this signature
        snp_file = f"{snp_list_dir}/snp_list_sig{sig}.txt"
        try:
            # Read the file with whitespace separator and two columns
            snp_df = pd.read_csv(snp_file, sep='\s+', header=None, names=['idx', 'rsid'])
            sig_snps = snp_df['rsid'].tolist()
            print(f"Found {len(sig_snps)} significant SNPs for signature {sig}")
            print("First few SNPs:", sig_snps[:5])
        except Exception as e:
            print(f"Error reading SNP file for signature {sig}: {e}")
            continue
            
        # Read case-control status
        pheno_file = f"{phenotype_dir}/case_control_sig{sig}.tsv"
        try:
            phenotypes = pd.read_csv(pheno_file, sep='\t')
            print(f"Loaded phenotypes with shape {phenotypes.shape}")
        except Exception as e:
            print(f"Error reading phenotype file: {e}")
            continue
            
        # Read genotypes (only for significant SNPs)
        try:
            genotypes = pd.read_csv(genotype_file, sep='\t')
            
            # Create mapping from column names to base SNP IDs
            geno_cols = genotypes.columns
            col_to_snp = {}
            for col in geno_cols:
                # Skip non-SNP columns
                if col in ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']:
                    continue
                    
                # Extract base SNP ID by removing allele suffix
                base_snp = col.split('_')[0]  # Split on underscore for rsIDs
                col_to_snp[col] = base_snp
            
            # Find matching columns for our SNPs of interest
            snp_to_col = {}
            for snp in sig_snps:
                matching_cols = [col for col, base_snp in col_to_snp.items() if base_snp == snp]
                if matching_cols:
                    snp_to_col[snp] = matching_cols[0]
            
            if not snp_to_col:
                print("No matching SNPs found in genotype file")
                print("Example genotype columns:", [col for col in list(geno_cols) if col not in ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']][:5])
                print("Example SNPs we're looking for:", sig_snps[:5])
                continue
                
            print(f"Found {len(snp_to_col)} matching SNPs in genotype file")
            
            # Select only needed columns
            genotypes = genotypes[['FID'] + list(snp_to_col.values())]
            
            # Rename columns to match SNP IDs
            rename_dict = {col: snp for snp, col in snp_to_col.items()}
            genotypes = genotypes.rename(columns=rename_dict)
            
            print(f"Loaded genotypes for {len(snp_to_col)} SNPs")
            
            # Print first few SNP mappings for verification
            print("\nExample SNP mappings:")
            for snp, col in list(snp_to_col.items())[:5]:
                print(f"{snp} -> {col}")
            
        except Exception as e:
            print(f"Error reading genotype file: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Read and process covariates
        try:
            # Read covariates file with proper delimiter
            covariates = pd.read_csv(covariate_file, delim_whitespace=True)
            
            # Get PC columns
            pc_cols = [col for col in covariates.columns if col.startswith('PC')]
            
            # Keep only needed columns and rename 'identifier' to match merge
            covariates = covariates[['identifier', 'sex'] + pc_cols].copy()
            covariates = covariates.rename(columns={'identifier': 'FID'})
            
            print("Processed covariates")
            print(f"Covariate columns: {covariates.columns.tolist()}")
            
        except Exception as e:
            print(f"Error processing covariates: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Merge data
        try:
            merged_data = (genotypes
                          .merge(covariates, on='FID', how='inner')
                          .merge(phenotypes, on='FID', how='inner'))
            print(f"Merged data shape: {merged_data.shape}")
        except Exception as e:
            print(f"Error merging data: {e}")
            continue
        
        # Initialize results storage
        snp_results = []
        
        # Analyze each SNP
        for snp in snp_to_col.keys():
            print(f"Analyzing {snp}")
            
            # Get phenotype columns (excluding metadata)
            pheno_cols = [col for col in phenotypes.columns if col != 'FID']
            
            # Test SNP against each phenotype
            for pheno in pheno_cols:
                # Prepare data
                model_data = merged_data[[snp, pheno, 'sex'] + pc_cols].dropna()
                
                if len(model_data) < 10:  # Skip if too few samples
                    print(f"Skipping {snp}-{pheno} due to insufficient samples")
                    continue
                    
                # Fit logistic regression
                try:
                    X = sm.add_constant(model_data[[snp, 'sex'] + pc_cols])
                    y = model_data[pheno]
                    
                    model = sm.Logit(y, X).fit(disp=0)
                    beta = model.params[snp]
                    z_stat = model.tvalues[snp]
                    p_val = model.pvalues[snp]
                    
                    snp_results.append({
                        'SNP': snp,
                        'Phenotype': pheno,
                        'Beta': beta,
                        'Z_statistic': z_stat,
                        'P_value': p_val,
                        'N': len(model_data)
                    })
                except Exception as e:
                    print(f"Error in regression for {snp}-{pheno}: {str(e)}")
                    continue
        
        if not snp_results:
            print(f"No results generated for signature {sig}")
            continue
            
        # Convert to DataFrame
        results_df = pd.DataFrame(snp_results)
        
        # Add signature Z-statistics
        results_df['Signature_Z'] = results_df['SNP'].map(sig_z_stats)
        
        # Find SNPs significant in signature but not individual phenotypes
        sig_threshold = 5e-8
        interesting_snps = results_df.groupby('SNP').agg({
            'P_value': lambda x: all(x > sig_threshold),
            'Z_statistic': lambda x: np.mean(np.abs(x)),
            'Signature_Z': 'first',  # Take the signature Z-stat
            'N': 'mean'  # Average sample size
        }).query('P_value == True').sort_values('Z_statistic', ascending=False)
        
        results_dict[sig] = {
            'full_results': results_df,
            'interesting_snps': interesting_snps
        }
        
        # Create visualization
        if len(interesting_snps) > 0:
            plt.figure(figsize=(15, 8))
            
            # Create subplot for heatmap
            plt.subplot(121)
            
            # Get data for top SNPs
            top_snps = interesting_snps.head(10).index
            plot_data = results_df[results_df['SNP'].isin(top_snps)]
            
            # Create heatmap of Z-statistics
            pivot_data = plot_data.pivot(index='SNP', 
                                       columns='Phenotype', 
                                       values='Z_statistic')
            
            # Add signature Z-statistics as an additional column
            sig_z_col = pd.DataFrame(
                interesting_snps.loc[pivot_data.index, 'Signature_Z'],
                columns=['Signature_Z']
            )
            pivot_data = pd.concat([pivot_data, sig_z_col], axis=1)
            
            # Plot heatmap
            sns.heatmap(pivot_data, 
                       center=0, 
                       cmap='RdBu_r',
                       annot=True, 
                       fmt='.2f',
                       cbar_kws={'label': 'Z-statistic'})
            
            plt.title(f'Signature {sig}: SNPs with No Individual Disease Associations\n'
                     f'Z-statistics for Constituent Phenotypes and Signature')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(f"{output_dir}/signature_{sig}_snp_heatmap.pdf", 
                          bbox_inches='tight', dpi=300)
            plt.close()
            
            # Print summary
            print("\nTop SNPs summary:")
            summary_df = interesting_snps.head(10)[['Z_statistic', 'Signature_Z', 'N']]
            summary_df = summary_df.round(2)
            print(summary_df.to_string())
    
    return results_dict

def create_traditional_calibration_plot(checkpoint_path, mu_dt, n_bins=10):
    """
    Create a traditional calibration plot with binned risk predictions
    for 1-year risks using a model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint containing state_dict
        mu_dt: The observed prevalence matrix (shape: D x T)
        n_bins: Number of bins to use (default: 10 for deciles)
        
    Returns:
        Calibration plot figure
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    # Reconstruct model from state dict
    model = YourModelClass()  # Replace with your actual model class
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get predictions from the model
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Convert to numpy for analysis
    pi_np = pi.detach().numpy()  # Shape: (N, D, T)
    
    # Flatten predictions and observations to 1D arrays
    all_predictions = []
    all_observations = []
    
    n_diseases, n_timepoints = mu_dt.shape
    
    # For each disease and timepoint, collect predictions and observations
    for d in range(n_diseases):
        for t in range(n_timepoints):
            # Skip if prevalence is NaN
            if np.isnan(mu_dt[d, t]):
                continue
                
            # Get predictions for this disease-timepoint
            pred_dt = pi_np[:, d, t]
            
            # Get actual observations from Y if available, otherwise use mu_dt
            if hasattr(model, 'Y'):
                obs_dt = model.Y[:, d, t].detach().numpy()
            else:
                # Use prevalence as proxy for observations
                obs_dt = np.random.binomial(1, mu_dt[d, t], size=len(pred_dt))
            
            # Add to our collections
            all_predictions.extend(pred_dt)
            all_observations.extend(obs_dt)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_observations = np.array(all_observations)
    
    # Create bins based on predictions
    bin_edges = np.percentile(all_predictions, np.linspace(0, 100, n_bins+1))
    
    # Ensure unique bin edges (can happen with very skewed distributions)
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    
    # Initialize arrays for bin statistics
    bin_pred_means = np.zeros(n_bins)
    bin_obs_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_lower_bounds = np.zeros(n_bins)
    bin_upper_bounds = np.zeros(n_bins)
    
    # Calculate statistics for each bin
    for i in range(n_bins):
        if i == n_bins - 1:
            # Include the right edge in the last bin
            mask = (all_predictions >= bin_edges[i]) & (all_predictions <= bin_edges[i+1])
        else:
            mask = (all_predictions >= bin_edges[i]) & (all_predictions < bin_edges[i+1])
        
        bin_pred_means[i] = np.mean(all_predictions[mask])
        bin_obs_means[i] = np.mean(all_observations[mask])
        bin_counts[i] = np.sum(mask)
        
        # Calculate 95% confidence intervals using Wilson score interval
        if bin_counts[i] > 0:
            n = bin_counts[i]
            p = bin_obs_means[i]
            z = 1.96  # 95% confidence
            
            # Wilson score interval
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n))/denominator
            halfwidth = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator
            
            bin_lower_bounds[i] = max(0, center - halfwidth)
            bin_upper_bounds[i] = min(1, center + halfwidth)
        else:
            bin_lower_bounds[i] = 0
            bin_upper_bounds[i] = 0
    
    # Create the calibration plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the calibration curve
    ax.plot(bin_pred_means, bin_obs_means, 'o-', markersize=8, label='Calibration curve')
    
    # Add error bars for 95% confidence intervals
    ax.errorbar(bin_pred_means, bin_obs_means, 
                yerr=[bin_obs_means - bin_lower_bounds, bin_upper_bounds - bin_obs_means],
                fmt='none', capsize=5, color='blue', alpha=0.5)
    
    # Add perfect calibration line
    ax.plot([0, max(bin_pred_means)*1.1], [0, max(bin_pred_means)*1.1], 'k--', label='Perfect calibration')
    
    # Add bin sizes as text
    for i in range(n_bins):
        ax.annotate(f'n={int(bin_counts[i])}', 
                   (bin_pred_means[i], bin_obs_means[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    # Calculate metrics
    mse = np.mean((bin_pred_means - bin_obs_means)**2)
    
    # Add metrics to plot
    ax.text(0.05, 0.95, 
            f"MSE: {mse:.6f}\nNumber of bins: {n_bins}\nTotal observations: {len(all_observations)}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Predicted Risk')
    ax.set_ylabel('Observed Event Rate')
    ax.set_title('Traditional Calibration Plot (1-year risks)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Set limits with some padding
    max_val = max(max(bin_pred_means), max(bin_obs_means)) * 1.1
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Calibration plot created with {n_bins} bins")
    print(f"MSE between predicted and observed: {mse:.6f}")
    
    return fig, bin_pred_means, bin_obs_means, bin_counts

def create_enrollment_calibration_plot(predictions, observations, n_bins=10):
    """
    Create a calibration plot for 1-year risks using enrollment predictions.
    
    Args:
        predictions: numpy array of predicted risks at enrollment
        observations: numpy array of observed outcomes
        n_bins: Number of bins to use (default: 10 for deciles)
        
    Returns:
        Calibration plot figure
    """
    # Create bins based on predictions
    bin_edges = np.percentile(predictions, np.linspace(0, 100, n_bins+1))
    
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    
    # Initialize arrays for bin statistics
    bin_pred_means = np.zeros(n_bins)
    bin_obs_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_lower_bounds = np.zeros(n_bins)
    bin_upper_bounds = np.zeros(n_bins)
    
    # Calculate statistics for each bin
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i+1])
        else:
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        
        bin_pred_means[i] = np.mean(predictions[mask])
        bin_obs_means[i] = np.mean(observations[mask])
        bin_counts[i] = np.sum(mask)
        
        # Calculate 95% confidence intervals using Wilson score interval
        if bin_counts[i] > 0:
            n = bin_counts[i]
            p = bin_obs_means[i]
            z = 1.96  # 95% confidence
            
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n))/denominator
            halfwidth = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator
            
            bin_lower_bounds[i] = max(0, center - halfwidth)
            bin_upper_bounds[i] = min(1, center + halfwidth)
        else:
            bin_lower_bounds[i] = 0
            bin_upper_bounds[i] = 0
    
    # Create the calibration plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the calibration curve
    ax.plot(bin_pred_means, bin_obs_means, 'o-', markersize=8, label='Calibration curve')
    
    # Add error bars for 95% confidence intervals
    ax.errorbar(bin_pred_means, bin_obs_means, 
                yerr=[bin_obs_means - bin_lower_bounds, bin_upper_bounds - bin_obs_means],
                fmt='none', capsize=5, color='blue', alpha=0.5)
    
    # Add perfect calibration line
    ax.plot([0, max(bin_pred_means)*1.1], [0, max(bin_pred_means)*1.1], 'k--', label='Perfect calibration')
    
    # Add bin sizes as text
    for i in range(n_bins):
        ax.annotate(f'n={int(bin_counts[i])}', 
                   (bin_pred_means[i], bin_obs_means[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    # Calculate metrics
    mse = np.mean((bin_pred_means - bin_obs_means)**2)
    
    # Add metrics to plot
    ax.text(0.05, 0.95, 
            f"MSE: {mse:.6f}\nNumber of bins: {n_bins}\nTotal observations: {len(observations)}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Predicted Risk')
    ax.set_ylabel('Observed Event Rate')
    ax.set_title('Calibration Plot at Enrollment')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Set limits with some padding
    max_val = max(max(bin_pred_means), max(bin_obs_means)) * 1.1
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    
    return fig, bin_pred_means, bin_obs_means, bin_counts

def plot_ten_year_roc_comparison(our_preds, cox_preds, pce_preds, prevent_preds, gail_preds, outcomes, age_groups=None):
    """
    Create ROC curves comparing 10-year predictions across different models.
    
    Args:
        our_preds: Dict mapping age groups to our model's predictions
        cox_preds: Dict mapping age groups to Cox model predictions
        pce_preds: Dict mapping age groups to PCE predictions (cardiovascular only)
        prevent_preds: Dict mapping age groups to PREVENT predictions
        gail_preds: Dict mapping age groups to Gail model predictions (breast cancer only)
        outcomes: Dict mapping age groups to observed outcomes
        age_groups: List of age group tuples (start_age, end_age)
    """
    if age_groups is None:
        age_groups = [(40,45), (45,50), (50,55), (55,60), (60,65)]
    
    n_groups = len(age_groups)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    aucs = {model: [] for model in ['Our Model', 'Cox', 'PCE', 'PREVENT', 'Gail']}
    
    for i, (start_age, end_age) in enumerate(age_groups):
        if i < len(axes):  # Only plot if we have space
            ax = axes[i]
            
            # Plot ROC curves for each model
            if our_preds.get((start_age, end_age)) is not None:
                fpr, tpr, _ = roc_curve(outcomes[(start_age, end_age)], our_preds[(start_age, end_age)])
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Our Model (AUC={auc_score:.3f})')
                aucs['Our Model'].append(auc_score)
            
            if cox_preds.get((start_age, end_age)) is not None:
                fpr, tpr, _ = roc_curve(outcomes[(start_age, end_age)], cox_preds[(start_age, end_age)])
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Cox (AUC={auc_score:.3f})')
                aucs['Cox'].append(auc_score)
            
            if pce_preds.get((start_age, end_age)) is not None:
                fpr, tpr, _ = roc_curve(outcomes[(start_age, end_age)], pce_preds[(start_age, end_age)])
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'PCE (AUC={auc_score:.3f})')
                aucs['PCE'].append(auc_score)
            
            if prevent_preds.get((start_age, end_age)) is not None:
                fpr, tpr, _ = roc_curve(outcomes[(start_age, end_age)], prevent_preds[(start_age, end_age)])
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'PREVENT (AUC={auc_score:.3f})')
                aucs['PREVENT'].append(auc_score)
            
            if gail_preds.get((start_age, end_age)) is not None:
                fpr, tpr, _ = roc_curve(outcomes[(start_age, end_age)], gail_preds[(start_age, end_age)])
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Gail (AUC={auc_score:.3f})')
                aucs['Gail'].append(auc_score)
            
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Age {start_age}-{end_age}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Use the last subplot for the AUC bar plot
    ax = axes[-1]
    models = list(aucs.keys())
    x = np.arange(len(models))
    means = [np.mean(aucs[m]) for m in models]
    stds = [np.std(aucs[m]) for m in models]
    
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylabel('Mean AUC')
    ax.set_title('Average Performance Across Age Groups')
    
    plt.tight_layout()
    return fig, aucs

def plot_model_comparison_bars(aladyn_results, cox_results, event_rates, major_diseases, figsize=(15, 8)):
    """
    Create a bar plot comparing 10-year model performances with event rates.
    
    Args:
        aladyn_results (dict): Results from Aladynoulli model
        cox_results (dict): Results from Cox model
        event_rates (dict): Event rates from calculate_enrollment_event_rates
        major_diseases (list): List of major diseases to include
        figsize (tuple): Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set style
    plt.style.use('seaborn')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Number of diseases and bar width
    n_diseases = len(major_diseases)
    bar_width = 0.2  # Narrower bars to fit more
    
    # Positions for bars
    indices = np.arange(n_diseases)
    
    # Extract 10-year performance metrics and event rates
    aladyn_scores = []
    cox_scores = []
    event_rate_values = []
    
    for disease in major_diseases:
        # Get Aladynoulli score
        aladyn_scores.append(aladyn_results[disease]['10_year_auc'])
        
        # Get Cox score
        cox_scores.append(cox_results[disease]['10_year_auc'])
        
        # Get event rate
        event_rate_values.append(event_rates[disease]['10_year_rate'])
    
    # Create bars
    ax.bar(indices - 1.5*bar_width, aladyn_scores, bar_width, 
           label='Aladynoulli', color='#2ecc71', alpha=0.8)
    ax.bar(indices - 0.5*bar_width, cox_scores, bar_width,
           label='Cox', color='#3498db', alpha=0.8)
    ax.bar(indices + 0.5*bar_width, event_rate_values, bar_width,
           label='Event Rate', color='#95a5a6', alpha=0.8)
    
    # Add PCE/PREVENT for ASCVD
    if 'ASCVD' in major_diseases:
        ascvd_idx = major_diseases.index('ASCVD')
        ax.bar(indices[ascvd_idx] + 1.5*bar_width, 0.678, bar_width,
               label='PCE', color='#e74c3c', alpha=0.8)
        ax.bar(indices[ascvd_idx] + 2.5*bar_width, 0.66, bar_width,
               label='PREVENT', color='#9b59b6', alpha=0.8)
    
    # Add Gail for Breast Cancer
    if 'Breast_Cancer' in major_diseases:
        bc_idx = major_diseases.index('Breast_Cancer')
        ax.bar(indices[bc_idx] + 1.5*bar_width, 0.541, bar_width,
               label='Gail', color='#f1c40f', alpha=0.8)
    
    # Customize plot
    ax.set_ylabel('10-year AUC / Event Rate', fontsize=12)
    ax.set_title('10-year Performance Comparison by Disease', fontsize=14, pad=20)
    ax.set_xticks(indices)
    ax.set_xticklabels(major_diseases, rotation=45, ha='right')
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, max(max(aladyn_scores), max(cox_scores), max(event_rate_values), 0.7) * 1.1)
    
    # Add legend with smaller font
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_ten_year_comparison_plot(aladyn_results, cox_results, event_rates, major_diseases):
    """
    Creates a 10-year comparison plot for major diseases including ASCVD and breast cancer.
    
    Args:
        aladyn_results (dict): Results from Aladynoulli model
        cox_results (dict): Results from Cox model
        event_rates (dict): Event rates dictionary
        major_diseases (list): List of major diseases to include
    """
    return plot_model_comparison_bars(
        aladyn_results=aladyn_results,
        cox_results=cox_results,
        event_rates=event_rates,
        major_diseases=major_diseases,
        figsize=(15, 8)
    )