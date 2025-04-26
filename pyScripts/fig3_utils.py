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
from matplotlib.legend_handler import HandlerTuple

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


def get_signature_colors(K):
    """
    Get a consistent color palette for K signatures.
    Returns a list of K colors that will be used consistently across all plots.
    Uses a carefully curated scientific color palette that is colorblind-friendly.
    """
    # Define a carefully curated scientific color palette
    # These colors are chosen to be distinct, professional, and colorblind-friendly
    base_colors = [
        '#4C72B0',  # Strong blue
        '#DD8452',  # Orange
        '#55A868',  # Green
        '#C44E52',  # Red
        '#8172B3',  # Purple
        '#937860',  # Brown
        '#DA8BC3',  # Pink
        '#8C8C8C',  # Gray
        '#CCB974',  # Light brown
        '#64B5CD',  # Light blue
        '#4C3B4D',  # Dark purple
        '#B47C80',  # Dusty rose
        '#7C9FB0',  # Steel blue
        '#A5A449',  # Olive
        '#BE9C8E',  # Taupe
    ]
    
    if K <= len(base_colors):
        return base_colors[:K]
    else:
        # If we need more colors, use seaborn's colorblind palette
        return sns.color_palette("colorblind", K)



# Helper Function to Find Batch Dirs
def find_batch_dirs(base_dir):
    dirs = []
    print(f"Searching for batch directories in: {base_dir}")
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith('output_'):
            try:
                parts = item.split('_')
                if len(parts) >= 3:
                    start = int(parts[1])
                    end = int(parts[2])
                    dirs.append({'path': full_path, 'start': start, 'end': end, 'name': item})
            except (IndexError, ValueError) as e:
                print(f"Warning: Error parsing directory name '{item}': {e}")
    if not dirs:
        print("Warning: No directories matching 'output_*' found.")
    return sorted(dirs, key=lambda x: x['start'])

# Disease Name Loading Function
def load_and_process_disease_names(model_data, expected_dim_D):
    """Loads and validates disease names from model_data."""
    disease_names_list = []
    print("Attempting to load disease names...")
    try:
        if 'disease_names' not in model_data: raise KeyError("disease_names key not found in model_data")
        raw_names = model_data['disease_names']
        print(f"  Found 'disease_names' with type: {type(raw_names)}")

        # Handle potential nested structures (like list containing one array/list)
        if isinstance(raw_names, (list, tuple)) and len(raw_names) == 1:
             if hasattr(raw_names[0], '__iter__') and not isinstance(raw_names[0], str):
                 print("  Adjusting nested structure.")
                 raw_names = raw_names[0]

        # Process based on type
        if isinstance(raw_names, pd.DataFrame):
             if raw_names.shape[1] > 0:
                 col_name = raw_names.columns[0]
                 print(f"  Detected DataFrame. Using first column ('{col_name}'). Shape: {raw_names.shape}")
                 disease_names_list = [str(item).strip() for item in raw_names.iloc[:, 0].tolist()]
             else: raise ValueError("Empty DataFrame provided for disease names")
        elif isinstance(raw_names, (list, tuple)):
            print(f"  Detected list/tuple. Length: {len(raw_names)}")
            disease_names_list = [str(item).strip() for item in raw_names]
        elif isinstance(raw_names, (np.ndarray, pd.Series, pd.Index)):
            print(f"  Detected numpy/pandas Series/Index. Shape: {raw_names.shape}")
            disease_names_list = [str(item).strip() for item in raw_names.tolist()]
        elif torch.is_tensor(raw_names):
             print(f"  Detected torch Tensor. Shape: {raw_names.shape}")
             if raw_names.ndim > 1: raw_names = raw_names.flatten()
             disease_names_list = [str(item).strip() for item in raw_names.tolist()]
        else: raise TypeError(f"Unexpected type for disease names: {type(raw_names)}")

        print(f"  Successfully processed {len(disease_names_list)} raw names.")

        # Validation
        if len(disease_names_list) != expected_dim_D:
            raise ValueError(f"Disease name count ({len(disease_names_list)}) does not match expected dimension D ({expected_dim_D})")
        if not all(isinstance(s, str) and s for s in disease_names_list):
            invalid_examples = [s for s in disease_names_list if not (isinstance(s, str) and s)][:5]
            raise ValueError(f"Invalid or empty string names found. Examples: {invalid_examples}")

        print(f"  Successfully loaded and validated {len(disease_names_list)} disease names.")

    except Exception as e:
        print(f"  Error loading/processing disease names: {e}. Traceback:")
        traceback.print_exc()
        print(f"  Fallback: Using generic disease names ('Disease 0' to 'Disease {expected_dim_D-1}')")
        disease_names_list = [f"Disease {i}" for i in range(expected_dim_D)]
    return disease_names_list

# Main Function for Plotting Multi-Morbid Patient Dynamics (Figure 3)
def plot_multi_morbid_dynamics_from_batches(results_base_dir,
                                            target_disease_indices,
                                            signature_refs_path,
                                            min_conditions=2,
                                            n_patients_to_plot=3,
                                            random_seed=42,
                                            figsize_per_patient=(14, 7),
                                            require_all_targets=False):  # New parameter
    """
    Loads reference signature trajectories from signature_refs_path.
    Scans model batches (*.pt files in output_* dirs) to find multi-morbid
    patients (that have target diseases and multiple conditions).
    Maps ALL patient conditions to signatures using the full psi matrix.
    Plots their signature dynamics (theta) relative to diagnoses and 
    the loaded reference signature trajectories.

    Args:
        results_base_dir: Directory containing output_* subdirectories with model.pt files
        target_disease_indices: List of disease indices to search for
        signature_refs_path: Path to .pt file containing {'signature_refs': tensor/array of shape K x T}
        min_conditions: Minimum number of total diseases a patient must have
        n_patients_to_plot: Number of randomly selected patients to plot
        random_seed: For reproducibility
        figsize_per_patient: Figure size for each patient plot
        require_all_targets: If True, patients must have ALL target diseases (not just one)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    essentials = {}  # Initialize essentials dictionary

    # --- Step 0: Load External Signature References ---
    print(f"\n--- Loading External Signature References from: {signature_refs_path} ---")
    signature_refs = None
    pop_avg_theta = {}  # Initialize population average dictionary
    try:
        if not os.path.exists(signature_refs_path):
             raise FileNotFoundError(f"Signature reference file not found: {signature_refs_path}")

        refs_data = torch.load(signature_refs_path, map_location='cpu')
        if 'signature_refs' not in refs_data:
            raise KeyError(f"'signature_refs' key not found in {signature_refs_path}")

        signature_refs = refs_data['signature_refs']

        # Ensure it's a numpy array
        if torch.is_tensor(signature_refs):
            signature_refs = signature_refs.detach().cpu().numpy()
        elif not isinstance(signature_refs, np.ndarray):
            signature_refs = np.array(signature_refs)

        print(f"  Successfully loaded signature_refs with shape: {signature_refs.shape}")
        if signature_refs.ndim != 2:
             print(f"  Warning: Expected signature_refs to have 2 dimensions (KxT), but got {signature_refs.ndim}. Proceeding, but errors may occur later if K or T don't match model data.")
             # Store K and T derived from refs IF possible and valid, otherwise will rely on model file
             essentials['K_refs'] = signature_refs.shape[0] if signature_refs.ndim >=1 else None
             essentials['T_refs'] = signature_refs.shape[1] if signature_refs.ndim ==2 else None

    except FileNotFoundError as e:
         print(f"  Error: {e}. Cannot load reference signatures.")
         print("  Population average plot lines will be unavailable.")
         # Proceed without reference signatures if the file isn't found
    except KeyError as e:
         print(f"  Error: {e}. Check the content of the .pt file.")
         print("  Population average plot lines will be unavailable.")
    except Exception as e:
         print(f"  An unexpected error occurred loading signature references: {e}")
         traceback.print_exc()
         print("  Population average plot lines will be unavailable.")

    # --- Step 1: Load Reference Data from First Batch ---
    batch_dirs = find_batch_dirs(results_base_dir)
    if not batch_dirs:
        print("Error: No batch directories found. Cannot proceed.")
        return essentials, []

    first_batch_dir = batch_dirs[0]['path']
    first_model_path = os.path.join(first_batch_dir, 'model.pt')
    if not os.path.exists(first_model_path):
        print(f"Error: model.pt not found in the first batch directory: {first_batch_dir}")
        return essentials, []

    print(f"\n--- Loading Model Structure Data from First Batch: {first_model_path} ---")
    try:
        ref_model_data = torch.load(first_model_path, map_location='cpu')
        if 'model_state_dict' not in ref_model_data: raise KeyError("'model_state_dict' not found.")
        state_dict = ref_model_data['model_state_dict']
        if 'psi' not in state_dict: raise KeyError("'psi' not found.")
        if 'lambda_' not in state_dict: raise KeyError("'lambda_' not found.")  # Need lambda for shape info

        ref_psi = state_dict['psi'].detach().cpu()      # K x D
        ref_lambda_shape_info = state_dict['lambda_']  # Only need shape N x K x T
        N_batch1, K_model, T_model = ref_lambda_shape_info.shape  # Get K and T from model
        D_ref = ref_psi.shape[1]
        print(f"  Model shapes: Psi ({K_model}x{D_ref}), Lambda T ({T_model}), Lambda K ({K_model})")

        # Validate consistency between loaded refs and model K, T
# Validate consistency between loaded refs and model K, T
        valid_refs = False
        if signature_refs is not None:
            K_refs = signature_refs.shape[0] if signature_refs.ndim >= 1 else None
            T_refs = signature_refs.shape[1] if signature_refs.ndim == 2 else None
            
            # Allow K_refs to be either equal to K_model (no healthy signature) or K_model-1 (with healthy signature)
            if K_model > 1 and K_refs == K_model - 1:
                print(f"  K mismatch explained: signature_refs K={K_refs}, model K={K_model} - the model has a healthy signature.")
                print("  Will use signature_refs for K disease signatures and add constant -5.0 for the healthy signature.")
                valid_refs = True
            elif K_refs == K_model:
                print("  Signature references K and T dimensions match model data exactly.")
                valid_refs = True
            elif T_refs != T_model:
                print(f"  ERROR: T mismatch! signature_refs T={T_refs}, model T={T_model}. Cannot use references.")
            elif signature_refs.ndim != 2:
                print(f"  ERROR: signature_refs has {signature_refs.ndim} dimensions, expected 2 (KxT). Cannot use references.")
            else:
                print(f"  ERROR: Unexpected K mismatch. signature_refs K={K_refs}, model K={K_model}.")

        # Load disease names
        essentials['disease_names'] = load_and_process_disease_names(ref_model_data, D_ref)

        # *** MAJOR CHANGE: Map ALL diseases to their primary signatures, not just targets ***
        # This will let us visualize any disease a patient has, not just target diseases
        disease_primary_sig = {}
        print("\nDetermining primary signatures for ALL diseases (using reference Psi):")
        for d_idx in range(D_ref):  # Loop over all diseases
            primary_sig = torch.argmax(ref_psi[:, d_idx]).item()
            disease_primary_sig[d_idx] = primary_sig
            # Only print details for target diseases to avoid verbose output
            if d_idx in target_disease_indices:
                disease_name = essentials['disease_names'][d_idx]
                print(f"  - Target Disease {d_idx} ('{disease_name}'): Primary Signature {primary_sig}")
        
        print(f"  Mapped all {D_ref} diseases to their primary signatures.")
        
        # Store the full mapping for later use
        essentials['disease_primary_sig'] = disease_primary_sig
        
        # Get the unique set of signatures we need references for
        all_sigs_to_use = sorted(list(set(disease_primary_sig.values())))

        # Populate pop_avg_theta using the VALIDATED external references
# Populate pop_avg_theta using the VALIDATED external references
        if valid_refs:
            print(f"\nAssigning population average theta using external references for signatures: {all_sigs_to_use[:10]}... (and more)")
            for sig_idx in all_sigs_to_use:
                if 0 <= sig_idx < K_model:
                    # Special handling for the last signature if K_refs < K_model
                    if K_refs < K_model and sig_idx == K_model - 1:
                        # This is the healthy signature - use constant -5.0
                        pop_avg_theta[sig_idx] = np.full(T_model, -5.0)
                        print(f"  Created healthy reference (-5.0) for signature {sig_idx}")
                    elif sig_idx < K_refs:
                        # Regular signature - use the loaded references
                        pop_avg_theta[sig_idx] = signature_refs[sig_idx, :]
                    else:
                        print(f"  Warning: Signature index {sig_idx} is out of bounds for signature_refs K={K_refs}. Cannot assign reference.")
                else:
                    print(f"  Warning: Signature index {sig_idx} is out of bounds for model K={K_model}. Cannot assign reference.")
            print(f"  Assigned references for {len(pop_avg_theta)} signatures.")   # Store essential info
        time_points = np.arange(T_model)  # Use T from the model
        essentials['time_points'] = time_points
        essentials['pop_avg_theta'] = pop_avg_theta  # Contains refs or is empty
        essentials['K'] = K_model  # Use K from the model
        essentials['T'] = T_model  # Use T from the model
        essentials['D'] = D_ref

    except Exception as e:
        print(f"Error loading or processing reference data from {first_model_path}: {e}")
        traceback.print_exc()
        # Ensure essentials dict has some defaults if possible, otherwise return empty
        essentials['pop_avg_theta'] = essentials.get('pop_avg_theta', {})  # Keep loaded refs if loading failed later
        essentials['disease_names'] = essentials.get('disease_names', [])
        return essentials, []

    # --- Step 2: Scan All Batches to Find Multi-Morbid Patients ---
    print(f"\n--- Scanning All {len(batch_dirs)} Batches for Patients with >= {min_conditions} Total Conditions and At Least 1 Target Disease ---")
    found_patients_pool = []  # List to store tuples: (batch_info, patient_idx_in_batch, conditions_list, global_idx)
    total_patients_scanned = 0

    for batch_info in batch_dirs:
        batch_model_path = os.path.join(batch_info['path'], 'model.pt')
        if not os.path.exists(batch_model_path):
            continue

        try:
            # Load the whole model file as Y needs to be checked
            model_data = torch.load(batch_model_path, map_location='cpu')

            if 'Y' not in model_data:
                continue
            Y_batch = model_data['Y']

            # Convert Y to numpy [N_batch x D x T] if it's a tensor
            if torch.is_tensor(Y_batch):
                Y_batch_np = Y_batch.detach().cpu().numpy()
            elif isinstance(Y_batch, np.ndarray):
                Y_batch_np = Y_batch  # Already numpy
            else:  # Try converting lists/other types
                try: Y_batch_np = np.array(Y_batch)
                except Exception as e_conv:
                    print(f"    Warning: Could not convert Y data in {batch_info['name']} to numpy array. Error: {e_conv}. Skipping batch.")
                    continue

            # Validate dimensions
            if Y_batch_np.ndim != 3:
                continue
            current_N, current_D, current_T = Y_batch_np.shape
            total_patients_scanned += current_N

            if current_D != D_ref:
                continue
            if current_T != T_model:
                continue  # Skip if time dimension doesn't match reference

            # Check each patient in this batch
            for i in range(current_N):  # Iterate through patients in this batch
                # For storing ALL conditions
                all_conditions = []  # List of (disease_idx, first_diagnosis_time_idx)
                # For tracking if the patient has at least one target disease
                has_target_disease = False
                
                # Scan through all diseases for this patient
                for d_idx in range(current_D):
                    # Find indices where disease occurs
                    diag_times_idx = np.where(Y_batch_np[i, d_idx, :] > 0.5)[0]
                    if len(diag_times_idx) > 0:
                        first_diag_idx = diag_times_idx[0]  # First diagnosis time
                        all_conditions.append((d_idx, first_diag_idx))
                
                # Check target disease requirements
                target_diseases_found = set(d_idx for d_idx, _ in all_conditions if d_idx in target_disease_indices)
                
                if require_all_targets:
                    # Must have ALL target diseases
                    has_all_targets = len(target_diseases_found) == len(target_disease_indices)
                    if not has_all_targets or len(all_conditions) < min_conditions:
                        continue
                else:
                    # Original behavior: must have at least one target disease
                    if not target_diseases_found or len(all_conditions) < min_conditions:
                        continue
                    
                # If we get here, the patient meets all criteria
                # Sort conditions by diagnosis time index
                all_conditions.sort(key=lambda x: x[1])
                global_idx = batch_info['start'] + i  # Calculate global patient index
                found_patients_pool.append((batch_info, i, all_conditions, global_idx))

        except Exception as e_batch:
            print(f"  Error processing batch {batch_info['name']}: {e_batch}")

    if not found_patients_pool:
        message = "No patients found meeting criteria:\n"
        message += f"- {'All' if require_all_targets else 'At least one of'} target diseases {target_disease_indices}\n"
        message += f"- At least {min_conditions} total conditions\n"
        message += f"Scanned {total_patients_scanned:,} patients total."
        print(f"\n{message}")
        return essentials, []

    print(f"\nScan complete. Found {len(found_patients_pool)} patients meeting criteria:")
    print(f"- {'All' if require_all_targets else 'At least one of'} target diseases {target_disease_indices}")
    print(f"- At least {min_conditions} total conditions")
    print(f"Out of {total_patients_scanned:,} total patients scanned.")
    
    # --- Step 3: Select Patients and Generate Plots ---
    n_to_select = min(n_patients_to_plot, len(found_patients_pool))
    print(f"\n--- Randomly Selecting {n_to_select} Patients to Plot ---")
    # selected_patient_info contains tuples: (batch_info, patient_idx_in_batch, conditions_list, global_idx)
    selected_patient_info = random.sample(found_patients_pool, n_to_select)

    # Retrieve reference data from essentials
    disease_names_list = essentials.get('disease_names', [f"Disease {i}" for i in range(D_ref)])
    time_points = essentials.get('time_points', np.arange(T_model))
    pop_avg_theta = essentials.get('pop_avg_theta', {})
    disease_primary_sig = essentials.get('disease_primary_sig', {})  # Map of ALL diseases to primary signatures
    
    print("\n--- Generating Plots ---")
    patient_figures = []  # List to store generated figures
    for batch_info, patient_idx_in_batch, conditions, global_patient_idx in selected_patient_info:
        print(f"\nGenerating plot for Patient {global_patient_idx} (from batch {batch_info['name']}, index {patient_idx_in_batch})...")
        print(f"  Conditions (Index, Time Index): {conditions}")

        try:
            # Load the specific model file again to get this patient's lambda
            batch_model_path = os.path.join(batch_info['path'], 'model.pt')
            model_data = torch.load(batch_model_path, map_location='cpu')
            if 'model_state_dict' not in model_data or 'lambda_' not in model_data['model_state_dict']:
                print(f"  Error: Could not load lambda_ for patient {global_patient_idx} from {batch_model_path}. Skipping.")
                continue

            # Extract the lambda for this specific patient: K x T
            lambda_patient = model_data['model_state_dict']['lambda_'][patient_idx_in_batch].detach().cpu()

            # Calculate patient's theta by applying softmax over signatures (dim 0)
            theta_patient = torch.softmax(lambda_patient, dim=0).numpy()  # Shape: K x T

            # --- Create the 2-panel plot side by side ---
            fig = plt.figure(figsize=(figsize_per_patient[0] * 1.3, figsize_per_patient[1]))  # Wider to accommodate side panel
            gs = GridSpec(2, 2, height_ratios=[2.5, 1], width_ratios=[4, 0.8], hspace=0.05)  # Make right panel skinnier
            ax1 = fig.add_subplot(gs[0, 0])  # Top left: temporal theta
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom left: timeline
            ax3 = fig.add_subplot(gs[:, 1])  # Right side: stacked bar

            # First collect valid conditions and prepare
            valid_conditions = []
            for d_idx, diag_time_idx in conditions:
                sig_idx = disease_primary_sig.get(d_idx)
                if sig_idx is not None:
                    valid_conditions.append((d_idx, diag_time_idx, sig_idx))
                else:
                    print(f"    Skipping condition {d_idx}: No primary signature found in reference.")

            if not valid_conditions:
                print(f"  Warning: None of the {len(conditions)} conditions could be mapped to signatures. Skipping patient.")
                continue

            # Get our consistent colors
            colors = get_signature_colors(K_model)  # Use K_model to get colors for all possible signatures

            # Prepare condition details and collect signatures
            condition_details = []
            signatures_to_plot = set()

            for i, (d_idx, diag_time_idx, sig_idx) in enumerate(valid_conditions):
                signatures_to_plot.add(sig_idx)
                disease_name = disease_names_list[d_idx] if 0 <= d_idx < len(disease_names_list) else f"Disease {d_idx}"
                diag_time = time_points[diag_time_idx]
                color = colors[sig_idx]  # Use the signature's consistent color
                condition_details.append({
                    'name': disease_name,
                    'time': diag_time,
                    'color': color,
                    'y_pos': i,
                    'sig_idx': sig_idx,
                    'd_idx': d_idx
                })

            # --- Left Panels: Original temporal plot ---
            # Create a single legend handler for all plots
            legend_handles = []
            legend_labels = []
            
            for i, detail in enumerate(condition_details):
                d_idx = detail['d_idx']
                sig_idx = detail['sig_idx']
                disease_name = detail['name']
                diag_time = detail['time']
                color = detail['color']
                
                label_text_short = f"{disease_name[:20]}{'...' if len(disease_name)>20 else ''}"
                label_patient = f"Sig {sig_idx} ({label_text_short})"
                
                # Plot trajectory
                line = ax1.plot(time_points, theta_patient[sig_idx, :],
                        color=color, linewidth=2.5, alpha=0.95, 
                        zorder=3)[0]
                
                # Mark diagnosis time with skinnier line
                ax1.axvline(x=diag_time, color=color, linestyle=':',
                          alpha=0.5, linewidth=0.8, zorder=2)
                
                # Add to legend handlers
                legend_handles.append(line)
                legend_labels.append(label_patient)

            # Configure top panel
            ax1.set_title(f'Signature Loadings ($\\Theta$) Over Time\nPatient {global_patient_idx}', 
                         fontsize=14, pad=10)
            ax1.set_ylabel('Signature Loading ($\\Theta$)', fontsize=12)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax1.set_ylim(bottom=0)

            # --- Panel 2: Disease Timeline ---
            ax2.set_yticks(range(len(valid_conditions)))
            ax2.set_yticklabels([d['name'] for d in condition_details], fontsize='small')
            ax2.set_ylim(-0.5, len(valid_conditions) - 0.5)
            ax2.invert_yaxis()

            for detail in condition_details:
                # Mark diagnosis point with smaller marker
                ax2.scatter(detail['time'], detail['y_pos'], marker='o', s=40, 
                          color=detail['color'], zorder=5, label=None, 
                          edgecolors='black', linewidth=0.5)
                # Draw thinner line up to diagnosis
                ax2.hlines(detail['y_pos'], time_points[0], detail['time'], 
                         colors=detail['color'], linestyles='-', alpha=0.4, linewidth=1.0)

            ax2.set_xlabel('Age (years)', fontsize=12)
            ax2.set_ylabel('Diagnosed Condition', fontsize=12)
            ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            # --- Right Panel: Stacked Bar Summary ---
            theta_avg = np.mean(theta_patient, axis=1)  # Average over time
            bottom = 0
            sig_indices = sorted(list(signatures_to_plot))

            # Create stacked bar without individual legend
            for sig_idx in sig_indices:
                matching_detail = next(detail for detail in condition_details if detail['sig_idx'] == sig_idx)
                color = matching_detail['color']
                ax3.bar([0], [theta_avg[sig_idx]], bottom=bottom, color=color, 
                        alpha=0.7, width=0.3)  # Make bars skinnier too
                bottom += theta_avg[sig_idx]

            ax3.set_title('Static Model\nSummary', pad=15)
            ax3.set_ylabel('Average Loading (θ)')
            ax3.set_xlim(-0.3, 0.3)  # Adjust xlim to match skinnier bars
            ax3.set_xticks([])
            ax3.grid(True, axis='y', linestyle='--', alpha=0.5)

            # Create single legend for all panels
            fig.legend(legend_handles, legend_labels,
                      loc='center right', bbox_to_anchor=(0.98, 0.5),
                      fontsize='small')

            # Adjust layout with more space for the single legend
            plt.tight_layout(rect=[0, 0, 0.82, 0.97])  # Slightly more space for legend
            
            patient_figures.append(fig)

        except Exception as e_plot:
            print(f"  Error generating plot for Patient {global_patient_idx}: {e_plot}")
            traceback.print_exc()

    print(f"\nFinished plotting {len(patient_figures)} multi-morbid dynamics examples.")
    return essentials, patient_figures




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
import os
import pandas as pd # For potential disease name handling
import matplotlib.cm as cm # For colormaps
import seaborn as sns # For color palettes in visualize_signature_mixture
import traceback # Added for error details

# =============================================================================
# Visualization Function (Claude's Suggestion #1 Adaption)
# =============================================================================
def visualize_signature_mixture(lambda_values_np, phi_values_np, individual_idx, disease_idx,
                               time_points, disease_names, signature_names=None,
                               highlight_diagnoses=None, kappa=1.0):
    """
    Visualize how multiple signatures contribute to an individual's disease risk.
    Assumes inputs lambda_values_np and phi_values_np are numpy arrays.

    Args:
        lambda_values_np (np.ndarray): Individual signature loadings (N, K, T).
        phi_values_np (np.ndarray): Signature-disease logit values (K, D, T).
        individual_idx (int): Index of the individual to visualize.
        disease_idx (int): Index of the disease to visualize.
        time_points (np.ndarray): Time points array (T).
        disease_names (list): List of disease names (D).
        signature_names (list, optional): List of signature names (K). Defaults to "Signature X".
        highlight_diagnoses (dict, optional): Dict mapping time point values to True for highlighting.
        kappa (float): Kappa calibration parameter.
    """
    print("  Inside visualize_signature_mixture...")
    # --- Calculations using NumPy ---
    N, K, T = lambda_values_np.shape
    _ , D, _ = phi_values_np.shape

    # Create default signature names if not provided
    if signature_names is None:
        signature_names = [f"Signature {k}" for k in range(K)]

    # Calculate theta (normalized lambda)
    print("    Calculating theta...")
    theta = np.zeros_like(lambda_values_np)
    for i in range(N): # Loop through individuals if needed outside this function, but here we only need one
        for t in range(T):
            exp_lambda = np.exp(lambda_values_np[i, :, t])
            sum_exp_lambda = np.sum(exp_lambda)
            with np.errstate(divide='ignore', invalid='ignore'):
                 theta[i, :, t] = exp_lambda / sum_exp_lambda if sum_exp_lambda > 0 else np.zeros(K)
                 theta[i, :, t] = np.nan_to_num(theta[i, :, t])

    # Calculate disease probabilities from phi logits (sigmoid)
    print("    Calculating phi probabilities...")
    phi_probs = 1 / (1 + np.exp(-phi_values_np)) # K x D x T

    # Extract data for the specific individual and disease
    phi_disease = phi_probs[:, disease_idx, :]      # Shape: (K, T)
    theta_individual = theta[individual_idx, :, :]  # Shape: (K, T)

    # Calculate signature contributions to disease risk
    print("    Calculating contributions...")
    contributions = np.zeros((K, T))
    for k in range(K):
        contributions[k, :] = theta_individual[k, :] * phi_disease[k, :] # theta_k * sigmoid(phi_kd)

    # Total *latent* risk (sum across signatures)
    total_latent_risk = np.sum(contributions, axis=0) # Shape: (T)

    # Apply kappa to get absolute risk
    print(f"    Calculating absolute risk with kappa={kappa}...")
    absolute_risk = kappa * total_latent_risk
    absolute_risk = np.minimum(1.0, np.maximum(0.0, absolute_risk)) # Clamp to [0, 1]

    # Scale contributions proportionally so they sum to absolute_risk
    print("    Scaling contributions...")
    scaled_contributions = contributions.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate scaling factor: absolute_risk / total_latent_risk
        scaling_factor = np.divide(absolute_risk, total_latent_risk)
        # If total_latent_risk is 0, contribution is 0, so scaling factor is irrelevant (set to 0)
        scaling_factor[~np.isfinite(scaling_factor)] = 0
    # Apply scaling factor using broadcasting
    scaled_contributions *= scaling_factor[np.newaxis, :] # Broadcast (1, T) over (K, T)

    # --- Plotting ---
    print("    Creating plot panels...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 2, 1]}, sharex=True)
    palette = sns.color_palette("husl", K) # Consistent color palette

    # Get the specific disease name safely
    title_d_name = disease_names[disease_idx] if 0 <= disease_idx < len(disease_names) else f"Disease {disease_idx}"

    # Panel 1: Stacked area chart of scaled contributions
    axes[0].stackplot(time_points, scaled_contributions, labels=signature_names, colors=palette, alpha=0.7)
    axes[0].plot(time_points, absolute_risk, 'k--', linewidth=2, label='Total Calculated Risk ($\pi_{idt}$)')
    axes[0].set_title(f'Calculated Risk of "{title_d_name}" for Individual {individual_idx}\n(Composition from Signatures)', fontsize=14)
    # axes[0].set_xlabel('Age (years)', fontsize=12) # Shared X-axis
    axes[0].set_ylabel('Absolute Risk ($\pi$)', fontsize=12)
    axes[0].legend(loc='upper left', fontsize='x-small')
    axes[0].grid(alpha=0.3)
    # Adjust ylim dynamically but ensure it's at least 0 to 0.1
    plot_max_y = (np.max(absolute_risk) * 1.1) if np.any(absolute_risk > 0) else 0.1
    axes[0].set_ylim(0, plot_max_y)

    # Highlight diagnoses on Panel 1
    if highlight_diagnoses:
        print("    Adding diagnosis highlights...")
        for time_point_val, diagnosis in highlight_diagnoses.items():
             # Find the closest index in time_points array
             closest_idx = np.abs(np.array(time_points) - time_point_val).argmin()
             actual_time_on_axis = time_points[closest_idx]
             if diagnosis: # Assuming True means has diagnosis
                 # Draw line slightly shorter than full axis height
                 axes[0].axvline(x=actual_time_on_axis, color='red', linestyle=':', linewidth=1.5, alpha=0.8, ymax=0.95)
                 # Place text marker near the top, rotated
                 axes[0].text(actual_time_on_axis + 0.2, plot_max_y * 0.9, f' Dx', va='top', ha='left', color='red', fontsize=9, rotation=90)

    # Panel 2: Signature loadings (theta)
    for k in range(K):
        axes[1].plot(time_points, theta_individual[k, :], '-', color=palette[k], linewidth=2, label=signature_names[k])
    axes[1].set_title(f'Individual Signature Loadings ($\Theta_{{ikt}}$)', fontsize=14)
    # axes[1].set_xlabel('Age (years)', fontsize=12) # Shared X-axis
    axes[1].set_ylabel('Loading ($\Theta$)', fontsize=12)
    axes[1].legend(loc='upper left', fontsize='x-small')
    axes[1].grid(alpha=0.3)
    #axes[1].set_ylim(0, 1) # Theta sums to 1

    # Panel 3: Signature-specific disease probabilities (phi_probs)
    for k in range(K):
        axes[2].plot(time_points, phi_disease[k, :], '-', color=palette[k], linewidth=2, label=signature_names[k])
    axes[2].set_title(f'Signature-Specific Probabilities ($\sigma(\phi_{{kdt}})$) for "{title_d_name}"', fontsize=14)
    axes[2].set_xlabel('Age (years)', fontsize=12) # Label only bottom x-axis
    axes[2].set_ylabel('P(Disease | Sig k)', fontsize=12)
    axes[2].legend(loc='upper left', fontsize='x-small')
    axes[2].grid(alpha=0.3)
    #axes[2].set_ylim(0, 1) # Probabilities

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly
    print("  Finished visualize_signature_mixture.")
    return fig


# =============================================================================
# Runner Function to Load Data and Call Visualization
# =============================================================================
def run_mixture_visualization_on_real_data(model_path,
                                           individual_idx_to_plot,
                                           disease_idx_to_plot,
                                           kappa_value=1.0, # Defaulting kappa to 1.0 - ADJUST THIS!
                                           output_filename=None):
    """
    Loads real model data, processes it, stores disease names in 'essentials',
    and calls visualize_signature_mixture. Ensures tensors are converted to numpy.

    Args:
        model_path (str): Path to the saved model.pt file.
        individual_idx_to_plot (int): Index of the individual to visualize.
        disease_idx_to_plot (int): Index of the disease to visualize.
        kappa_value (float): The kappa calibration parameter. Needs to be set appropriately.
        output_filename (str, optional): If provided, saves the figure to this path.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model data from {model_path}...")
    essentials = {} # Dictionary to store processed essentials like disease names

    try:
        # Load model data onto CPU
        model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False) # Consider weights_only=True if safe

        # --- Extract required data ---
        lambda_values = model_data['model_state_dict'].get('lambda_', None)
        if lambda_values is None: raise ValueError("'lambda_' not found in model_state_dict.")

        phi_values = model_data['model_state_dict'].get('phi', None)
        if phi_values is None: raise ValueError("'phi' not found in model_state_dict.")

        Y_observed = model_data.get('Y', None) # Optional

        # --- Get dimensions ---
        N, K_total, T = lambda_values.shape
        _, D, _ = phi_values.shape

        # --- Process disease names and store in essentials ---
        disease_names_list = []
        try:
            if 'disease_names' not in model_data: raise KeyError("disease_names not found")
            raw_names = model_data['disease_names']
            # Handle common nesting/types
            if isinstance(raw_names, (list, tuple)) and len(raw_names) == 1:
                 if hasattr(raw_names[0], '__iter__') and not isinstance(raw_names[0], str): raw_names = raw_names[0]
            # Convert to list of strings based on type
            if isinstance(raw_names, pd.DataFrame):
                 if raw_names.shape[1] > 0:
                     print(f"  Detected DataFrame. Using first column ('{raw_names.columns[0]}').")
                     disease_names_list = [str(item).strip() for item in raw_names.iloc[:, 0].tolist()]
                 else: raise ValueError("Empty DataFrame")
            elif isinstance(raw_names, (list, tuple)): disease_names_list = [str(item).strip() for item in raw_names]
            elif isinstance(raw_names, (np.ndarray, pd.Series, pd.Index)): disease_names_list = [str(item).strip() for item in raw_names.tolist()]
            elif torch.is_tensor(raw_names):
                 if raw_names.ndim > 1: raw_names = raw_names.flatten()
                 disease_names_list = [str(item).strip() for item in raw_names.tolist()]
            else: raise TypeError(f"Unexpected type {type(raw_names)}")
            # Validate processed list
            if len(disease_names_list) != D: raise ValueError(f"Length mismatch ({len(disease_names_list)} vs {D})")
            if not all(isinstance(s, str) and s for s in disease_names_list): raise ValueError("Invalid names found")
            # Store in essentials
            essentials['disease_names'] = disease_names_list
        except Exception as e:
            print(f"  Fallback: Using generic disease names due to error: {e}")
            disease_names_list = [f"Disease {i}" for i in range(D)]
            essentials['disease_names'] = disease_names_list # Store fallback
        # --- End disease name processing ---
        disease_names_to_pass = essentials['disease_names'] # Use the stored list
        print(f"Using {len(disease_names_to_pass)} disease names. Target: {disease_names_to_pass[disease_idx_to_plot] if 0 <= disease_idx_to_plot < len(disease_names_to_pass) else 'Index out of bounds'}")

        # --- Create time points ---
        time_points = np.linspace(30, 30 + T - 1, T) # Example: Ages 30 to (30+T-1)

        # --- Prepare diagnosis highlights (Optional) ---
        highlight_diagnoses = None
        if Y_observed is not None and 0 <= individual_idx_to_plot < N and 0 <= disease_idx_to_plot < D:
            y_patient_disease_np = None
            if torch.is_tensor(Y_observed):
                 y_patient_disease_np = Y_observed[individual_idx_to_plot, disease_idx_to_plot, :].detach().cpu().numpy()
            elif isinstance(Y_observed, np.ndarray):
                 y_patient_disease_np = Y_observed[individual_idx_to_plot, disease_idx_to_plot, :]
            else: # Try converting list/other
                 try: y_patient_disease_np = np.array(Y_observed[individual_idx_to_plot, disease_idx_to_plot, :])
                 except: print("Warning: Could not process Y_observed for diagnosis highlighting.")

            if y_patient_disease_np is not None:
                diagnosis_times_indices = np.where(y_patient_disease_np > 0.5)[0]
                if len(diagnosis_times_indices) > 0:
                    highlight_diagnoses = {time_points[idx]: True for idx in diagnosis_times_indices}
                    print(f"Found diagnoses for Patient {individual_idx_to_plot}, Disease idx {disease_idx_to_plot} at times: {[time_points[idx] for idx in diagnosis_times_indices]}")


        # --- Convert Tensors to Numpy BEFORE calling plot function ---
        print("Converting tensors to numpy arrays...")
        if torch.is_tensor(lambda_values): lambda_np = lambda_values.detach().cpu().numpy()
        else: lambda_np = np.array(lambda_values)

        if torch.is_tensor(phi_values): phi_np = phi_values.detach().cpu().numpy()
        else: phi_np = np.array(phi_values)

        kappa_float = float(kappa_value)
        # --- End Conversion ---

        # --- Call the visualization function ---
        print(f"Creating mixture visualization for Individual {individual_idx_to_plot}, Disease index {disease_idx_to_plot}...")
        fig = visualize_signature_mixture(
            lambda_np,             # Pass numpy array
            phi_np,                # Pass numpy array
            individual_idx_to_plot,
            disease_idx_to_plot,
            time_points,
            disease_names_to_pass, # Use list from essentials
            signature_names=None,  # Use default names
            highlight_diagnoses=highlight_diagnoses,
            kappa=kappa_float      # Pass float kappa
        )

        # --- Save or Show Plot ---
        if fig and output_filename:
            try:
                fig.savefig(output_filename, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {output_filename}")
            except Exception as e:
                print(f"Error saving figure: {e}")
                plt.show() # Show plot even if saving failed
        elif fig:
             plt.show()

    except Exception as e:
        print(f"An error occurred in run_mixture_visualization_on_real_data: {str(e)}")
        traceback.print_exc()


        

# model_file = '/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt' # Example path

#     # --- Parameters to set ---
#     # SELECT an individual and disease index to plot
#     # Find interesting examples by looking at Y_observed or model outputs
# example_individual_index = 100 # Choose an patient index (0 to N-1)
#     # Find disease index for 'Coronary atherosclerosis' (or another disease)
#     # This requires loading the names first, or knowing the index beforehand.
#     # Let's assume index 114 corresponds to it based on previous logs.
# example_disease_index = 114

#     # SET the KAPPA value - this is crucial for scaling the risk
#     # Load it from model_data if saved, e.g., model_data['model_state_dict']['kappa'].item()
#     # Or use the value from training. Defaulting to 1.0 is likely incorrect.

#     # <<< !!! VERIFY this KAPPA value is appropriate !!! >>>
# kappa = model.kappa
""" 
    # Define output filename
output_file = 'mix.png'

    # --- Run the visualization ---
run_mixture_visualization_on_real_data(model_file,
                                           example_individual_index,
                                           example_disease_index,
                                           kappa_value=kappa,
                                           output_filename=output_file)
 """


def visualize_individual_vs_population(
    lambda_values_np: np.ndarray,  # (N, K, T)
    phi_values_np: np.ndarray,     # (K, D, T)
    individual_idx: int,           # Index of individual with high sig 5 loading
    cv_diseases: List[str],        # List of cardiovascular diseases to show
    disease_indices: List[int],    # Corresponding disease indices
    time_points: np.ndarray,       # Time points array
    kappa: float = 1.0,
    output_path: Optional[str] = None
):
    """
    Visualize how an individual's signature loadings compare to population,
    and how this affects their disease probabilities.
    """
    N, K, T = lambda_values_np.shape
    
    # Get our consistent color palette
    signature_colors = get_signature_colors(K)
    
    # Calculate population average thetas
    print("Calculating population theta...")
    theta_pop = np.zeros((N, K, T))
    for i in range(N):
        for t in range(T):
            exp_lambda = np.exp(lambda_values_np[i, :, t])
            sum_exp_lambda = np.sum(exp_lambda)
            theta_pop[i, :, t] = exp_lambda / sum_exp_lambda if sum_exp_lambda > 0 else np.zeros(K)
    
    # Get average population theta
    theta_pop_mean = np.mean(theta_pop, axis=0)  # (K, T)
    theta_pop_std = np.std(theta_pop, axis=0)    # (K, T)
    
    # Get individual's theta
    theta_individual = theta_pop[individual_idx]  # (K, T)
    
    # Calculate disease probabilities from phi (sigmoid)
    phi_probs = 1 / (1 + np.exp(-phi_values_np))  # (K, D, T)
    
    # Create figure with 3 rows
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.5])
    
    # --- Panel 1: Signature Loadings Comparison ---
    ax1 = fig.add_subplot(gs[0])
    # Plot population mean ± std for each signature
    for k in range(K):
        color = signature_colors[k]
        ax1.fill_between(time_points, 
                        theta_pop_mean[k] - theta_pop_std[k],
                        theta_pop_mean[k] + theta_pop_std[k],
                        alpha=0.2, color=color, label=f'Population θ_{k} ± SD')
        ax1.plot(time_points, theta_pop_mean[k], '--', color=color, label=f'Population Mean θ_{k}')
        ax1.plot(time_points, theta_individual[k], '-', color=color, linewidth=2, label=f'Individual θ_{k}')
    
    ax1.set_title('Signature Loadings Over Time')
    ax1.set_ylabel('Loading (θ)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Phi Values for CV Diseases ---
    ax2 = fig.add_subplot(gs[1])
    for i, (disease, idx) in enumerate(zip(cv_diseases, disease_indices)):
        # For each disease, plot its relationship with each signature
        for k in range(K):
            color = signature_colors[k]
            ax2.plot(time_points, phi_probs[k, idx], color=color, 
                    label=f'Sig {k} - {disease}', alpha=0.7)
    
    ax2.set_title('Signature-Disease Associations (φ)')
    ax2.set_ylabel('P(Disease | Signature)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Individual vs Population Disease Probabilities ---
    ax3 = fig.add_subplot(gs[2])
    
    # Calculate probabilities for population and individual
    pop_probs = np.zeros((N, len(disease_indices), T))
    for i in range(N):
        for d_idx, disease_idx in enumerate(disease_indices):
            contribution = np.sum(theta_pop[i] * phi_probs[:, disease_idx], axis=0)
            pop_probs[i, d_idx] = kappa * contribution
    
    pop_mean = np.mean(pop_probs, axis=0)  # (D, T)
    pop_std = np.std(pop_probs, axis=0)    # (D, T)
    
    # Calculate individual probabilities
    ind_probs = np.zeros((len(disease_indices), T))
    for d_idx, disease_idx in enumerate(disease_indices):
        contribution = np.sum(theta_individual * phi_probs[:, disease_idx], axis=0)
        ind_probs[d_idx] = kappa * contribution
    
    # Plot for each disease using signature colors
    for i, (disease, disease_idx) in enumerate(zip(cv_diseases, disease_indices)):
        # Find primary signature for this disease
        primary_sig = np.argmax(np.mean(phi_probs[:, disease_idx, :], axis=1))
        color = signature_colors[primary_sig]
        
        # Population
        ax3.fill_between(time_points,
                        pop_mean[i] - pop_std[i],
                        pop_mean[i] + pop_std[i],
                        alpha=0.2, color=color)
        ax3.plot(time_points, pop_mean[i], '--', color=color, 
                label=f'{disease} (Pop, Sig {primary_sig})')
        # Individual
        ax3.plot(time_points, ind_probs[i], '-', color=color, linewidth=2,
                label=f'{disease} (Ind, Sig {primary_sig})')
    
    ax3.set_title('Disease Probabilities: Individual vs Population')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Probability')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

def visualize_disease_contribution_breakdown(
    lambda_values_np: np.ndarray,  # (N, K, T)
    phi_values_np: np.ndarray,     # (K, D, T)
    individual_idx: int,
    disease_idx: int,              
    disease_name: str,             
    time_points: np.ndarray,       
    signature_names: Optional[List[str]] = None,
    kappa: float = 1.0,
    output_path: Optional[str] = None
):
    """
    Visualize how all signatures contribute to a specific disease's probability,
    with a heatmap showing signature-disease associations over time.
    """
    # Convert inputs to numpy if they're torch tensors
    if torch.is_tensor(lambda_values_np):
        lambda_values_np = lambda_values_np.detach().cpu().numpy()
    if torch.is_tensor(phi_values_np):
        phi_values_np = phi_values_np.detach().cpu().numpy()
    if torch.is_tensor(kappa):
        kappa = kappa.item()

    N, K, T = lambda_values_np.shape

    # --- Calculations ---
    # Calculate theta for all individuals and the population average
    theta_all = np.zeros((N, K, T))
    for i in range(N):
        exp_lambda = np.exp(lambda_values_np[i, :, :])
        theta_all[i, :, :] = exp_lambda / np.sum(exp_lambda, axis=0, keepdims=True)
    
    # Population statistics
    theta_pop_mean = np.mean(theta_all, axis=0)  # (K, T)
    theta_pop_std = np.std(theta_all, axis=0)    # (K, T)

    # Individual theta
    theta_individual = theta_all[individual_idx]  # (K, T)

    # Get raw phi values for the target disease
    phi_disease = phi_values_np[:, disease_idx, :]  # (K, T)
    
    # Calculate contributions using sigmoid of phi
    phi_probs_disease = 1 / (1 + np.exp(-phi_disease))
    contributions = theta_individual * phi_probs_disease
    total_latent_risk = np.sum(contributions, axis=0)  # (T,)
    total_prob_disease = kappa * total_latent_risk     # (T,)
    total_prob_disease = np.minimum(1.0, np.maximum(0.0, total_prob_disease))

    # Scale contributions
    scaled_contributions = contributions.copy()  # (K, T)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.divide(total_prob_disease, total_latent_risk)  # (T,)
        scaling_factor = np.where(np.isfinite(scaling_factor), scaling_factor, 0.0)  # (T,)
        scaled_contributions = scaled_contributions * scaling_factor[None, :]  # Broadcasting (T,) to (K, T)
    if signature_names is None:
        signature_names = [f"Signature {k}" for k in range(K)]

    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.2, 2])
    # Use our consistent color palette
    palette = get_signature_colors(K)

    # --- Panel 1: Population and Individual Signature Loadings (theta) ---
    ax1 = fig.add_subplot(gs[0])
    for k in range(K):
        ax1.plot(time_points, theta_pop_mean[k], '--', color=palette[k], 
                 alpha=0.5, label=f'Pop. Mean {signature_names[k]}')
        ci = theta_pop_std[k]
        ax1.fill_between(time_points, 
                        theta_pop_mean[k] - ci,
                        theta_pop_mean[k] + ci,
                        color=palette[k], alpha=0.1)
        ax1.plot(time_points, theta_individual[k], '-', color=palette[k],
                 linewidth=2, label=f'Individual {signature_names[k]}')

    ax1.set_title(f'Signature Loadings (θ): Individual {individual_idx} vs Population', pad=15)
    ax1.set_ylabel('Loading (θ)')
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # --- Panel 2: Signature-Disease Association Heatmap ---
    ax2 = fig.add_subplot(gs[1])
    df_heatmap = pd.DataFrame(
        phi_disease,
        index=signature_names,
        columns=time_points
    )
    # Use a consistent colormap for the heatmap that works well with our color scheme
    sns.heatmap(df_heatmap, 
                cmap='RdBu_r',
                ax=ax2,
                cbar_kws={'label': 'log-odds(Disease | Signature k, Age t)'})
    ax2.set_title(f'Temporal Signature Associations for {disease_name}', pad=15)
    ax2.set_xlabel('')  # Remove x-label since it's shared
    ax2.set_ylabel('Signature')
    ax2.tick_params(rotation=0)

    # --- Panel 3: Stacked Contribution to Disease Probability ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    # Sort contributions by their maximum value to ensure consistent stacking order
    max_contribs = np.max(scaled_contributions, axis=1)
    sorted_indices = np.argsort(max_contribs)
    sorted_contributions = scaled_contributions[sorted_indices]
    sorted_names = [signature_names[i] for i in sorted_indices]
    sorted_colors = [palette[i] for i in sorted_indices]
    
    ax3.stackplot(time_points, sorted_contributions, labels=sorted_names, 
                 colors=sorted_colors, alpha=0.7)
    ax3.plot(time_points, total_prob_disease, 'k--', linewidth=2, 
             label='Total Risk ($\pi_{idt}$)')
    ax3.set_title(f'Signature Contributions to Risk of "{disease_name}"', pad=15)
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Risk ($\pi$)')
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    plot_max_y = (np.max(total_prob_disease) * 1.1) if np.any(total_prob_disease > 0) else 0.1
    ax3.set_ylim(0, plot_max_y)

    plt.suptitle(f"Probability Breakdown for '{disease_name}' - Individual {individual_idx}", 
                fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_path}")
        plt.close(fig)
    else:
        plt.show()

def analyze_age_onset_patterns_across_batches(
    results_base_dir: str,
    disease_index: int = 114,
    early_threshold: int = 55,
    late_threshold: int = 65,
    output_path: Optional[str] = None
):
    """
    Optimized version that loads each batch file only once
    """
    batch_dirs = find_batch_dirs(results_base_dir)
    if not batch_dirs:
        print("Error: No batch directories found. Cannot proceed.")
        return [], []
    
    print(f"\n--- Scanning {len(batch_dirs)} Batches for Early/Late Onset Patterns ---")
    
    # Initialize storage
    all_early_onset = []  # (global_idx, lambda_values, diagnosis_age)
    all_late_onset = []
    total_patients_scanned = 0
    
    # Process each batch - load data only once per batch
    for batch_info in batch_dirs:
        batch_model_path = os.path.join(batch_info['path'], 'model.pt')
        if not os.path.exists(batch_model_path):
            continue
            
        try:
            # Load batch data once
            print(f"Processing batch {batch_info['name']}...")
            model_data = torch.load(batch_model_path, map_location='cpu')
            Y_batch = model_data['Y']
            lambda_batch = model_data['model_state_dict']['lambda_']
            
            # Convert to numpy
            if torch.is_tensor(Y_batch):
                Y_batch = Y_batch.cpu().numpy()
            if torch.is_tensor(lambda_batch):
                lambda_batch = lambda_batch.cpu().numpy()
            
            current_N = Y_batch.shape[0]
            total_patients_scanned += current_N
            
            # Process all patients in this batch
            for local_idx in range(current_N):
                diagnosis_times = np.where(Y_batch[local_idx, disease_index] > 0.5)[0]
                if len(diagnosis_times) > 0:
                    diagnosis_time = diagnosis_times[0]
                    diagnosis_age = 30 + diagnosis_time
                    global_idx = batch_info['start'] + local_idx
                    
                    # Store lambda values with the patient info
                    lambda_values = lambda_batch[local_idx]
                    if diagnosis_age < early_threshold:
                        all_early_onset.append((global_idx, lambda_values, diagnosis_age))
                    elif diagnosis_age > late_threshold:
                        all_late_onset.append((global_idx, lambda_values, diagnosis_age))
                        
        except Exception as e:
            print(f"Error processing batch {batch_info['name']}: {e}")
            continue
    
    print(f"\nFound across {total_patients_scanned} patients:")
    print(f"- {len(all_early_onset)} early-onset cases (<{early_threshold} years)")
    print(f"- {len(all_late_onset)} late-onset cases (>{late_threshold} years)")
    
    if not all_early_onset or not all_late_onset:
        print("Insufficient data for analysis")
        return [], []
    
    # Calculate statistics and prepare plotting data
    early_ages = [age for _, _, age in all_early_onset]
    late_ages = [age for _, _, age in all_late_onset]
    print(f"\nEarly onset mean age: {np.mean(early_ages):.1f} (range: {min(early_ages):.1f}-{max(early_ages):.1f})")
    print(f"Late onset mean age: {np.mean(late_ages):.1f} (range: {min(late_ages):.1f}-{max(late_ages):.1f})")
    
    # Get dimensions from the stored lambda values
    K = all_early_onset[0][1].shape[0]
    T = all_early_onset[0][1].shape[1]
    time_points = np.arange(30, 30 + T)
    
    # Calculate theta values directly from stored lambda values
    early_theta = np.zeros((len(all_early_onset), K, T))
    late_theta = np.zeros((len(all_late_onset), K, T))
    
    for i, (_, lambda_values, _) in enumerate(all_early_onset):
        exp_lambda = np.exp(lambda_values)
        early_theta[i] = exp_lambda / np.sum(exp_lambda, axis=0)
    
    for i, (_, lambda_values, _) in enumerate(all_late_onset):
        exp_lambda = np.exp(lambda_values)
        late_theta[i] = exp_lambda / np.sum(exp_lambda, axis=0)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    def plot_group_patterns(theta_values, mean_age, axes, title_prefix):
        ax_prop, ax_vel = axes
        
        # Get consistent colors
        colors = get_signature_colors(K)
        
        # Calculate mean and SEM
        mean_theta = np.mean(theta_values, axis=0)
        sem_theta = np.std(theta_values, axis=0) / np.sqrt(len(theta_values))
        
        # Calculate velocities
        velocities = np.gradient(mean_theta, axis=1)
        vel_sem = np.std(np.gradient(theta_values, axis=2), axis=0) / np.sqrt(len(theta_values))
        
        # Plot proportions
        for k in range(K):
            color = colors[k]
            ax_prop.plot(time_points, mean_theta[k], label=f'Signature {k}', color=color)
            ax_prop.fill_between(time_points, 
                               mean_theta[k] - sem_theta[k],
                               mean_theta[k] + sem_theta[k],
                               color=color, alpha=0.2)
        
        # Plot velocities
        for k in range(K):
            color = colors[k]
            ax_vel.plot(time_points, velocities[k], label=f'Signature {k}', color=color)
            ax_vel.fill_between(time_points,
                              velocities[k] - vel_sem[k],
                              velocities[k] + vel_sem[k],
                              color=color, alpha=0.2)
        
        # Add vertical lines for diagnosis age
        ax_prop.axvline(mean_age, color='red', linestyle=':', label=f'Avg Diagnosis Age: {mean_age:.1f}')
        ax_vel.axvline(mean_age, color='red', linestyle=':', label=f'Avg Diagnosis Age: {mean_age:.1f}')
        
        # Customize plots
        ax_prop.set_title(f'{title_prefix}\nProportions')
        ax_vel.set_title(f'{title_prefix}\nVelocities')
        ax_prop.set_ylabel('Average Signature Loading')
        ax_vel.set_ylabel('Velocity')
        ax_prop.grid(True, alpha=0.3)
        ax_vel.grid(True, alpha=0.3)
    
    # Plot patterns
    plot_group_patterns(
        early_theta, 
        np.mean(early_ages),
        (ax1, ax2),
        f'Early Onset (<{early_threshold}, n={len(all_early_onset)})'
    )
    
    plot_group_patterns(
        late_theta,
        np.mean(late_ages),
        (ax3, ax4),
        f'Late Onset (>{late_threshold}, n={len(all_late_onset)})'
    )
    
    # Add legend and adjust layout
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.5), loc='center left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_path}")
        plt.close()
    else:
        plt.show()
    
    return [idx for idx, _, _ in all_early_onset], [idx for idx, _, _ in all_late_onset]

def plot_disease_signature_clusters_all_batches(disease_idx, batch_size=10000, n_batches=10, n_clusters=3, n_top_sigs=20, subtract_reference=True, output_path=None):
    """
    Plot signature proportion deviations by cluster for a specific disease across all batches
    
    Args:
        disease_idx: Index of the disease to analyze
        batch_size: Number of patients per batch
        n_batches: Number of batches to process
        n_clusters: Number of clusters to use
        n_top_sigs: Number of top signatures to show
        subtract_reference: Whether to subtract reference trajectories
        output_path: Path to save the PDF file. If None, plot is shown but not saved.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import traceback
    
    print(f"Starting analysis for disease {disease_idx} across all batches...")
    
    # Load batch 1 model to get disease names and signature information
    model_path = '/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt'
    print("\nLoading batch 1 model for reference...")
    
    try:
        model1 = torch.load(model_path)
        disease_names = model1['disease_names'][0].tolist()
        K_total = model1['model_state_dict']['lambda_'].shape[1]  # Number of signatures
        time_points = model1['Y'].shape[2]  # Number of time points
        
        disease_name = disease_names[disease_idx] if disease_idx < len(disease_names) else f"Disease {disease_idx}"
        print(f"Analyzing {disease_name}")
        
        # Get reference trajectories from first batch
        if 'signature_refs' in model1:
            if model1.get('healthy_ref') is not None:
                # Create full reference including healthy state
                reference_lambda = torch.zeros((K_total, time_points))
                reference_lambda[:-1] = model1['signature_refs']  # Disease signatures
                reference_lambda[-1] = model1['healthy_ref']  # Healthy reference
                # Convert to proportions using softmax
                reference_theta = torch.softmax(reference_lambda, dim=0).detach().numpy()
            else:
                reference_theta = torch.softmax(model1['signature_refs'], dim=0).detach().numpy()
        else:
            # If no reference, we'll calculate population average later
            reference_theta = None
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        traceback.print_exc()
        return None
    
    # Collect patients with this disease across all batches
    all_patients = []  # Will store (batch_idx, patient_idx) tuples
    all_features = []  # Will store signature proportions
    all_thetas = []    # Will store full theta matrices
    
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
            
            # Find patients with this disease
            for i in range(Y_batch.shape[0]):
                if np.any(Y_batch[i, disease_idx]):
                    # Get signature proportions
                    theta = torch.softmax(lambda_values[i], dim=0).detach().numpy()
                    mean_props = theta.mean(axis=1)
                    
                    # Store patient data
                    all_patients.append((batch, i))
                    all_features.append(mean_props)
                    all_thetas.append(theta)
            
            print(f"Found {len(all_patients) - (0 if batch == 0 else sum(1 for p in all_patients if p[0] < batch))} patients in batch {batch+1}")
            
        except FileNotFoundError:
            print(f"Warning: Could not find model file for batch {batch}")
            break  # Assume we've reached the end of batches
        except Exception as e:
            print(f"Error processing batch {batch}: {str(e)}")
            traceback.print_exc()
            continue
    
    if len(all_patients) == 0:
        print(f"No patients found with disease {disease_idx}")
        return None
    
    print(f"\nTotal patients with {disease_name}: {len(all_patients)}")
    
    # If no reference theta, use population average
    if reference_theta is None:
        reference_theta = np.mean(all_thetas, axis=0)
        print("Using population average as reference")
    
    # Convert to numpy arrays
    all_features = np.array(all_features)
    
    # Perform clustering
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    patient_clusters = kmeans.fit_predict(all_features)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 5*n_clusters))
    if n_clusters == 1:
        axes = [axes]  # Make axes iterable if only one cluster
    
    time_points = np.arange(time_points)
    
    for cluster_idx in range(n_clusters):
        # Get patients in this cluster
        cluster_indices = np.where(patient_clusters == cluster_idx)[0]
        cluster_thetas = [all_thetas[i] for i in cluster_indices]
        
        # Calculate average theta for this cluster
        avg_theta = np.mean(cluster_thetas, axis=0)
        
        if subtract_reference:
            # Subtract reference from average
            avg_theta = avg_theta - reference_theta
        
        # Calculate mean deviation for each signature
        mean_props = np.abs(avg_theta).mean(axis=1)
        
        # Get top signatures by absolute mean deviation
        top_sig_idx = np.argsort(mean_props)[-n_top_sigs:][::-1]
        
        # Create stacked area plot
        bottom_pos = np.zeros(time_points.shape)
        bottom_neg = np.zeros(time_points.shape)
        colors = get_signature_colors(n_top_sigs)  # Use our consistent color scheme
        
        for i, sig in enumerate(top_sig_idx):
            values = avg_theta[sig]
            if subtract_reference:
                # For deviations, split positive and negative
                pos_values = np.maximum(values, 0)
                neg_values = np.minimum(values, 0)
                
                axes[cluster_idx].fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                                             label=f'Sig {sig} (Δθ={mean_props[sig]:.3f})',
                                             color=colors[i])
                axes[cluster_idx].fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                                             color=colors[i], alpha=0.5)
                
                bottom_pos += pos_values
                bottom_neg += neg_values
            else:
                axes[cluster_idx].fill_between(time_points, bottom_pos, bottom_pos + values,
                                             label=f'Sig {sig} (θ={mean_props[sig]:.3f})',
                                             color=colors[i])
                bottom_pos += values
        
        title = f'Cluster {cluster_idx}: '
        title += 'Signature Proportion Deviations from Population Average' if subtract_reference else 'Average Signature Proportions'
        title += f'\n({len(cluster_indices)} patients)'
        
        axes[cluster_idx].set_title(title)
        axes[cluster_idx].set_xlabel('Time (Age 30-81)')
        axes[cluster_idx].set_ylabel('Δ Proportion (θ)' if subtract_reference else 'Proportion (θ)')
        axes[cluster_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if subtract_reference:
            # Center y-axis around 0 for deviations
            max_dev = max(abs(bottom_pos.max()), abs(bottom_neg.min()))
            axes[cluster_idx].set_ylim(-max_dev, max_dev)
            axes[cluster_idx].axhline(y=0, color='k', linestyle='-', alpha=0.2)
        else:
            axes[cluster_idx].set_ylim(0, 1.05)
    
    plt.suptitle(f'{"Signature Proportion Deviations" if subtract_reference else "Average Signature Proportions"}\n'
                f'by Cluster for {disease_name}')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig



def plot_single_patient_dynamics(
    model_path: str,
    patient_idx: int,
    sig_refs_path: str,
    figsize=(14, 7)
):
    """
    Plot dynamics for a single patient, similar to plot_multi_morbid_dynamics_from_batches
    but simplified for one patient.
    
    Args:
        model_path: Path to the model.pt file containing the patient
        patient_idx: Local index of patient in the batch
        sig_refs_path: Path to signature references
        figsize: Figure size
    """
    # Load patient data
    model_data = torch.load(model_path, map_location='cpu')
    Y_batch = model_data['Y']
    lambda_batch = model_data['model_state_dict']['lambda_']
    psi = model_data['model_state_dict']['psi']
    
    # Get patient's data
    patient_Y = Y_batch[patient_idx]
    patient_lambda = lambda_batch[patient_idx]
    
    # Convert to numpy if needed
    if torch.is_tensor(patient_Y):
        patient_Y = patient_Y.detach().cpu().numpy()
    if torch.is_tensor(patient_lambda):
        patient_lambda = patient_lambda.detach().cpu().numpy()
    if torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
    
    # Calculate theta
    exp_lambda = np.exp(patient_lambda)
    theta_patient = exp_lambda / np.sum(exp_lambda, axis=0)
    
    # Get disease names if available
    disease_names = model_data.get('disease_names', None)
    if disease_names is not None and torch.is_tensor(disease_names):
        disease_names = disease_names.tolist()
    if disease_names is None:
        disease_names = [f"Disease {i}" for i in range(patient_Y.shape[0])]
    
    # Find patient's conditions
    conditions = []
    for d_idx in range(patient_Y.shape[0]):
        diag_times = np.where(patient_Y[d_idx, :] > 0.5)[0]
        if len(diag_times) > 0:
            conditions.append((d_idx, diag_times[0]))
    
    # Create plot
    fig = plt.figure(figsize=(figsize[0] * 1.3, figsize[1]))
    gs = GridSpec(2, 2, height_ratios=[2.5, 1], width_ratios=[4, 0.8], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Top left: temporal theta
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Bottom left: timeline
    ax3 = fig.add_subplot(gs[:, 1])  # Right side: stacked bar
    
    # Get colors
    K = theta_patient.shape[0]
    colors = get_signature_colors(K)
    
    # Map diseases to primary signatures
    disease_primary_sig = {}
    for d_idx in range(psi.shape[1]):
        primary_sig = np.argmax(psi[:, d_idx])
        disease_primary_sig[d_idx] = primary_sig
    
    # Plot trajectories
    legend_handles = []
    legend_labels = []
    signatures_to_plot = set()
    
    for d_idx, diag_time_idx in conditions:
        sig_idx = disease_primary_sig[d_idx]
        signatures_to_plot.add(sig_idx)
        disease_name = disease_names[d_idx]
        color = colors[sig_idx]
        
        # Plot trajectory
        line = ax1.plot(np.arange(theta_patient.shape[1]), 
                       theta_patient[sig_idx, :],
                       color=color, linewidth=2.5, alpha=0.95,
                       zorder=3)[0]
        
        # Mark diagnosis
        ax1.axvline(x=diag_time_idx, color=color, linestyle=':',
                   alpha=0.5, linewidth=0.8, zorder=2)
        
        legend_handles.append(line)
        legend_labels.append(f"Sig {sig_idx} ({disease_name[:20]}{'...' if len(disease_name)>20 else ''})")
    
    # Configure plots
    ax1.set_title(f'Signature Loadings (Θ) Over Time\nPatient {patient_idx}', fontsize=14, pad=10)
    ax1.set_ylabel('Signature Loading (Θ)', fontsize=12)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(bottom=0)
    
    # Timeline
    ax2.set_yticks(range(len(conditions)))
    ax2.set_yticklabels([disease_names[d_idx] for d_idx, _ in conditions], fontsize='small')
    ax2.set_ylim(-0.5, len(conditions) - 0.5)
    ax2.invert_yaxis()
    
    for i, (d_idx, diag_time) in enumerate(conditions):
        sig_idx = disease_primary_sig[d_idx]
        color = colors[sig_idx]
        ax2.scatter(diag_time, i, marker='o', s=40,
                   color=color, zorder=5,
                   edgecolors='black', linewidth=0.5)
        ax2.hlines(i, 0, diag_time,
                  colors=color, linestyles='-',
                  alpha=0.4, linewidth=1.0)
    
    ax2.set_xlabel('Age (years)', fontsize=12)
    ax2.set_ylabel('Diagnosed Condition', fontsize=12)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Static summary
    theta_avg = np.mean(theta_patient, axis=1)
    bottom = 0
    sig_indices = sorted(list(signatures_to_plot))
    
    for sig_idx in sig_indices:
        color = colors[sig_idx]
        ax3.bar([0], [theta_avg[sig_idx]], bottom=bottom,
                color=color, alpha=0.7, width=0.3)
        bottom += theta_avg[sig_idx]
    
    ax3.set_title('Static Model\nSummary', pad=15)
    ax3.set_ylabel('Average Loading (θ)')
    ax3.set_xlim(-0.3, 0.3)
    ax3.set_xticks([])
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    fig.legend(legend_handles, legend_labels,
              loc='center right', bbox_to_anchor=(0.98, 0.5),
              fontsize='small')
    
    plt.tight_layout(rect=[0, 0, 0.82, 0.97])
    
    return fig

