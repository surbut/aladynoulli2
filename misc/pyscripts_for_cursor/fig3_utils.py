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
                                            figsize_per_patient=(14, 7)):
    """
    Loads reference signature trajectories from signature_refs_path.
    Scans model batches (*.pt files in output_* dirs) to find multi-morbid
    patients (that have at least one target disease and multiple conditions).
    Maps ALL patient conditions to signatures using the full psi matrix.
    Plots their signature dynamics (theta) relative to diagnoses and 
    the loaded reference signature trajectories.

    Args:
        results_base_dir: Directory containing output_* subdirectories with model.pt files
        target_disease_indices: List of disease indices to search for (patients must have ≥1)
        signature_refs_path: Path to .pt file containing {'signature_refs': tensor/array of shape K x T}
        min_conditions: Minimum number of total diseases a patient must have
        n_patients_to_plot: Number of randomly selected patients to plot
        random_seed: For reproducibility
        figsize_per_patient: Figure size for each patient plot

    Returns:
        essentials: Dictionary containing reference data
        patient_figures: List of matplotlib figure objects
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
                        # Check if this is one of our target diseases
                        if d_idx in target_disease_indices:
                            has_target_disease = True
                
                # If no target diseases or not enough conditions, skip this patient
                if not has_target_disease or len(all_conditions) < min_conditions:
                    continue
                    
                # If patient has enough total conditions AND at least one target, add to the pool
                # Sort conditions by diagnosis time index
                all_conditions.sort(key=lambda x: x[1])
                global_idx = batch_info['start'] + i  # Calculate global patient index
                found_patients_pool.append((batch_info, i, all_conditions, global_idx))

        except Exception as e_batch:
            print(f"  Error processing batch {batch_info['name']}: {e_batch}")

    if not found_patients_pool:
        print(f"\nScan complete. No patients found across {total_patients_scanned} scanned patients meeting the criteria (>= {min_conditions} total conditions AND at least 1 target disease).")
        return essentials, []

    print(f"\nScan complete. Found {len(found_patients_pool)} suitable patients across {total_patients_scanned} scanned patients.")   
    
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

            # --- Create the 2-panel plot ---
            fig = plt.figure(figsize=figsize_per_patient)
            gs = GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.05)  # Reduced space between plots
            ax1 = fig.add_subplot(gs[0])  # Top panel for theta
            ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Bottom panel for timeline

            # Panel 1: Theta trajectories
            plotted_signatures = set()  # Keep track of signatures already plotted (for pop avg label)
            
            # Replace the theta plotting section with this improved code:
            # First, collect valid conditions and prepare
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

            print(f"  Found {len(valid_conditions)} out of {len(conditions)} conditions with valid signature mappings.")

            # Use a distinct colormap
            num_conditions = len(valid_conditions)
            cmap_name = 'viridis' if num_conditions > 10 else 'tab10'
            colors = plt.cm.get_cmap(cmap_name, max(10, num_conditions))

            # Prepare condition details for timeline
            condition_details = []
            signatures_to_plot = set()

            # Collect all signatures and prepare condition details
            for i, (d_idx, diag_time_idx, sig_idx) in enumerate(valid_conditions):
                signatures_to_plot.add(sig_idx)
                disease_name = disease_names_list[d_idx] if 0 <= d_idx < len(disease_names_list) else f"Disease {d_idx}"
                diag_time = time_points[diag_time_idx]
                color = colors(i / max(1, num_conditions - 1)) if num_conditions > 1 else colors(0)
                condition_details.append({
                    'name': disease_name,
                    'time': diag_time,
                    'color': color,
                    'y_pos': i,
                    'sig_idx': sig_idx,
                    'd_idx': d_idx
                })

            # FIRST: Plot ALL population averages (with dashed lines)
            '''
            for sig_idx in signatures_to_plot:
                if sig_idx in pop_avg_theta:
                    # Find conditions with this signature
                    matching_conditions = [c for c in condition_details if c['sig_idx'] == sig_idx]
                    if matching_conditions:
                        color = matching_conditions[0]['color']
                        label = f"Pop Avg Sig {sig_idx}"
                        ax1.plot(time_points, pop_avg_theta[sig_idx],
                                color=color, linestyle='--', linewidth=2.0, 
                                alpha=0.7, zorder=1, label=label)
           '''

            # SECOND: Plot patient-specific trajectories (with solid lines)
            for i, detail in enumerate(condition_details):
                d_idx = detail['d_idx']
                sig_idx = detail['sig_idx']
                disease_name = detail['name']
                diag_time = detail['time']
                color = detail['color']
                
                # Truncate long disease names for the label
                label_text_short = f"{disease_name[:20]}{'...' if len(disease_name)>20 else ''}"
                label_patient = f"Sig {sig_idx} ({label_text_short})"
                
                # Plot patient's trajectory with HIGHER zorder
                ax1.plot(time_points, theta_patient[sig_idx, :],
                        color=color, linewidth=2.5, alpha=0.95, 
                        zorder=3, label=label_patient)
                
                # Mark diagnosis time
                ax1.axvline(x=diag_time, color=color, linestyle=':', 
                            alpha=0.7, linewidth=1.5, zorder=2)
                
                # Find and mark the peak
                peak_idx = np.argmax(theta_patient[sig_idx, :])
                peak_time = time_points[peak_idx]
                peak_value = theta_patient[sig_idx, peak_idx]
                ax1.scatter(peak_time, peak_value, marker='*', s=120, 
                            color=color, edgecolor='black', linewidth=1, 
                            zorder=4, label=None)

            # Configure theta plot (ax1)
            ax1.set_title(f'Figure 3 Example: Signature Loadings ($\\Theta$) Over Time\nPatient {global_patient_idx}', fontsize=14, pad=10)
            ax1.set_ylabel('Signature Loading ($\\Theta$)', fontsize=12)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.5)  # Grid lines only horizontal
            ax1.legend(loc='upper left', fontsize='x-small', ncol=2)  # Adjust legend position/size
            ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-axis ticks and labels
            ax1.set_ylim(bottom=0)  # Ensure y-axis starts at 0

            # Panel 2: Disease Timeline
            ax2.set_yticks(range(num_conditions))
            # Use full names for y-tick labels on timeline
            ax2.set_yticklabels([d['name'] for d in condition_details], fontsize='small')
            ax2.set_ylim(-0.5, num_conditions - 0.5)
            ax2.invert_yaxis()  # Show first diagnosis at the top

            for detail in condition_details:
                # Mark diagnosis point
                ax2.scatter(detail['time'], detail['y_pos'], marker='o', s=80, color=detail['color'], zorder=5, label=None, edgecolors='black', linewidth=0.5)
                # Draw line up to diagnosis
                ax2.hlines(detail['y_pos'], time_points[0], detail['time'], colors=detail['color'], linestyles='-', alpha=0.6, linewidth=1.5)

            # Configure timeline plot (ax2)
            ax2.set_xlabel('Age (years)', fontsize=12)
            ax2.set_ylabel('Diagnosed Condition', fontsize=12)
            ax2.grid(True, axis='x', linestyle='--', alpha=0.5)  # Grid lines only vertical
            ax2.spines['top'].set_visible(False)  # Remove top spine
            ax2.spines['right'].set_visible(False)  # Remove right spine

            # Adjust layout and display
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add space for main title
            plt.show()
            patient_figures.append(fig)  # Store the figure object

        except Exception as e_plot:
            print(f"  Error generating plot for Patient {global_patient_idx}: {e_plot}")
            traceback.print_exc()

    print(f"\nFinished plotting {len(patient_figures)} multi-morbid dynamics examples.")
    # Return the essentials dict (containing reference data) and the list of generated figures
    return essentials, patient_figures

# Example Usage
results_dir = '/Users/sarahurbut/Dropbox/resultshighamp/results/'
sig_refs_file = '/Users/sarahurbut/Dropbox/data_for_running/reference_trajectories.pt'

target_diseases = [112, 127, 66, 17,74]  # Example indices - verify these match your disease codes of interest
min_cond = 8 # Minimum number of total diseases a patient must have
n_plots = 10  # How many example patients to plot

# Check if required files/directories exist before running
if not os.path.isdir(results_dir):
    print(f"ERROR: Results directory not found: {results_dir}")
elif not os.path.exists(sig_refs_file):
    print(f"ERROR: Signature reference file not found: {sig_refs_file}")
else:
    # Run the analysis and plotting
    essentials_dict, figures_list = plot_multi_morbid_dynamics_from_batches(
        results_base_dir=results_dir,
        target_disease_indices=target_diseases,
        signature_refs_path=sig_refs_file,
        min_conditions=min_cond,
        n_patients_to_plot=n_plots
    )

    # You can access reference data via essentials_dict
    # To save the figures:
    # for i, fig in enumerate(figures_list):
    #     fig.savefig(f'figure3_patient_example_{i+1}.png', dpi=300)




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


        

model_file = '/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt' # Example path

    # --- Parameters to set ---
    # SELECT an individual and disease index to plot
    # Find interesting examples by looking at Y_observed or model outputs
example_individual_index = 100 # Choose an patient index (0 to N-1)
    # Find disease index for 'Coronary atherosclerosis' (or another disease)
    # This requires loading the names first, or knowing the index beforehand.
    # Let's assume index 114 corresponds to it based on previous logs.
example_disease_index = 114

    # SET the KAPPA value - this is crucial for scaling the risk
    # Load it from model_data if saved, e.g., model_data['model_state_dict']['kappa'].item()
    # Or use the value from training. Defaulting to 1.0 is likely incorrect.

    # <<< !!! VERIFY this KAPPA value is appropriate !!! >>>
kappa = model.kappa

    # Define output filename
output_file = 'mix.png'

    # --- Run the visualization ---
run_mixture_visualization_on_real_data(model_file,
                                           example_individual_index,
                                           example_disease_index,
                                           kappa_value=kappa,
                                           output_filename=output_file)

