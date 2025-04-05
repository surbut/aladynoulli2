import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional

# Import the signature evolution plotting function from the other file
from plot_signature_evolution import plot_signature_evolution

def generate_synthetic_mi_patients(
    n_patients: int = 4,
    time_span: int = 15,
    pre_mi_years: int = 3,
    signature_names: List[str] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Generates synthetic patient data for a cohort of patients who develop MI.
    
    Parameters:
    -----------
    n_patients : int
        Number of patients to generate
    time_span : int
        Total years of follow-up
    pre_mi_years : int
        Years before MI when signature association begins to increase
    signature_names : List[str]
        Names of signatures to include (default: cardiovascular, metabolic, healthy)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    patients : List[Dict]
        List of patient data dictionaries
    """
    np.random.seed(seed)
    
    if signature_names is None:
        signature_names = ["Cardiovascular (Sig 1)", "Metabolic (Sig 2)", "Healthy (Sig 0)"]
    
    patients = []
    
    for i in range(n_patients):
        # Randomly determine when MI occurs (between year 8 and year 13)
        mi_year = np.random.randint(8, 14)
        
        # Generate time points
        time_points = np.arange(0, time_span)
        
        # Initialize theta values for each signature
        theta_values = {}
        
        # Generate cardiovascular signature trajectory
        # Start with low values, then increase before MI
        cv_theta = np.zeros(time_span)
        
        # Base trajectory before increase
        base_level = np.random.uniform(0.05, 0.15)  # Starting level varies by patient
        cv_theta[:mi_year-pre_mi_years] = np.linspace(base_level, base_level + 0.05, mi_year-pre_mi_years)
        
        # Accelerating increase before MI
        increase_points = np.linspace(cv_theta[mi_year-pre_mi_years], 0.75 + np.random.uniform(-0.1, 0.1), pre_mi_years+1)
        cv_theta[mi_year-pre_mi_years:mi_year+1] = increase_points
        
        # After MI (if any time remains)
        if mi_year < time_span - 1:
            cv_theta[mi_year+1:] = 0.8  # Remain high after MI
            
        theta_values[signature_names[0]] = cv_theta  # Cardiovascular signature
        
        # Generate metabolic signature (varies by patient)
        if len(signature_names) > 1:
            metabolic_level = np.random.uniform(0.15, 0.35)
            metabolic_theta = np.linspace(metabolic_level, metabolic_level + np.random.uniform(-0.15, 0.15), time_span)
            # Add some random variation
            metabolic_theta += np.random.normal(0, 0.03, size=time_span)
            # Ensure values stay reasonable
            metabolic_theta = np.clip(metabolic_theta, 0.05, 0.5)
            theta_values[signature_names[1]] = metabolic_theta  # Metabolic signature
        
        # Generate healthy signature (inverse of cardiovascular + metabolic)
        if len(signature_names) > 2:
            healthy_theta = 1.0 - cv_theta
            if len(signature_names) > 1:
                healthy_theta -= metabolic_theta * 0.3  # Partial contribution from metabolic
            healthy_theta = np.clip(healthy_theta, 0.0, 0.95)  # Ensure values stay reasonable
            theta_values[signature_names[2]] = healthy_theta  # Healthy signature
        
        # Generate clinical events
        clinical_events = []
        
        # Always include the MI event
        clinical_events.append({
            'time': mi_year,
            'event': 'Myocardial Infarction',
            'color': 'red',
            'signature': signature_names[0],  # Cardiovascular signature
            'theta': cv_theta[mi_year]
        })
        
        # Add preceding events (varies by patient)
        
        # Hypertension (70-90% probability, 1-3 years before MI)
        if np.random.random() < 0.8:
            htn_year = mi_year - np.random.randint(1, 4)
            if htn_year >= 0:
                clinical_events.append({
                    'time': htn_year,
                    'event': 'Hypertension',
                    'color': 'skyblue',
                    'signature': signature_names[0],
                    'theta': cv_theta[htn_year]
                })
        
        # Hyperlipidemia (50-70% probability, 1-3 years before MI)
        if np.random.random() < 0.6:
            lipid_year = mi_year - np.random.randint(1, 4)
            if lipid_year >= 0:
                clinical_events.append({
                    'time': lipid_year,
                    'event': 'Hyperlipidemia',
                    'color': 'orange',
                    'signature': signature_names[0],
                    'theta': cv_theta[lipid_year]
                })
        
        # Abnormal test (30-50% probability, 0-2 years before MI)
        if np.random.random() < 0.4:
            test_year = mi_year - np.random.randint(0, 3)
            if test_year >= 0:
                clinical_events.append({
                    'time': test_year,
                    'event': 'Abnormal Test',
                    'color': 'purple',
                    'signature': signature_names[0],
                    'theta': cv_theta[test_year]
                })
        
        # Sort events by time
        clinical_events.sort(key=lambda x: x['time'])
        
        # Create patient data dictionary
        patient_data = {
            'id': f"Patient {i+1}",
            'age': np.random.randint(45, 75),
            'sex': np.random.choice(['Male', 'Female']),
            'time_points': time_points,
            'theta_values': theta_values,
            'clinical_events': clinical_events,
            'mi_year': mi_year
        }
        
        patients.append(patient_data)
    
    return patients

def plot_mi_figure6a(
    patients: List[Dict],
    figsize: Tuple[int, int] = (14, 10),
    save_path: str = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Creates a multi-panel figure showing multiple MI patient trajectories,
    similar to Figure 6A in the paper.
    
    Parameters:
    -----------
    patients : List[Dict]
        List of patient data dictionaries, each containing:
        - id: patient identifier
        - age: patient age
        - sex: patient sex
        - time_points: time points for the x-axis
        - theta_values: dictionary mapping signature names to association values
        - clinical_events: list of clinical events
        - mi_year: year of MI event
    figsize : Tuple[int, int], optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    show_plot : bool, optional
        Whether to display the plot
        
    Returns:
    --------
    fig : plt.Figure
        The created figure object
    """
    # Determine the layout (rows and columns)
    n_patients = len(patients)
    n_cols = min(2, n_patients)
    n_rows = (n_patients + n_cols - 1) // n_cols  # Ceiling division
    
    # Create the main figure
    fig = plt.figure(figsize=figsize)
    
    # Add an overall title
    fig.suptitle('Figure 6A: Signature Association Evolution Before Myocardial Infarction', 
                fontsize=16, y=0.98)
    
    # Create a subplot for each patient
    for i, patient in enumerate(patients):
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Get patient data
        patient_id = patient['id']
        age = patient['age']
        sex = patient['sex']
        time_points = patient['time_points']
        theta_values = patient['theta_values']
        clinical_events = patient['clinical_events']
        mi_year = patient['mi_year']
        
        # Number of years to show before/after MI
        pre_mi = 6  # Show 6 years before MI
        post_mi = 2  # Show 2 years after MI
        
        # Calculate the time window to show
        start_year = max(0, mi_year - pre_mi)
        end_year = min(len(time_points), mi_year + post_mi + 1)
        
        # Slice the time points and theta values
        time_window = time_points[start_year:end_year]
        theta_window = {sig: values[start_year:end_year] for sig, values in theta_values.items()}
        
        # Filter events to those within the time window
        events_window = [e for e in clinical_events if start_year <= e['time'] < end_year]
        
        # Determine if this is the MI event
        mi_event_idx = next((i for i, e in enumerate(events_window) if e['event'] == 'Myocardial Infarction'), None)
        
        # Custom colors for signatures
        colors = {
            "Cardiovascular (Sig 1)": "red",
            "Metabolic (Sig 2)": "green",
            "Healthy (Sig 0)": "blue"
        }
        
        # Plot simplified version for the multi-panel figure
        for sig_name, theta in theta_window.items():
            is_primary = "Cardiovascular" in sig_name
            ax.plot(
                time_window, theta, 
                color=colors.get(sig_name, 'blue'),
                linewidth=2 + (1 if is_primary else 0),
                alpha=0.9 if is_primary else 0.7,
                label=sig_name if is_primary else None  # Only label primary signature
            )
        
        # Mark MI event
        if mi_event_idx is not None:
            mi_event = events_window[mi_event_idx]
            ax.axvline(x=mi_event['time'], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(
                mi_event['time'] + 0.05, 
                0.5, 
                'MI', 
                rotation=90, 
                transform=ax.get_xaxis_transform(),
                verticalalignment='center', 
                color='red',
                fontsize=10
            )
        
        # Format subplot
        ax.set_title(f"{patient_id}: {age}y {sex}", fontsize=12)
        ax.set_ylabel('θ (Cardiovascular)' if i % n_cols == 0 else '')
        ax.set_xlabel('Years' if i >= (n_rows-1)*n_cols else '')
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only show legend for the first subplot
        if i == 0:
            ax.legend(loc='upper left', fontsize='small')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show figure if requested
    if show_plot:
        plt.show()
    
    return fig

def create_detailed_figure6b(save_path: str = None):
    """
    Creates a detailed version of Figure 6B showing one patient's signature 
    evolution leading up to MI with clinical events.
    """
    # Time points (years)
    time_points = np.arange(0, 13)
    
    # Signature association values
    theta_values = {
        "Cardiovascular (Sig 1)": np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.35, 0.58, 0.75]),
        "Metabolic (Sig 2)": np.array([0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.33, 0.32, 0.28, 0.20]),
        "Healthy (Sig 0)": np.array([0.80, 0.74, 0.70, 0.66, 0.60, 0.54, 0.50, 0.46, 0.42, 0.33, 0.14, 0.05])
    }
    
    # Clinical events
    clinical_events = [
        {
            'time': 9, 
            'event': 'Hypertension Diagnosed', 
            'color': 'skyblue',
            'signature': 'Cardiovascular (Sig 1)',
            'theta': 0.25
        },
        {
            'time': 10, 
            'event': 'Hyperlipidemia Diagnosed', 
            'color': 'orange',
            'signature': 'Cardiovascular (Sig 1)',
            'theta': 0.35
        },
        {
            'time': 11, 
            'event': 'Abnormal Stress Test', 
            'color': 'purple',
            'signature': 'Cardiovascular (Sig 1)',
            'theta': 0.58
        },
        {
            'time': 12, 
            'event': 'Myocardial Infarction', 
            'color': 'red',
            'signature': 'Cardiovascular (Sig 1)',
            'theta': 0.75
        }
    ]
    
    # Custom colors for signatures
    colors = {
        "Cardiovascular (Sig 1)": "red",
        "Metabolic (Sig 2)": "green",
        "Healthy (Sig 0)": "blue"
    }
    
    # Create the detailed plot
    fig = plot_signature_evolution(
        time_points=time_points,
        theta_values=theta_values,
        clinical_events=clinical_events,
        primary_signature="Cardiovascular (Sig 1)",
        title="Figure 6B: Signature Evolution Before Myocardial Infarction",
        subtitle="58-year-old Male Patient with Progressive Cardiovascular Risk",
        colors=colors,
        highlight_event_idx=3,  # Highlight the MI event
        save_path=save_path,
        show_plot=True
    )
    
    return fig

def create_full_figure6(n_patients: int = 4, save_path: str = None):
    """
    Creates a complete Figure 6 with both panels A and B.
    Panel A shows multiple patient trajectories.
    Panel B shows a detailed view of one specific patient.
    
    Parameters:
    -----------
    n_patients : int
        Number of patients to include in panel A
    save_path : str
        Path to save the complete figure
    """
    # Generate synthetic patient data
    patients = generate_synthetic_mi_patients(n_patients=n_patients)
    
    # Create Figure 6A (multi-patient overview)
    fig6a = plot_mi_figure6a(patients, show_plot=False)
    
    # Create Figure 6B (detailed single patient)
    fig6b = create_detailed_figure6b(save_path=None)
    
    # If you want to save both as one combined figure, you could create a new figure
    # and copy the contents of both figures into separate panels
    
    print("Created Figure 6A showing multiple patients and Figure 6B showing detailed analysis of one patient.")
    
    if save_path:
        fig6a.savefig(f"{save_path}_panel_A.png", dpi=300, bbox_inches='tight')
        fig6b.savefig(f"{save_path}_panel_B.png", dpi=300, bbox_inches='tight')
        print(f"Saved figures to {save_path}_panel_A.png and {save_path}_panel_B.png")
    
    return fig6a, fig6b

if __name__ == "__main__":
    # Generate and plot the complete Figure
    create_full_figure6(n_patients=4, save_path="figure6") 