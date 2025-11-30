import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional, Union

def plot_signature_evolution(
    time_points: np.ndarray,
    theta_values: Dict[str, np.ndarray],
    clinical_events: List[Dict],
    primary_signature: str = None,
    title: str = "Patient Signature Evolution",
    subtitle: str = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Dict[str, str] = None,
    event_marker_size: int = 100,
    line_width: float = 2.5,
    highlight_event_idx: int = None,
    ylim: Tuple[float, float] = (0, 1.0),
    save_path: str = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Creates a plot showing the evolution of signature associations leading up to a clinical event.
    
    Parameters:
    -----------
    time_points : np.ndarray
        Time points (e.g., years) for the x-axis
    theta_values : Dict[str, np.ndarray]
        Dictionary mapping signature names to their association values over time
    clinical_events : List[Dict]
        List of dictionaries, each with keys:
        - 'time': when the event occurred
        - 'event': name of the event
        - 'color': (optional) color for the event
        - 'signature': (optional) associated signature
        - 'theta': (optional) signature association value at event time
    primary_signature : str, optional
        Name of the primary signature to highlight
    title : str, optional
        Main title for the plot
    subtitle : str, optional
        Subtitle for the plot
    figsize : Tuple[int, int], optional
        Figure size (width, height)
    colors : Dict[str, str], optional
        Dictionary mapping signature names to colors
    event_marker_size : int, optional
        Size of event markers
    line_width : float, optional
        Width of signature trajectory lines
    highlight_event_idx : int, optional
        Index of event to highlight with a vertical line
    ylim : Tuple[float, float], optional
        Y-axis limits
    save_path : str, optional
        Path to save the figure
    show_plot : bool, optional
        Whether to display the plot
        
    Returns:
    --------
    fig : plt.Figure
        The created figure object
    """
    # Generate default colors if not provided
    if colors is None:
        cmap = cm.get_cmap('tab10', max(10, len(theta_values)))
        colors = {sig: cmap(i) for i, sig in enumerate(theta_values.keys())}
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])  # Top panel for theta trajectories
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Bottom panel for timeline
    
    # Plot each signature trajectory
    for sig_name, theta in theta_values.items():
        is_primary = (sig_name == primary_signature)
        ax1.plot(
            time_points[:len(theta)], theta, 
            color=colors.get(sig_name, 'blue'),
            linewidth=line_width + (1 if is_primary else 0),
            alpha=0.9 if is_primary else 0.7,
            label=f"{sig_name}{' (Primary)' if is_primary else ''}"
        )
    
    # Highlight data points at clinical events if theta values are provided
    for event in clinical_events:
        if 'signature' in event and 'theta' in event:
            event_time = event['time']
            sig_name = event['signature']
            theta_val = event['theta']
            
            # Only plot if the signature exists in our data
            if sig_name in colors:
                ax1.plot(
                    event_time, theta_val, 
                    'o', 
                    color=colors.get(sig_name, 'blue'),
                    markersize=8, 
                    zorder=5
                )
                
                # Add annotation for the theta value
                ax1.annotate(
                    f'θ = {theta_val}', 
                    xy=(event_time, theta_val), 
                    xytext=(5, 10),
                    textcoords='offset points', 
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7)
                )
    
    # Highlight a specific event with a vertical line if requested
    if highlight_event_idx is not None and 0 <= highlight_event_idx < len(clinical_events):
        event = clinical_events[highlight_event_idx]
        event_time = event['time']
        event_name = event['event']
        event_color = event.get('color', 'red')
        
        ax1.axvline(x=event_time, color=event_color, linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(
            event_time + 0.05, 
            ylim[1] * 0.5, 
            event_name, 
            rotation=90, 
            verticalalignment='center', 
            color=event_color,
            fontsize=10
        )
    
    # Configure top plot
    ax1.set_title(title, fontsize=14, pad=10)
    if subtitle:
        ax1.text(
            0.5, 0.98, 
            subtitle,
            transform=ax1.transAxes,
            ha='center', 
            va='top',
            fontsize=12, 
            style='italic'
        )
    
    ax1.set_ylabel('Signature Association (θ)', fontsize=12)
    ax1.set_ylim(*ylim)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # Remove x-axis labels from top plot
    ax1.tick_params(axis='x', labelsize=0)
    
    # Bottom plot: Timeline of clinical events
    for i, event in enumerate(clinical_events):
        event_time = event['time']
        event_name = event['event']
        event_color = event.get('color', colors.get(event.get('signature', ''), 'gray'))
        
        ax2.scatter(
            event_time, i, 
            color=event_color, 
            s=event_marker_size, 
            zorder=5,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax2.text(
            event_time + 0.1, i, 
            event_name, 
            verticalalignment='center',
            fontsize=10
        )
    
    # Configure bottom plot
    ax2.set_yticks([])
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_xlim(time_points[0], time_points[-1] + 0.5)
    ax2.set_ylim(-0.5, len(clinical_events) - 0.5)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig

def example_mi_patient():
    """
    Example demonstrating how to use the plot_signature_evolution function
    to recreate the MI patient scenario from Figure 6B.
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
    
    # Create the plot
    fig = plot_signature_evolution(
        time_points=time_points,
        theta_values=theta_values,
        clinical_events=clinical_events,
        primary_signature="Cardiovascular (Sig 1)",
        title="Figure 6B: Signature Evolution Before Myocardial Infarction",
        subtitle="58-year-old Male Patient with Progressive Cardiovascular Risk",
        colors=colors,
        highlight_event_idx=3,  # Highlight the MI event
        save_path=None  # Set to a path to save the figure
    )
    
    return fig

if __name__ == "__main__":
    example_mi_patient() 