import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot_mi_patient_trajectory():
    """
    Creates a plot showing the trajectory of a patient's signature associations
    leading up to a myocardial infarction event, as described in Figure 6B.
    """
    # Time points (years)
    years = np.arange(0, 13)
    
    # Signature 1 (Cardiovascular) association values
    # Starting low, then increasing with key events, especially rapidly before MI
    theta_values = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.35, 0.58, 0.75, 0.0])
    
    # Create figure with two subplots - trajectory on top, events on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    
    # Top plot: Signature association trajectory
    ax1.plot(years[:-1], theta_values[:-1], 'r-', linewidth=2.5, label="Cardiovascular Signature (θ)")
    
    # Highlight key measurement points
    key_years = [9, 10, 11, 12]
    key_thetas = [0.25, 0.35, 0.58, 0.75]
    ax1.plot(key_years, key_thetas, 'ro', markersize=8)
    
    # Add value annotations at key points
    for i, (year, theta) in enumerate(zip(key_years, key_thetas)):
        ax1.annotate(f'θ = {theta}', xy=(year, theta), xytext=(5, 10),
                    textcoords='offset points', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))
    
    # Mark MI event with a vertical line
    ax1.axvline(x=12, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(12.05, 0.5, 'MI Event', rotation=90, verticalalignment='center', color='red')
    
    # Configure top plot
    ax1.set_title('Figure 6B: 58-year-old Male Patient Trajectory Leading to MI', fontsize=14)
    ax1.set_ylabel('Cardiovascular Signature Association (θ)', fontsize=12)
    ax1.set_ylim(0, 0.9)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # Remove x-axis labels from top plot
    ax1.tick_params(axis='x', labelsize=0)
    
    # Bottom plot: Timeline of clinical events
    events = [
        {'year': 9, 'event': 'Hypertension Diagnosed', 'color': 'skyblue'},
        {'year': 10, 'event': 'Hyperlipidemia Diagnosed', 'color': 'orange'},
        {'year': 11, 'event': 'Abnormal Stress Test', 'color': 'purple'},
        {'year': 12, 'event': 'Myocardial Infarction', 'color': 'red'}
    ]
    
    # Plot each event
    for i, event in enumerate(events):
        ax2.scatter(event['year'], i, color=event['color'], s=100, zorder=5)
        ax2.text(event['year'] + 0.1, i, event['event'], verticalalignment='center')
    
    # Configure bottom plot
    ax2.set_yticks([])
    ax2.set_xlabel('Year of Follow-up', fontsize=12)
    ax2.set_xlim(0, 12.5)
    ax2.set_ylim(-0.5, len(events) - 0.5)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add annotation about dynamic risk assessment
    plt.figtext(0.5, 0.01, 
                "This example illustrates how Aladyn continuously integrates new clinical information\n"
                "to update risk assessments, showing increasing cardiovascular signature association\n"
                "2-3 years before the MI event.",
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()
    
    return fig

if __name__ == "__main__":
    plot_mi_patient_trajectory() 