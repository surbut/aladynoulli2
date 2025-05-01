def plot_disease_contributions(ax, disease_contributions, disease_names, title='Disease Contributions', ylabel='Contribution', xlabel='Disease', rotation=45, figsize=(12, 6)):
    """
    Plot disease contributions on the given axes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    disease_contributions : numpy.ndarray
        Array of disease contributions
    disease_names : list
        List of disease names
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label
    rotation : int, optional
        Rotation angle for x-axis labels
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    # Plot all diseases
    ax.bar(range(len(disease_names)), disease_contributions)
    ax.set_xticks(range(len(disease_names)))
    ax.set_xticklabels(disease_names, rotation=rotation, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout() 