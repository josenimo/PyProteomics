import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

def plot_histograms(adata, n_cols=4):
    """
    Created by Jose Nimo on 2023 07 10
    Modified by Jose Nimo on 2023 09 29

    Plot histograms of protein abundance for each sample in adata
    Variables:
        adata: anndata object
        n_cols: number of columns for the subplots (default = 4)

    Known issues:
        - Log2(x) your data before plotting
    """
    # Create a figure with subplots dynamically based on the number of rows
    n_of_samples = adata.X.shape[0]

    # Calculate the optimal number of rows and columns for subplots
    n_rows = int(np.ceil(n_of_samples / n_cols))

    # Set up the number of rows and columns for your subplots
    num_rows = n_rows
    num_cols = n_cols

    # Define the fixed size for each subplot in inches (width x height)
    fixed_subplot_size = (5, 5)
    # Calculate the figure size based on the number of rows and columns
    fig_width = fixed_subplot_size[0] * num_cols
    fig_height = fixed_subplot_size[1] * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten()
    bins=np.arange(0,25,1) #fixed bins
    
    for i in range(n_of_samples):
        ax = axes[i]                                
        sns.histplot(adata.X[i], bins=bins, ax=ax, kde=True)
        ax.set_box_aspect(1)
        ax.set_xlim(5, 25)
        # Shapiro-Wilk test, to calculate normality test and add to plot
        res = scipy.stats.shapiro(adata.X[i].toarray())
        ax.text(x=0.80, y=0.86, s= f"Schapiro p: {res[1]}", transform=ax.transAxes, fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        #set title
        ax.set_title(f'file_id: {adata.obs.raw_file_id[i]}')
        ax.grid(False)
        
    # fig.suptitle(f"Histograms showing the distribution of protein abundance for each sample of {celltype}")
    fig.tight_layout()
    # plt.subplots_adjust(top=1)
    plt.show()