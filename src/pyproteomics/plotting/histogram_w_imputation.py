import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def histogram_w_imputation(adata_before_imputation, adata_after_imputation, n_cols=4, save=False, save_name=None):

    """
    Created by Jose Nimo on 2023 07 15
    Modified by Jose Nimo on 2023 08 15

    Description: 
        This function creates a histogram for each sample in the dataset.
        The histogram shows the distribution of the log2 quantified protein abundance values for each sample.
        The histogram also shows the distribution of the imputed values for each sample.
        The imputed values are shown in red.
        The raw values are shown in blue.
    Variables:
        adata_before_imputation: AnnData object with the raw data
        adata_after_imputation: AnnData object with the imputed data
        n_cols: number of columns for the subplot
        save: boolean, if True the figure is saved
        save_name: string, name of the file to save the figure
    Returns:
        None
    """
    adata1 = adata_before_imputation.copy()
    adata2 = adata_after_imputation.copy()

    raw_data = adata1.X
    imputed_data = adata2.X

    #create figure and subplots
    n_samples = adata_before_imputation.shape[0]
    n_rows = int(np.ceil(n_samples / n_cols))
    # Define the fixed size for each subplot in inches (width x height)
    fixed_subplot_size = (5, 5)
    # Calculate the figure size based on the number of rows and columns
    fig_width = fixed_subplot_size[0] * n_cols
    fig_height = fixed_subplot_size[1] * n_rows
    # Create the figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols= n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True) #figsize = (width, height)
    #flatten axes
    axes = axes.flatten()
    #create bins for both histograms
    bins = np.arange(5,25.5,0.5)

    for i in range(n_samples):
        # Extract the values for the current row
        row_raw = raw_data[i, :]
        row_imputed = imputed_data[i, :]
        #select current subplot
        ax = axes[i]
        
        # Remove NaN values from the raw data
        row_raw_for_plot = row_raw[~np.isnan(row_raw)]
        # Plot the histogram for raw data
        sns.histplot(row_raw_for_plot, bins=bins, color='blue', label='Raw Data', ax=ax, kde=True)
        # Find out which values were imputed
        imputed_data_only = row_imputed[~np.isin(element=row_imputed,test_elements=row_raw)]
        # Plot the histogram for imputed data
        sns.histplot(imputed_data_only, bins=bins, color='red', label='Imputed Data', ax=ax, kde=True)

        ax.set_box_aspect(1)
        ax.set_xlim(5, 25)
        ax.grid(False)
        # Set the title and labels
        ax.set_title(f'Histogram for {adata_after_imputation.obs.raw_file_id[i]}')
        ax.set_xlabel('Log2 Quantified Protein Abundance')
        ax.set_ylabel('Protein hits')

        # Add a legend
        ax.legend()
        #remove grid
        
    fig.tight_layout()
    fig.suptitle("Gaussian Imputation (per protein) for each sample", fontsize=30, y=1.015)
    plt.show()