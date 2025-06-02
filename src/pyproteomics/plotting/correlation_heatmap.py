import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_heatmap(
        adata, 
        correlation_method="spearman",
        Title="Spearman Correlation Heatmap", 
        Sample_Label="raw_file_id"):

    """
    Plot a correlation heatmap of the protein abundance for all samples in adata
    Variables:
        adata: anndata object
        correlation_method: method to calculate the correlation (default = "spearman") being passed to df.corr()
        Title: title of the plot (default = "Spearman Correlation Heatmap")
        Sample_Label: column name in adata.obs to label samples with (default = "raw_file_id")
    """

    df = pd.DataFrame(index=adata.obs[Sample_Label], data=adata.X, columns=adata.var.Genes)
    df = df.transpose() # so that it correlates the samples, not the proteins
    correlation_matrix = df.corr(method=correlation_method)
    # Generate masks
    mask_bottom_left = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    mask_top_rigth = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=((0.55 * df.shape[1]),( 0.4 * df.shape[1])))

    sns.heatmap(correlation_matrix, annot=True, cmap='magma', fmt='.2f', linewidths=0.5, vmin=0.7, mask=mask_top_rigth, square=True, annot_kws={"size":8})
    sns.heatmap(correlation_matrix, annot=False, cmap='magma', fmt='.2f', linewidths=0.5, vmin=0.7, mask=mask_bottom_left, square=True, cbar=False)
    # Turn off gridlines
    plt.grid(False)
    plt.title(Title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.show()