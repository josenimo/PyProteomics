from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np

def plot_pca_protein_loadings(adata):
    
    top = 30
    n_pcs = 2

    fig, ax = plt.subplots(figsize=(10,10))
    top_genes = []
    all_top_indices = []

    for i, PC in enumerate(adata.varm['PCs'].T[:n_pcs]):  # Transpose to loop over columns
        indices = np.argsort(np.abs(PC))[::-1]
        top_indices = indices[:top]
        all_top_indices.append(top_indices.tolist())

    flattened_list = np.concatenate(all_top_indices).tolist()

    ax.scatter(x=adata.varm['PCs'].T[0], y=adata.varm['PCs'].T[1], c="b", s=7)
    # Add grid and center lines
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')  # Horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.4, linestyle='--')  # Vertical line at x=0

    x = adata.varm['PCs'][:,0][flattened_list]
    ax.set_xlabel(f"PC1 {np.round(adata.uns['pca']['variance_ratio'][0]*100, 2)} %")
    y = adata.varm['PCs'][:,1][flattened_list]
    ax.set_ylabel(f"PC2 {np.round(adata.uns['pca']['variance_ratio'][1]*100, 2)} %")
    genenames = adata.var.iloc[flattened_list]['Genes'].values

    ax.scatter(x,y, s=12, c="r")

    # Prepare the text objects
    texts = []
    for i, label in enumerate(genenames):
        # Create text labels and store them in the texts list
        text = ax.text(x[i], y[i], label, fontsize=8)
        texts.append(text)

    # Adjust text positions using adjustText
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black'))

    ax.set(xticklabels=[], yticklabels=[])
    ax.grid(False)
    plt.show()