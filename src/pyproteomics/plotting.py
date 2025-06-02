#basic python
import os
import sys
from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
from tabulate import tabulate
import shutil
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#dataframe management
import numpy as np
import pandas as pd
import scanpy as sc
sc.settings.verbosity = 1
sc.set_figure_params(dpi=150)
import anndata as ad
from sklearn.utils import shuffle
import scipy

#plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from adjustText import adjust_text











def plot_volcano_v2(adata, 
                    x="log2_FC", 
                    y="-log10(p_val_corr)_BH", 
                    significant=True, 
                    FDR=None, 
                    tag_top=None, 
                    group1=None, 
                    group2=None):
    
    adata_copy = adata.copy()

    plt.figure(figsize=(10,10))
    sns.scatterplot(x=adata_copy.var[x], y=adata_copy.var[y])
    
    if group1 is not None and group2 is not None:
        plt.xlabel(f"Difference in mean protein expression (log2) \n {group1}(right) vs {group2}(left)")
    plt.ylabel("-log10 corrected p-value BH")

    if significant:

        df = adata_copy.var
        df = df[df["p_val_corr_BH"] < FDR]
        plt.scatter(x=df[x], y=df[y], color="red")

        if tag_top:
            if df.shape[0] > tag_top:
                df = df.sort_values(by=y, ascending=False)[:tag_top]
            else:
                df = df.sort_values(by=y, ascending=False)

        # create texts for labelling top proteins on the left side of the x axis
        df_left = df[df[y] < 0]
        texts_left = [plt.text(df_left[x][i], df_left[y][i], df_left.index[i], ha='right', va='center', fontdict={"fontsize":8}) for i in range(df_left.shape[0])]

        adjust_text(texts_left,
            lim=500, 
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=.5)
            )

        df_right = df[df[y] > 0]
        texts_right = [plt.text(df_right[x][i], df_right[y][i], df_right.index[i], ha='right', va='center', fontdict={"fontsize":8}) for i in range(df_right.shape[0])]

        adjust_text(texts_right, 
            lim=500, 
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=.5)
            )

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.2)
    plt.grid(False)
    plt.show()

import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc
sc.settings.verbosity = 1
import plotly.io as pio
# import nbformat



def plot_boxplots_plotly(adata, 
                        x_axis="Phenotype_1", 
                        y_axis:str="n_proteins", 
                        hover_data:list=["Phenotype_2"], 
                        color_column="Phenotype_1",
                        return_fig=False,
                        save_path:str=None,
                        save_df_path:str=None,
                        **kwargs):

    adata_copy = adata.copy()

    df = pd.DataFrame(index=adata_copy.obs.index, data=adata_copy.obs.values, columns=adata_copy.obs_keys())

    fig = px.box(df, x=x_axis, y=y_axis, 
        points='all', hover_data=hover_data, 
        color=color_column, width=1000, height=800,
        color_discrete_sequence=px.colors.qualitative.G10,
        **kwargs
        )

    fig.update_layout(
        title=dict(text="Proteins per sample", 
                font=dict(size=30), 
                automargin=True, 
                yref='paper'),
        font=dict( size=18, color='black'),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True,   
    )

    if save_df_path is not None:
        df.to_csv(save_df_path)
    if save_path is not None:
        # plt.savefig(save_path, format="png")
        fig.write_image(save_path, engine='kaleido') 


    if return_fig:
        return fig
    else:
        fig.show()


def plot_pca_protein_loadings(adata):
    from adjustText import adjust_text
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


def complex_heatmap(adata, obs_keys):

    from PyComplexHeatmap import HeatmapAnnotation, anno_simple, ClusterMapPlotter

    #create dataframe to plot values
    df2 = pd.DataFrame(data=adata.layers['zscored'], columns=adata.var_names, index=adata.obs['LCMS_run_id'])
    df2t = df2.T

    #create dataframe to labels categoricals
    df_color = adata_copy.obs[['RCN_long', 'Biopsy_type']]

    #create palettes
    palette_RCN = sns.color_palette("Set2", len(df_color['RCN_long'].unique()))
    palette_Biopsy_type = sns.color_palette("Set1", len(df_color['Biopsy_type'].unique()))

    #create colormaps
    color_map_RCN = dict(zip(df_color['RCN_long'].unique(), palette_RCN))
    color_map_Biopsy_type = dict(zip(df_color['Biopsy_type'].unique(),palette_Biopsy_type))

    plt.figure(figsize=(5, 8))

    col_ha = HeatmapAnnotation(
                    Cellular_Neighborhood=anno_simple(
                        df_color['RCN_long'], add_text=False, colors=color_map_RCN,legend_kws={'frameon':False}),
                    Biopsy_type=anno_simple(
                        df_color['Biopsy_type'], add_text=False, colors=color_map_Biopsy_type,legend_kws={'frameon':False} )
                            )

    cm = ClusterMapPlotter(
                    data=df2t, 
                    top_annotation=col_ha,
                    label='Zscored',
                    row_dendrogram=True,
                    show_rownames=False,
                    show_colnames=False,
                    # tree_kws={'row_cmap': 'Dark2'},
                    cmap='seismic', 
                    vmax=3, vmin=-3, center=0,
                    legend_gap=5,
                    legend_hpad=2,
                    legend_vpad=5)

    plt.show()


## Plot density curves between two groups, for specific genes/proteins

# # plot 25 proteins in a 5,5 grid
# n = 25
# n_cols = 5
# n_rows = 5
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15), constrained_layout=True)
# axes = axes.flatten()

# # Randomly select 25 proteins
# adata_RCN1_check = adata_RCN1[:, (adata_RCN1.var['-log10(p_val_corr)_BH']>5) & (adata_RCN1.var['-log10(p_val_corr)_BH']<15)]
# df = pd.DataFrame(data=adata_RCN1_check.X, columns=adata_RCN1_check.var_names, index=adata_RCN1_check.obs['Biopsy_type'])
# random_columns = np.random.choice(df.columns, n, replace=False)
# df_subset = df[random_columns]

# for i, col in enumerate(df_subset.columns):
#     df_tmp = df_subset[[col]]
#     df_tmp.reset_index(inplace=True)
#     sns.kdeplot(data=df_subset, x=col, hue="Biopsy_type", ax=axes[i])
#     axes[i].set_title(col)
#     axes[i].set_xlabel("")
#     axes[i].set_ylabel("")
#     axes[i].legend_.remove()