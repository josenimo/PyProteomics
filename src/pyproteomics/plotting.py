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

def plot_correlation_heatmap(adata, correlation_method="spearman",Title="Spearman Correlation Heatmap", Sample_Label="raw_file_id"):

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

def plot_PCA_from_adata(adata, color_category, title_PCA="PCA", PC1=1, PC2=2):
    """
    Created by Jose Nimo on 2023 09 29
    Modified by Jose Nimo on 2023 09 29

    Description: 
        This function creates a PCA plot, using scanpy.plot.pca, from an adata object.
        The user can color the data points by a categorical variable, found in adata.obs
        The user can also select which principal components to plot.

    Variables:
        adata: AnnData object
        color_category: string, name of the column in adata.obs to color the plot by
        title_PCA: string, title of the plot (default = "PCA")
        PC1: integer, number of the first principal component to plot (default = 1)
        PC2: integer, number of the second principal component to plot (default = 2)

    Returns:
        None
    """

    if adata.uns['pca'] is None:
        sc.tl.pca(adata, svd_solver='arpack')

    #obtain sample coordinates per PC chosen
    

    variance_ratio = adata.uns['pca']['variance_ratio'].tolist()

    plt.xlabel(f'PC{PC1}  ({variance_ratio[PC1]*100:.2f}%)')
    plt.ylabel(f'PC{PC2}  ({variance_ratio[PC2]*100:.2f}%)')

def PCA_comparison(adata1, adata2, color, categorical=False):
    """
    Created by Jose Nimo on 2023 07 15
    Modified by Jose Nimo on 2023 08 23

    Description:
        This function creates a PCA plot for two adata objects.
    Variables:
        adata1: AnnData object
        adata2: AnnData object
        color: string, name of the column in adata.obs to color the plot by
        categorical: boolean, if True the column in adata.obs is converted to categorical
    Returns:
        None
    """
    
    adata1_tmp = adata1.copy()
    adata2_tmp = adata2.copy()

    if categorical:
        adata1_tmp.obs[color] = adata1_tmp.obs[color].astype('category')
        adata2_tmp.obs[color] = adata2_tmp.obs[color].astype('category')
        
    sc.tl.pca(adata1_tmp, svd_solver='arpack')
    sc.tl.pca(adata2_tmp, svd_solver='arpack')
    
    #create figure with two subplots, side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sc.pl.pca(adata1_tmp, 
                color=color, 
                show=False, 
                size=300, 
                alpha=0.8,
                title=f'{celltype} before batch correction',
                ax=ax1)
    
    sc.pl.pca(adata2_tmp, 
                color=color, 
                show=False, 
                size=300, 
                alpha=0.8,
                title=f'{celltype} after batch correction',
                ax=ax2)

    variance_ratio_before = adata1_tmp.uns['pca']['variance_ratio'].tolist()
    variance_ratio_after = adata2_tmp.uns['pca']['variance_ratio'].tolist()

    ax1.set_xlabel(f'PC1  ({variance_ratio_before[0]*100:.2f}%)', fontsize=10)
    ax1.set_ylabel(f'PC2  ({variance_ratio_before[1]*100:.2f}%)', fontsize=10)
    ax2.set_xlabel(f'PC1  ({variance_ratio_after[0]*100:.2f}%)', fontsize=10)
    ax2.set_ylabel(f'PC2  ({variance_ratio_after[1]*100:.2f}%)', fontsize=10)

    ax1.legend('',frameon=False)
    handles, labels = ax2.get_legend_handles_labels()
    labels = ['Batch 1', 'Batch 2', 'Batch 3']
    ax2.legend(handles, labels, frameon=True)
    
    plt.show()

def plot_volcano_v2(adata, x="log2_FC", y="-log10(p_val_corr)_BH", significant=True, FDR=None, tag_top=None, group1=None, group2=None):
    
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

        if df.shape[0] > 30:
            df = df.sort_values(by=y, ascending=False)[:30] #top 30 maximum labels
        
        # texts = [plt.text(df[x][i], df[y][i], df.Genes[i], ha='center', va='center') for i in range(df.shape[0])]

        # create texts for labelling top proteins on the left side of the x axis
        df_left = df[df[y] < 0]
        texts_left = [plt.text(df_left[x][i], df_left[y][i], df_left.Genes[i], ha='right', va='center') for i in range(df_left.shape[0])]

        adjust_text(texts_left,
        lim=500, 
        expand_points=(1.2, 1.2),
        # expand_text=(1.2, 1.2),
        # force_text=(1,0.5),
        # only_move={'points':'-x', 'text':'-x', 'objects':'-x'},
        # precision=0.1,
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=.5)
        )


        df_right = df[df[y] > 0]
        texts_right = [plt.text(df_right[x][i], df_right[y][i], df_right.Genes[i], ha='right', va='center') for i in range(df_right.shape[0])]

        adjust_text(texts_right, 
        lim=500, 
        expand_points=(1.2, 1.2),
        # expand_text=(1.2, 1.2),
        # force_text=(1,0.5),
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=.5)
        )

        # Create custom legend handles
        # legend_elements = [Patch(facecolor='red', edgecolor='black', label='significant')] 
    
    #create a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.2)
    #remove grid
    plt.grid(False)
    # plt.xlim(-6, 6)    # Add a legend with custom handles
    # plt.legend(handles=legend_elements, loc="upper right")
    plt.show()

    #TODO add 

import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc
sc.settings.verbosity = 1
import plotly.io as pio
# import nbformat

def PCA_adata(adata, 
            color:str=None, 
            group_colors:dict=None,
            symbol:str=None, 
            hoverwith:list=["sampleid"],
            choose_PCs:list=[1,2],
            multi_scatter:bool=False, how_many_PCs:int=4,
            scatter_3d:bool=False,
            save_path:str=None,
            return_fig:bool=False,
            ):

    if adata.uns['pca'] is None:
        sc.pp.pca(adata, svd_solver='arpack')
        print("PCA was not found in adata.uns['pca']. It was computed now.")
    
    df = pd.DataFrame(data=adata.obsm['X_pca'], 
                        columns=[f'PC{i+1}' for i in range(adata.obsm['X_pca'].shape[1])], 
                        index=adata.obs.index)
    df = pd.concat([df, adata.obs], axis=1)

    if multi_scatter and scatter_3d:
        print("Please choose between multi_scatter and scatter_3d. Not both.")
        return None
    
    elif multi_scatter:
        features = [ f'PC{i+1}' for i in range(how_many_PCs)]
        components = df[features].values
        labels = {str(i): f"PC {i+1} ({var:.1f}%)" 
                for i, var in enumerate(adata.uns['pca']['variance_ratio']*100)}
        fig = px.scatter_matrix(
            components, labels=labels, dimensions=range(how_many_PCs), color=df[color], symbol=df[symbol])
        fig.update_traces(diagonal_visible=False,
                        marker={'size': 18, 'opacity': 0.8})
        dimension = how_many_PCs*500
        fig.update_layout(height=dimension,width=dimension,
                        font=dict(size=20, color='black'),)
        if save_path is not None:
            fig.write_image(save_path, engine='kaleido') 
        if return_fig:
            return fig

    elif scatter_3d:
        features = [ f'PC{i+1}' for i in range(3)]
        components = df[features].values
        fig = px.scatter_3d(
            components, 
            x=0, y=1, z=2, 
            color=df[color],
            symbol=df[symbol],
            labels={'0': f'PC1  {adata.uns["pca"]["variance_ratio"][0]*100:.2f}%', 
                    '1': f'PC2  {adata.uns["pca"]["variance_ratio"][1]*100:.2f}%', 
                    '2': f'PC3  {adata.uns["pca"]["variance_ratio"][2]*100:.2f}%'},
            )
        fig.update_layout(width=1000, height=1000)
        if save_path is not None:
            fig.write_html(save_path)
        if return_fig:
            return fig
        
    
    else:
        fig = px.scatter(df, x=f'PC{choose_PCs[0]}', y=f'PC{choose_PCs[1]}', 
                        color=color, 
                        symbol=symbol, 
                        hover_data=hoverwith,
                        labels={
                            f'PC{choose_PCs[0]}': 
                            f'PC{choose_PCs[0]} ({adata.uns["pca"]["variance_ratio"][choose_PCs[0]-1]*100:.2f}%)',
                            f'PC{choose_PCs[1]}': 
                            f'PC{choose_PCs[1]} ({adata.uns["pca"]["variance_ratio"][choose_PCs[1]-1]*100:.2f}%)'
                        },
                        color_discrete_map=group_colors
                        )
        fig.update_layout(
            title=dict(text=f"PCA of samples by {color} and {symbol}", font=dict(size=24), 
                        automargin=False, yref='paper'),
            font=dict( size=15, color='black'),
            width=1500,
            height=1000,
            # template='plotly_white'
            )
        fig.update_traces(
            marker={'size': 15, 'opacity': 0.8})
        
        if save_path is not None:
            fig.write_image(save_path, engine='kaleido') 
        if return_fig:
            return fig
    
    if not return_fig:
        fig.show()

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