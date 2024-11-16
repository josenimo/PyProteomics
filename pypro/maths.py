#functions.py
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

#math
import pingouin as pg
import statsmodels.api as sm
import statsmodels.stats.multitest
from scipy.stats import zscore
from scipy.stats import pearsonr
from scipy.stats import spearmanr 
from scipy.stats import ttest_ind
from scipy.sparse import csr_matrix
from scipy.stats import shapiro

#plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from adjustText import adjust_text

def Average_by_group(adata, group_to_average, group_to_keep, save_path=None):
    """
    Created by Jose Nimo on 2023-07-20
    Modified by Jose Nimo on 2023-08-16

    Description:
        This function averages the values of the adata object by a group.
        Keeping another variable as metadata.
    Variables:
        adata: AnnData object
        group_to_average: str, column header in adata.obs
            The column header to average by
        group_to_keep: str, column header in adata.obs
            The column header to keep as metadata
        save_path: str, default=None
            Path to save the averaged data as csv matrix
    Returns:
        adata: AnnData object
    """

    df = adata.obs
    #dictionary to create a mapping between the group to average and the group to keep
    dictionary = {}
    for index, row in df.iterrows():
        key = row[group_to_average]
        value = row[group_to_keep]
        if key in dictionary:
            pass
        else:
            dictionary[key] = value

    metadata = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Smoke_status'])
    metadata['patient_id'] = metadata.index
    metadata = metadata.reset_index(drop=True)
    metadata.sort_values(by=['patient_id'], inplace=True, ascending=True)

    #create df from adata
    df = pd.DataFrame(index=adata.obs[group_to_average], data=adata.X, columns=adata.var.index)
    df = df.groupby(df.index).mean().sort_index(ascending=True)

    #metadata

    adata = ad.AnnData(X=df.values, obs=metadata, var=adata.var)
    df = pd.DataFrame(data=adata.X, index=adata.obs, columns=adata.var.Genes)
    df.index.name = f"{adata.obs.keys().values}"
    if save_path:
        df.to_csv(save_path)

    print("Example of averaged values in dataframe")
    df.iloc[:, 21:27]

    return adata


def calculate_category_averages(adata, categories):
    """
    Created by CHATGPT on 2023-08-18
    Modified by CHATGPT on 2023-08-22

    Calculate averages for all permutations of given categories in adata.obs.

    Parameters:
        adata (anndata.AnnData): Annotated data matrix with observations (cells) and variables (features).
        categories (list): List of categories (column names in adata.obs) to calculate averages for.

    Returns:
        pandas.DataFrame: DataFrame containing category combinations and their corresponding averages.
    """
    print(f" --- --- --- Calculating averages for {categories} --- --- --- ")

    adata_copy = adata.copy()
    # Get the unique values for each category
    unique_values = [adata_copy.obs[cat].unique() for cat in categories]
    
    # Generate all possible combinations of category values
    combinations = pd.MultiIndex.from_product(unique_values, names=categories)
    
    # Create an empty DataFrame to store averages
    avg_df = pd.DataFrame(index=combinations, columns=adata_copy.var_names)
    
    # Loop through each category combination
    for combination in combinations:

        # Select cells that match the current category combination
        mask = np.all(np.vstack([adata_copy.obs[cat] == val for cat, val in zip(categories, combination)]), axis=0)
        selected_cells = adata.X[mask]
        
        # Calculate average for the selected cells and store it in the DataFrame
        avg_values = np.mean(selected_cells, axis=0)
        avg_df.loc[combination] = avg_values
    
    df_reset = avg_df.reset_index()

    adata_res = ad.AnnData(X=df_reset.iloc[:,2:].values, 
                            obs=df_reset[categories], 
                            var=adata_copy.var)
    
    print(" --- --- --- Category averages calculated! --- --- --- ")

    return adata_res