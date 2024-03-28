#Created by Jose Nimo

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

def filter_invalid_proteins(adata, grouping, threshold, save_df_path:str=None):
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-03-28

    Description:
        Filter out proteins that have a NaN proportion above the threshold, for each group in the grouping variable.
    Variables:
        adata: anndata object
        grouping: string, name of the column in adata.obs that contains the groups
        threshold: float, between 0 and 1, proportion of valid values above which a protein is considered valid
    Returns:
        adata: anndata object, filtered
    """

    print(f"------------------Filter Invalid Proteins ------------------")
    print(f"Keeping proteins that have at least {threshold} valid values for any group in {grouping}\n")

    df_proteins = pd.DataFrame(index=adata.var_names, columns=['Genes'], data=adata.var['Genes'])
    df_proteins['Genes'].fillna('None', inplace=True)

    if grouping:

        for group in adata.obs[grouping].unique():
            
            adata_group = adata[adata.obs[grouping] == group]
            protein_data_group = adata_group.X
            
            df_proteins[f"{group}_mean"]            = np.nanmean(adata_group.X, axis=0)
            df_proteins[f'{group}_nan_count']       = np.isnan(protein_data_group).sum(axis=0)
            df_proteins[f'{group}_valid_count']     = (~np.isnan(protein_data_group)).sum(axis=0)
            df_proteins[f'{group}_nan_proportions'] = np.isnan(protein_data_group).mean(axis=0)
            df_proteins[f'{group}_valid']           = df_proteins[f'{group}_nan_proportions'] < threshold
        
        df_proteins['valid_in_all'] = df_proteins[[f'{group}_valid' for group in adata.obs[grouping].unique()]].all(axis=1)
        df_proteins['valid_in_any'] = df_proteins[[f'{group}_valid' for group in adata.obs[grouping].unique()]].any(axis=1)
        df_proteins['not_valid_in_any'] = ~df_proteins['valid_in_any']

        adata = adata[:, df_proteins.valid_in_any.values]
        print(f"{df_proteins['valid_in_any'].sum()} proteins were kept")
        print(f"{df_proteins['not_valid_in_any'].sum()} proteins were filtered out")

    else:

        print("No grouping variable was provided. Filtering out proteins with NaN values")
        df_proteins["mean"]            = np.nanmean(adata.X, axis=0)
        df_proteins['nan_count']       = np.isnan(adata.X).sum(axis=0)
        df_proteins['valid_count']     = (~np.isnan(adata.X)).sum(axis=0)
        df_proteins['nan_proportions'] = np.isnan(adata.X).mean(axis=0)
        df_proteins['valid']           = df_proteins[f'nan_proportions'] < threshold
        df_proteins['not_valid']       = ~df_proteins['valid']

        adata = adata[:, df_proteins.valid.values]
        print(f"{df_proteins['valid'].sum()} proteins were kept")
        print(f"{df_proteins['not_valid'].sum()} proteins were filtered out")
    
    if save_df_path:
        df_proteins.to_csv(save_df_path)
    
    print(f"For more details check the dataframe saved with save_df_path argument")
    
    return adata