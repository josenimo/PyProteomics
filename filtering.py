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
warnings.simplefilter(action="ignore", category=RuntimeWarning)

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

#dataframe management
import numpy as np
import pandas as pd
import scanpy as sc
sc.settings.verbosity = 1
sc.set_figure_params(dpi=150)
import anndata as ad

def filter_invalid_proteins(adata, threshold:float=0.6, grouping:str=None, qc_export_path:str=None) -> ad.AnnData:
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-05-07

    Description:
        Filter out proteins that have a NaN proportion above the threshold, for each group in the grouping variable.
    Variables:
        adata: anndata object
        threshold: float, between 0 and 1, proportion of valid values above which a protein is considered valid
        grouping: Optional, string, name of the column in adata.obs to discriminate the groups by,
            if two groups are chosen, the protein must have a valid value in at least one of the groups
        qc_export_path: Optional, string, path to save the dataframe with the filtering results
    Returns:
        adata: anndata object, filtered
    """

    logger.info(f"Filtering proteins with too many NaNs")

    df_proteins = pd.DataFrame(index=adata.var_names, columns=['Genes'], data=adata.var['Genes'])
    df_proteins['Genes'].fillna('None', inplace=True)


    if grouping:
        logger.info(f"Filtering proteins by groups, {grouping}: {adata.obs[grouping].unique().tolist()}")
        logger.info(f"Any protein with a NaN proportion above {threshold} in ALL groups will be filtered out")

        for group in adata.obs[grouping].unique():
            logger.debug(f"Processing group: {group}")

            adata_group = adata[adata.obs[grouping] == group]
            logger.debug(f"Group {group} has {adata_group.shape[0]} samples and {adata_group.shape[1]} proteins")
            
            df_proteins[f"{group}_mean"]            = np.nanmean(adata_group.X, axis=0)
            df_proteins[f'{group}_nan_count']       = np.isnan(adata_group.X).sum(axis=0)
            df_proteins[f'{group}_valid_count']     = (~np.isnan(adata_group.X)).sum(axis=0)
            df_proteins[f'{group}_nan_proportions'] = np.isnan(adata_group.X).mean(axis=0)
            df_proteins[f'{group}_valid']           = df_proteins[f'{group}_nan_proportions'] < threshold
        
        df_proteins['valid_in_all'] = df_proteins[[f'{group}_valid' for group in adata.obs[grouping].unique()]].all(axis=1)
        df_proteins['valid_in_any'] = df_proteins[[f'{group}_valid' for group in adata.obs[grouping].unique()]].any(axis=1)
        df_proteins['not_valid_in_any'] = ~df_proteins['valid_in_any']

        adata = adata[:, df_proteins.valid_in_any.values]
        logger.info(f"{df_proteins['valid_in_any'].sum()} proteins were kept")
        logger.info(f"{df_proteins['not_valid_in_any'].sum()} proteins were filtered out")

    else:
        logger.info(f"No grouping variable was provided, any protein with a NaN proportion above {threshold} will be filtered out")
        logger.debug(f"adata has {adata.shape[0]} samples and {adata.shape[1]} proteins")

        df_proteins["mean"]            = np.nanmean(adata.X, axis=0)
        df_proteins['nan_count']       = np.isnan(adata.X).sum(axis=0)
        df_proteins['valid_count']     = (~np.isnan(adata.X)).sum(axis=0)
        df_proteins['nan_proportions'] = np.isnan(adata.X).mean(axis=0)
        df_proteins['valid']           = df_proteins[f'nan_proportions'] < threshold
        df_proteins['not_valid']       = ~df_proteins['valid']

        adata = adata[:, df_proteins.valid.values]
        print(f"{df_proteins['valid'].sum()} proteins were kept")
        print(f"{df_proteins['not_valid'].sum()} proteins were filtered out")
    
    if qc_export_path:
        logger.info(f"Saving dataframe with filtering results to {qc_export_path}")
        df_proteins.to_csv(qc_export_path)
    
    return adata