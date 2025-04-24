import os, sys, time
import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger
import warnings

datetime = time.strftime("%Y%m%d_%H%M%S")
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def filter_invalid_proteins(
        adata, 
        threshold:float=0.7, 
        grouping:str=None, 
        qc_export_path:str=None,
        valid_in_ANY_or_ALL_groups:str='ANY') -> ad.AnnData:
    """
    Created by Jose Nimo on 2023-07-01
    Latest modified by Jose Nimo on 2025-01-24

    Description:
        Filter out proteins that have a NaN proportion above the threshold, for each group in the grouping variable.
    Arg:
        adata: anndata object
        threshold: float, between 0 and 1, proportion of valid values above which a protein is considered valid
        grouping: Optional, string, name of the column in adata.obs to discriminate the groups by,
            if two groups are chosen, the protein must have a valid value in at least one of the groups
        qc_export_path: Optional, string, path to save the dataframe with the filtering results
        valid_in_ANY_or_ALL_groups:str='ANY'
            "ANY" means that if a protein passes the threshold in any group it will be kept
            "ALL" means that a protein must pass validity threshold for all groups to be kept (more stringent)
    Returns:
        adata: anndata object, filtered
    """

    # TODO let users decide on an absolute number of valid values

    logger.info(f"Filtering protein without atleast {threshold*100}% valid values in {valid_in_ANY_or_ALL_groups} group")

    import warnings #for numpy mean of empty slice, which is expected
    warnings.simplefilter("ignore", category=RuntimeWarning)

    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"
    assert valid_in_ANY_or_ALL_groups in ['ANY', 'ALL'], "valid_in_ANY_or_ALL_groups must be 'ANY' or 'ALL'"

    adata_copy = adata.copy()

    df_proteins = pd.DataFrame(index=adata_copy.var_names, columns=["Genes"], data=adata_copy.var['Genes'])
    df_proteins['Genes'] = df_proteins['Genes'].astype(str)
    df_proteins.fillna({"Genes":'None'}, inplace=True)

    df_proteins = pd.DataFrame(index=adata_copy.var_names, columns=["Genes"], data=adata_copy.var['Genes'])

    if grouping:
        logger.info(f"Filtering by groups, {grouping}: {adata.obs[grouping].unique().tolist()}")

        for group in adata.obs[grouping].unique():
            adata_group = adata[adata.obs[grouping] == group]
            logger.info(f" {group} has {adata_group.shape[0]} samples and {adata_group.shape[1]} proteins") 
            
            X = adata_group.X.astype('float64')
            df_proteins[f"{group}_mean"]            = np.nanmean(X, axis=0).round(3)    
            df_proteins[f'{group}_nan_count']       = np.isnan(X).sum(axis=0)
            df_proteins[f'{group}_valid_count']     = (~np.isnan(X)).sum(axis=0)
            df_proteins[f'{group}_nan_proportions'] = np.isnan(X).mean(axis=0).round(3)
            df_proteins[f'{group}_valid']           = df_proteins[f'{group}_nan_proportions'] < (1.0 - threshold)   
        
        df_proteins['valid_in_all'] = df_proteins[[f'{group}_valid' for group in adata.obs[grouping].unique()]].all(axis=1)
        df_proteins['valid_in_any'] = df_proteins[[f'{group}_valid' for group in adata.obs[grouping].unique()]].any(axis=1)
        df_proteins['not_valid_in_any'] = ~df_proteins['valid_in_any']

        if valid_in_ANY_or_ALL_groups == 'ALL':
            adata_copy = adata_copy[:, df_proteins.valid_in_all.values]
            logger.info(f"{df_proteins['valid_in_all'].sum()} proteins were kept")
            logger.info(f"{df_proteins.shape[0] - df_proteins['valid_in_all'].sum()} proteins were removed")
        elif valid_in_ANY_or_ALL_groups == 'ANY':
            adata_copy = adata_copy[:, df_proteins.valid_in_any.values]
            logger.info(f"{df_proteins['valid_in_any'].sum()} proteins were kept")
            logger.info(f"{df_proteins.shape[0] - df_proteins['valid_in_any'].sum()} proteins were removed")

    else:
        logger.info("No grouping variable was provided")
        logger.debug(f"adata has {adata_copy.shape[0]} samples and {adata_copy.shape[1]} proteins")

        df_proteins["mean"]            = np.nanmean(adata_copy.X, axis=0)
        df_proteins['nan_count']       = np.isnan(adata_copy.X).sum(axis=0)
        df_proteins['valid_count']     = (~np.isnan(adata_copy.X)).sum(axis=0)
        df_proteins['nan_proportions'] = np.isnan(adata_copy.X).mean(axis=0)
        df_proteins['valid']           = df_proteins[f'nan_proportions'] < (1.0 - threshold)
        df_proteins['not_valid']       = ~df_proteins['valid']

        adata_copy = adata_copy[:, df_proteins.valid.values]
        print(f"{df_proteins['valid'].sum()} proteins were kept")
        print(f"{df_proteins['not_valid'].sum()} proteins were filtered out")
    
    if qc_export_path:
        logger.info(f"Saving dataframe with filtering results to {qc_export_path}")
        df_proteins.to_csv(qc_export_path)
    
    return adata_copy