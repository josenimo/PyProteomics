import os, sys, time
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
import tabulate

datetime = time.strftime("%Y%m%d_%H%M%S")
# from tabulate import tabulate
# import shutil
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.simplefilter(action="ignore", category=RuntimeWarning)

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")
sc.settings.verbosity = 1
sc.set_figure_params(dpi=150)

def filter_proteins_without_genenames(adata) -> ad.AnnData:
    """
    Description:
        Removes proteins without gene names
    """

    adata_copy = adata.copy()

    df_missing = adata.var[adata.var.Genes.isna()]
    logger.info(f"Found {df_missing.shape[0]} genes as NaNs")
    print(tabulate.tabulate(df_missing, headers=df_missing.columns, tablefmt="grid"))
    logger.info("Returning adata with those proteins/genes")

    return adata_copy[:, ~adata_copy.var.Genes.isna()]
    
def filter_out_contaminants(
        adata,
        adata_var_column_with_label:str="Protein.Ids",
        string_to_indicate_removal:str="Cont_",
        keep_genes:list=[],
        print_summary=False, 
        qc_export_path=None) -> ad.AnnData:
    """
    Version 2.0.0
    Description:
        This function filters out contaminants from the adata object.
    Parameters:
        adata: AnnData object
            The AnnData object containing the protein expression data.
        adata_var_column_with_label:
            The column in adata.var that contains the protein names with substring to be removed.
        string_to_indicate_removal: str
            The string that indicates that a protein is a contaminant.
        keep_genes: list of str, default=None
            List of gene names to retain, even if they contain contaminant indicators.
        print_summary: bool, default=False
            If True, a summary of the filtered out contaminants will be printed.
        qc_export_path: str, default=None
            If not None, the filtered out contaminants will be exported to this path.
    Returns:
        AnnData object
            The AnnData object with contaminants filtered out.
    Date: 17.11.2024
    """

    #TODO hardcoding DIANN columns is an issue
    
    # as of 17.12.2024 DIANN outputs contamination string on the index


    logger.info("Filtering out contaminants")
    adata_copy = adata.copy()

    condition = adata_copy.var.index.str.contains(string_to_indicate_removal)

    # if len(keep_genes)>0 :
    #     logger.info(f"Keeping {keep_genes} from being removed")
    #     accumulated_boolean = np.zeros(adata_copy.var.shape[0], dtype=bool)
    #     for gene in keep_genes:
    #         logger.info(f"{gene} being kept")
    #         match_boolean = adata.var["Genes"].str.contains(gene, na=False).values.astype(bool)
    #         accumulated_boolean |= match_boolean
    #         logger.info(f"Number of excluded contaminants: {accumulated_boolean.sum()}")
    #     condition = np.where(condition & accumulated_boolean, False, condition)

    # filtered_out = adata_copy[:, condition].copy()
    # filtered_out.var["Species"] = filtered_out.var["Protein.Names"].str.split("_").str[-1]

    # if print_summary:
    #     print("the following proteins were filtered out:")
    #     print(tabulate.tabulate(
    #         filtered_out.var.sort_values(by="Species")[["Genes","Protein.Names","Species"]].values,
    #         headers=["Genes","Protein.Names","Species"], 
    #         tablefmt='psql',
    #         showindex="always",
    #         maxcolwidths=[20,20,20]))

    # if qc_export_path:
    #     filtered_out.var.sort_values(by="Species")[["Genes","Protein.Names","Species"]].to_csv(qc_export_path)
    
    adata_copy = adata_copy[:, ~condition]

    print(f"The output object has {adata_copy.shape[1]} proteins in it")
    print("\n")
    return adata_copy

#### ERROR: ## 

# ---------------------------------------------------------------------------
# KeyError                                  Traceback (most recent call last)
# Cell In[93], line 3
#       1 datetime = time.strftime("%Y%m%d_%H%M%S")
#       2 import filtering
# ----> 3 adata = filtering.filter_out_contaminants(adata, 
#       4                                         adata_var_column_with_label="Protein.Names",
#       5                                         print_summary=False)

# File ~/Jose_BI/3_Python_Functions/PyProteomics/pypro/filtering.py:82, in filter_out_contaminants(adata, adata_var_column_with_label, string_to_indicate_removal, keep_genes, print_summary, qc_export_path)
#      79         logger.info(f"Number of excluded contaminants: {accumulated_boolean.sum()}")
#      80     condition = np.where(condition & accumulated_boolean, False, condition)
# ---> 82 filtered_out = adata_copy[:, condition].copy()
#      83 filtered_out.var["Species"] = filtered_out.var["Protein.Names"].str.split("_").str[-1]
#      85 if print_summary:

# File /opt/homebrew/Caskroom/mambaforge/base/envs/proteomics/lib/python3.12/site-packages/anndata/_core/anndata.py:1021, in AnnData.__getitem__(self, index)
#    1019 def __getitem__(self, index: Index) -> AnnData:
#    1020     """Returns a sliced view of the object."""
# -> 1021     oidx, vidx = self._normalize_indices(index)
#    1022     return AnnData(self, oidx=oidx, vidx=vidx, asview=True)

# File /opt/homebrew/Caskroom/mambaforge/base/envs/proteomics/lib/python3.12/site-packages/anndata/_core/anndata.py:1002, in AnnData._normalize_indices(self, index)
#    1001 def _normalize_indices(self, index: Index | None) -> tuple[slice, slice]:
# -> 1002     return _normalize_indices(index, self.obs_names, self.var_names)

# File /opt/homebrew/Caskroom/mambaforge/base/envs/proteomics/lib/python3.12/site-packages/anndata/_core/index.py:39, in _normalize_indices(index, names0, names1)
#      37 ax0, ax1 = unpack_index(index)
#      38 ax0 = _normalize_index(ax0, names0)
# ---> 39 ax1 = _normalize_index(ax1, names1)
#      40 return ax0, ax1

# File /opt/homebrew/Caskroom/mambaforge/base/envs/proteomics/lib/python3.12/site-packages/anndata/_core/index.py:103, in _normalize_index(indexer, index)
#     101         if np.any(positions < 0):
#     102             not_found = indexer[positions < 0]
# --> 103             raise KeyError(
#     104                 f"Values {list(not_found)}, from {list(indexer)}, "
#     105                 "are not valid obs/ var names or indices."
#     106             )
#     107         return positions  # np.ndarray[int]
#     108 else:

# KeyError: 'Values [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, False, False, False, False, False, False, False, False, Fa




def filter_invalid_proteins(
        adata, 
        threshold:float=0.7, 
        grouping:str=None, 
        qc_export_path:str=None,
        valid_in_ANY_or_ALL_groups:str='ANY') -> ad.AnnData:
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-11-16

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

    logger.info(f"Filtering proteins, they need to have {threshold*100}% valid values to be kept")

    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"
    assert valid_in_ANY_or_ALL_groups in ['ANY', 'ALL'], "valid_in_ANY_or_ALL_groups must be 'ANY' or 'ALL'"

    adata_copy = adata.copy()

    df_proteins = pd.DataFrame(index=adata_copy.var_names, columns=["Genes"], data=adata_copy.var['Genes'])
    df_proteins['Genes'] = df_proteins['Genes'].astype(str)
    df_proteins.fillna({"Genes":'None'}, inplace=True)

    if grouping:
        logger.info(f"Filtering proteins by groups, {grouping}: {adata.obs[grouping].unique().tolist()}")

        for group in adata.obs[grouping].unique():
            logger.debug(f"Processing group: {group}")
            adata_group = adata[adata.obs[grouping] == group]
            logger.debug(f"Group {group} has {adata_group.shape[0]} samples and {adata_group.shape[1]} proteins")
            
            df_proteins[f"{group}_mean"]            = np.nanmean(adata_group.X, axis=0).round(3)
            df_proteins[f'{group}_nan_count']       = np.isnan(adata_group.X).sum(axis=0)
            df_proteins[f'{group}_valid_count']     = (~np.isnan(adata_group.X)).sum(axis=0)
            df_proteins[f'{group}_nan_proportions'] = np.isnan(adata_group.X).mean(axis=0).round(3)
            df_proteins[f'{group}_valid']           = df_proteins[f'{group}_nan_proportions'] < (1.0 - threshold)   
        
        logger.info(f"Any protein that has a minimum of {threshold*100} valid values in {valid_in_ANY_or_ALL_groups} group, will be kept")
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