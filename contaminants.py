import pandas as pd
import numpy as np
from tabulate import tabulate
import anndata as ad

def filter_out_contaminants(adata, print_summary=False, qc_export_path=None) -> ad.AnnData:
    """
    Version 1.1.1

    Description:
    This function filters out contaminants from the adata object.

    Parameters:
    adata: AnnData object
        The AnnData object containing the protein expression data.
    print_summary: bool, default=False
        If True, a summary of the filtered out contaminants will be printed.
    qc_export_path: str, default=None
        If not None, the filtered out contaminants will be exported to this path.

    Returns:
        AnnData object
            The AnnData object with contaminants filtered out.

    Date: 13.11.2024
    """
    print("----- Filter out contaminants -----")
    adata_copy = adata.copy()

    #create condition
    condition1 = adata_copy.var["Protein.Ids"].str.contains("Cont_")
    condition2 = adata_copy.var_names.str.contains("Cont_")
    combined_condition = condition1 | condition2
    filtered_out = adata_copy[:, combined_condition].copy()
    filtered_out.var["Species"] = filtered_out.var["Protein.Names"].str.split("_").str[-1]
    if print_summary:
        print("the following proteins were filtered out:")
        print(tabulate(
            filtered_out.var.sort_values(by="Species")[["Genes","Protein.Names","Species"]].values,
            headers=["Genes","Protein.Names","Species"], 
            tablefmt='psql',
            showindex="always",
            maxcolwidths=[20,20,20]))

    if qc_export_path:
        filtered_out.var.sort_values(by="Species")[["Genes","Protein.Names","Species"]].to_csv(qc_export_path)
    
    adata_copy = adata_copy[:, ~combined_condition]

    print(f"The output object has {adata_copy.shape[1]} proteins in it")
    print("\n")
    return adata_copy