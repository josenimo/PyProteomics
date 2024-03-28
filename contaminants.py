import pandas as pd
import numpy as np
from tabulate import tabulate



def filter_out_contaminants(adata, export_path=None):
    print("----- Filter out contaminants -----")
    
    #create condition
    condition1 = adata.var["Protein.Ids"].str.contains("Cont_")
    condition2 = adata.var_names.str.contains("Cont_")
    combined_condition = condition1 | condition2

    filtered_out = adata[:, combined_condition]
    filtered_out.var["Species"] = filtered_out.var["Protein.Names"].str.split("_").str[-1]

    print("the following proteins were filtered out:")
    print(tabulate(
        filtered_out.var.sort_values(by="Species")[["Genes","Protein.Names","Species"]].values,
        headers=["Genes","Protein.Names","Species"], 
        tablefmt='psql',
        showindex="always",
        maxcolwidths=[20,20,20]))

    if export_path:
        filtered_out.var.sort_values(by="Species")[["Genes","Protein.Names","Species"]].to_csv(export_path)
    
    adata = adata[:, ~combined_condition]

    print(f"The output object has {adata.shape[1]} proteins in it")
    print("\n")
    return adata