import os, sys
import pandas as pd
import numpy as np
import tabulate
import anndata as ad
import time
datetime = time.strftime("%Y%m%d_%H%M%S")

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")


def DIANN_to_adata( DIANN_path:str,
                    DIANN_sep:str="\t",
                    metadata_path:str = None, 
                    metadata_sep:str=",", 
                    metadata_check=False, 
                    sample_id_column:str="Name" ) -> ad.AnnData:

    """
    Created by Jose Nimo on 19.01.2024
    Modified by Jose Nimo on 16.11.2024

    Description:
    Converts DIANN output file and metadata file into anndata object.
    Assumes DIANN output file is tab-delimited, with the first 5 columns being metadata, and the rest being protein expression data.
    Assumes metadata file is comma delimited, with a column of sample names with columnd header called 'Name', and the rest being metadata.

    Arguments:
    DIANN_path: path to DIANN output file
    DIANN_sep: delimiter for DIANN output file
    metadata_path: path to metadata file
    metadata_sep: delimiter for metadata file
    metadata_check: boolean, if True, prints metadata values
    sample_id_column: name of the column in metadata file that contains the sample names

    """

    logger.info("DIANN_to_adata function started (v2.0)")

    logger.info("Step 1: Loading DIANN output file")
    #load DIANN output file
    df = pd.read_csv(DIANN_path, sep=DIANN_sep)
    #all rows, all columns except first 5 to remove metadata
    rawdata = df.iloc[:,5:]
    rawdata = rawdata.transpose() #transpose to have samples as rows and proteins as columns

    logger.info("Step 2: Loading metadata file")
    sample_metadata = pd.read_csv(metadata_path, sep=metadata_sep) #load metadata file
    

    assert sample_id_column in sample_metadata.columns, f"ERROR: {sample_id_column} column not found in metadata file. Please check your files."
    assert sample_metadata[sample_id_column].nunique() == sample_metadata.shape[0], f"ERROR: {sample_id_column} has duplicates. I should not. Please check your files."

    sample_metadata.index = sample_metadata[sample_id_column] #set index to be the sample name, matching the rawdata index
    sample_metadata = sample_metadata.drop(sample_id_column, axis=1) #drop the name column, since it is now the index
    assert rawdata.shape[0] == sample_metadata.shape[0], logger.error("ERROR: Number of samples in DIANN output and metadata file do not match. Please check your files.")

    if metadata_check:
        categorical_values_dict = {}
        for column in sample_metadata.columns:
            categorical_values_dict[column] = sample_metadata[column].unique().tolist()
        data_list = [(key, value) for key, value in categorical_values_dict.items()]
        print("Sample Metadata")
        print(tabulate.tabulate(data_list, headers=["Column Name", "Unique Values"], tablefmt="grid"))
        print("\n")

    logger.info("Step 3: Loading protein metadata")
    protein_metadata = df.iloc[:,:5] #protein metadata
    protein_metadata.index = protein_metadata["Protein.Group"] #set index to be the protein name, matching the rawdata index
    protein_metadata = protein_metadata.drop("Protein.Group", axis=1) #drop the name column, since it is now the index
    logger.info(f"For a total of {protein_metadata.shape[0]} proteins \n")

    logger.info("Step 4: Creating anndata object:")
    adata = ad.AnnData(X=rawdata.values, obs=sample_metadata, var= protein_metadata) #create anndata object
    print(adata)
    print("\n")

    return adata


def adata_to_DIANN(adata):


    #TODO can improve this
    
    df_var = adata.var
    df = pd.DataFrame(data=adata.X.T, index=adata.var_names, columns=adata.obs.index) 
    df = pd.concat([df, df_var], axis=1)
    column_order = ['Protein.Ids','Protein.Names', 'Genes', 'First.Protein.Description']
    for sample_name in adata.obs.index:
        column_order.append(sample_name)
    df = df[column_order]
    return df