import os, sys
import pandas as pd
import numpy as np
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

    #TODO weird space between step 1 and step 2
    #TODO consider by default placing Genes as index in adata.var

    logger.info("DIANN_to_adata function started (v2.0)")

    logger.info("Step 1: Loading DIANN output file")
    df = pd.read_csv(DIANN_path, sep=DIANN_sep)
    # all rows, all columns except first 5 to remove metadata
    # TODO hard coding is an issue
    rawdata = df.iloc[:,4:]
    rawdata = rawdata.transpose() #transpose to have samples as rows and proteins as columns

    print("")

    logger.info("Step 2: Loading metadata file")
    sample_metadata = pd.read_csv(metadata_path, sep=metadata_sep) #load metadata file
    assert sample_id_column in sample_metadata.columns, f"ERROR: {sample_id_column} column not found in metadata file. Please check your files."
    assert sample_metadata[sample_id_column].nunique() == sample_metadata.shape[0], f"ERROR: {sample_id_column} has duplicates. I should not. Please check your files."
    sample_metadata.index = sample_metadata[sample_id_column] #set index to be the sample name, matching the rawdata index
    sample_metadata = sample_metadata.drop(sample_id_column, axis=1) #drop the name column, since it is now the index
    if not rawdata.shape[0] == sample_metadata.shape[0]:
        logger.error(f"ERROR: Number of samples in DIANN output {rawdata.shape[0]} and metadata {sample_metadata.shape[0]} do not match. Please check your files.")

    logger.info("Step 3: Loading protein metadata")
    protein_metadata = df.iloc[:,:4] #protein metadata
    protein_metadata.index = protein_metadata["Protein.Group"] #set index to be the protein name, matching the rawdata index
    protein_metadata = protein_metadata.drop("Protein.Group", axis=1) #drop the name column, since it is now the index
    logger.info(f"For a total of {protein_metadata.shape[0]} proteins \n")

    logger.info("Step 4: Creating anndata object:")
    adata = ad.AnnData(X=rawdata.values, obs=sample_metadata, var= protein_metadata) #create anndata object
    print(adata)
    print("\n")

    return adata

def switch_adat_var_index(adata, new_index):
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-11-16

    Description:
    Switch the index of adata.var to a new index. Useful for switching between gene names and protein names.

    Arg:
        adata: anndata object
        new_index: pandas series, new index to switch to
    Returns:
        adata: anndata object, with the new index
    """
    adata_copy = adata.copy()

    adata_copy.var[adata_copy.var.index.name] = adata_copy.var.index
    adata_copy.var.set_index(new_index, inplace=True)
    adata_copy.var.index.name = new_index
    
    return adata_copy

def remove_genelists_from_adata(adata, genes_index=True) -> ad.AnnData:
    
    adata_copy = adata.copy()

    print("We assume the Genes are in the index of adata.var")

    print(f"To confirm we found that {adata_copy.var[ adata_copy.var['Protein.Names'].str.contains(";") & 
                adata_copy.var["Protein.Ids"].str.contains(";") &
                adata_copy.var.index.str.contains (";") &
                adata_copy.var.index.str.contains(";")
                ].shape[0]} proteins/genes were found with ';' in their name in the four descriptive columns")
    
    adata_copy.var.index = [gene.split(";")[0] for gene in adata_copy.var.index.tolist()]

    return adata_copy

def perseus_to_anndata(path_to_perseus_txt):
    from perseuspy import pd
    df = pd.read_perseus(path_to_perseus_txt)
    # get obs headers
    obs_headers = list(df.columns.names)
    # get obs contents
    obs = [col for col in df2.columns.values] #tuples
    obs = pd.DataFrame(obs)
    # var headers HARDCODED
    var_headers = obs.iloc[-4:,0].values.tolist()
    #remove rows with empty strings
    obs = obs[obs != '']
    obs.dropna(inplace=True)
    #rename headers
    obs.columns = obs_headers
    #var 
    var = df[var_headers]
    var.columns = var_headers
    #get data
    data = df.iloc[:,:-(len(var_headers))].values.T
    adata = ad.AnnData(X=data, obs=obs, var=var)
    return adata