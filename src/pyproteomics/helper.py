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
                    metadata_sep:str = ",", 
                    metadata_filepath_header:str = "File.Name",
                    filter_contamination = True,
                    filter_nan_genes = True,
                    n_of_protein_metadata_cols = 4 ) -> ad.AnnData:

    """
    Created by Jose Nimo on 19.01.2024
    Modified by Jose Nimo on 23.01.2025

    Description:
    Converts DIANN output file and metadata file into anndata object.
    Assumes DIANN output file is tab-delimited, with the first 4 columns being metadata.
    Assumes metadata file is comma delimited, with a column of matching DIANN filepaths.

    Arguments:
    DIANN_path: path to DIANN output file
    DIANN_sep: delimiter for DIANN output file
    metadata_path: path to metadata file
    metadata_sep: delimiter for metadata file
    metadata_filepath_header: name of the column in metadata file that contains the DIANN file paths
    filter_contamination: default True, removes Protein.Names labelled with 'Cont_' as a prefix
    filter_nan_genes: default True, removes variable rows that contain NaN in the 'Genes' column, good for downstream
    """

    df = pd.read_csv(DIANN_path, sep=DIANN_sep)
    logger.info(f"Starting DIANN matrix shape {df.shape}")
    if filter_contamination:
        condition_cont = df['Protein.Group'].str.contains("Cont_")
        logger.info(f"Removing {df[condition_cont].shape[0]} proteins considered contaminants")
        df = df[~condition_cont]
    if filter_nan_genes:
        condition_na = df['Genes'].isna()
        logger.info(f"Filtering {df[condition_na].shape[0]} genes that are NaN, {df[condition_na]['Protein.Names'].tolist()}")
        df = df[~condition_na]

    ### numerical data ###
    dft= df.T.copy()
    dft.columns = dft.loc["Protein.Group",:]
    dft.index.name = "Sample_filepath"
    rawdata = dft.iloc[n_of_protein_metadata_cols:,:]
    logger.info(f" Data comprises {rawdata.shape[0]} samples, and {rawdata.shape[1]} proteins ")

    ### protein metadata ###
    protein_metadata = df.iloc[:,:n_of_protein_metadata_cols]
    protein_metadata['Genes_simplified'] = [gene.split(";")[0] for gene in protein_metadata['Genes'].tolist()]
    protein_metadata.index = protein_metadata['Genes_simplified']
    logger.info(f"{protein_metadata[protein_metadata['Genes'].str.contains(";")].shape[0]} gene lists (eg 'TMA7;TMA7B') were simplified to their first element ('TMA7').")
    protein_metadata.index.name = "Gene" #changed index name to Gene, instead of Genes to avoid conflicts writing the data

    #load sample metadata
    sample_metadata = pd.read_csv(metadata_path, sep=metadata_sep)
    assert metadata_filepath_header in sample_metadata.columns, f"ERROR: {metadata_filepath_header} column not found in metadata file. Please check your files."
    assert sample_metadata[metadata_filepath_header].nunique() == sample_metadata.shape[0], f"ERROR: {metadata_filepath_header} has duplicates. I should not. Please check your files."
    sample_metadata.index = sample_metadata[metadata_filepath_header]
    sample_metadata = sample_metadata.drop(metadata_filepath_header, axis=1)

    # check sample_metadata filename_paths are unique, and matches df
    if not set(sample_metadata.index) == set(rawdata.index): 
        logger.warning("unique values from sample metadata and DIANN table do not match")
        logger.warning("consider double checking 'n_of_protein_metadata_cols', it varies per DIANN version")
        raise ValueError("uniques don't match")
    
    if not rawdata.shape[0] == sample_metadata.shape[0]:
        logger.error(f"ERROR: Number of samples in DIANN output {rawdata.shape[0]} and metadata {sample_metadata.shape[0]} do not match. Please check your files.")

    # reindex to match rawdata to sample metadata
    sample_metadata_aligned = sample_metadata.reindex(rawdata.index)

    # create adata object
    adata = ad.AnnData(X=rawdata.values.astype(np.float64), obs=sample_metadata_aligned, var=protein_metadata)
    logger.success("Anndata object has been created :) ")
    return adata

def perseus_to_anndata(path_to_perseus_txt):
    from perseuspy import pd
    df = pd.read_perseus(path_to_perseus_txt)
    # get obs headers
    obs_headers = list(df.columns.names)
    # get obs contents
    obs = [col for col in df.columns.values] #tuples
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

def nan_difference( array1, array2):
    """
    Calculate how many NaNs do not match between two arrays.
    Good quality control, since this can happen.
    """
    assert array1.shape == array2.shape
    total = array1.shape[0] * array1.shape[1]

    print("how many nans are not matched between arrays?")
    nan_mask1 = np.isnan(array1)
    nan_mask2 = np.isnan(array2)

    #True only if True,False or False,True. True True, or False False will be False.
    mismatch = np.logical_xor(nan_mask1, nan_mask2) & np.logical_or(nan_mask1, nan_mask2)
    print(f"Number of NaNs not matching: {np.sum(mismatch)}") 
    print(f"{np.sum(mismatch)*100/total} % of entire table")