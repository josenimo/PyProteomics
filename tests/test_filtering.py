import os, sys
import pandas as pd
import numpy as np
import anndata as ad
import time

from loguru import logger

import PyProteomics.helper
import PyProteomics.filtering
import pytest

def test_filter_proteins_without_genenames(): 

    start_time = time.time()

    #Arrange
    adata = helper.DIANN_to_adata( 
        DIANN_path = "./tests/data/20241211_DIANN.pg_matrix.tsv",
        DIANN_sep  = "\t",
        metadata_path = "./tests/data/20241217_DIANN_metadata.csv", 
        metadata_sep  = ";", 
        metadata_check = False, 
        sample_id_column = "LCMS_run_id" )
    
    adata_without_genenames = filtering.filter_proteins_without_genenames(adata)

def test_filter_out_contaminants():

    start_time = time.time()

    #Arrange
    adata = helper.DIANN_to_adata( 
        DIANN_path = "./tests/data/20241211_DIANN.pg_matrix.tsv",
        DIANN_sep  = "\t",
        metadata_path = "./tests/data/20241217_DIANN_metadata.csv", 
        metadata_sep  = ";", 
        metadata_check = False, 
        sample_id_column = "LCMS_run_id" )
    
    adata_without_contaminants = filtering.filter_out_contaminants(
        adata,
        adata_var_column_with_label="Protein.Ids",
        string_to_indicate_removal="Cont_",
        keep_genes=[],
        print_summary=False, 
        qc_export_path=None,
    )

    #Act

    #Assert

    #Cleanup

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time}")

