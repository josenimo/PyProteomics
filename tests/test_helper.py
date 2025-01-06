import os, sys
import pandas as pd
import numpy as np
import anndata as ad
import time

from loguru import logger

import pyproteomics.helper
import pytest

def test_DIANN_to_adata():

    start_time = time.time()

    #Arrange
    adata = helper.DIANN_to_adata( 
        DIANN_path = "./tests/data/20241211_DIANN.pg_matrix.tsv",
        DIANN_sep  = "\t",
        metadata_path = "./tests/data/20241217_DIANN_metadata.csv", 
        metadata_sep  = ";", 
        metadata_check = False, 
        sample_id_column = "LCMS_run_id" )
    
    #Act
    adata_shape = adata.shape
    adata_obs = adata.obs
    adata_var = adata.var

    adata_expected_shape = (248, 7030)  # Current known data shape
    adata_obs_expected_shape = (248, 36)  # Current known obs shape
    adata_var_expected_shape = (7030, 3)  # Current known var shape

    #Assert
    assert isinstance(adata, ad.AnnData), f"Expected ad.AnnData, got {type(adata)}"
    assert adata_shape == adata_expected_shape, f"Excepted shape {adata_expected_shape}, but got {adata_shape}"
    assert adata_obs.shape == adata_obs_expected_shape, f"Excepted obs shape {adata_obs_expected_shape}, but got {adata_obs.shape}"
    assert adata_var.shape == adata_var_expected_shape, f"Excepted var shape {adata_var_expected_shape}, but got {adata_var.shape}"
    
    #Cleanup
    del adata

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time}")