import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from loguru import logger

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def gaussian(adata, mean_shift=-1.8, std_dev_shift=0.3, perSample=False, qc_export_path:str=None) -> ad.AnnData:
    """
    Description:
        This function imputes missing values in the adata object using a Gaussian distribution.
        The mean and standard deviation of the Gaussian distribution are calculated for each column (protein) in the adata object.
        The mean is shifted by the mean shift value multiplied by the standard deviation.
        The standard deviation is reduced by the std_dev_shift value.
    Variables:
        adata: AnnData object
        mean_shift: float, default=-1.8
            How many standard deviations to shift the mean of the Gaussian distribution
        std_dev_shift: float, default=0.3
            How much to reduce the standard deviation of the Gaussian distribution, in terms of percentage
        perSample: bool, default=False
            Whether to impute missing values per sample or per protein
            Recommended to impute per protein when more than 12 protein values available
    Returns:
        adata_copy: AnnData object
            AnnData object with imputed values
    """
    logger.info("Starting imputation with Gaussian distribution version 2.0.0")

    adata_copy = adata.copy()
    df = pd.DataFrame(data = adata_copy.X, columns = adata_copy.var.index, index = adata_copy.obs_names)

    if perSample:
        logger.info("Imputation with Gaussian distribution PER SAMPLE")
        df = df.T
    else:
        logger.info("Imputation with Gaussian distribution PER PROTEIN")

    logger.info(f'Mean number of missing values per sample: {round(df.isnull().sum(axis=1).mean(),2)} out of {df.shape[1]} proteins')
    logger.info(f'Mean number of missing values per protein: {round(df.isnull().sum(axis=0).mean(),2)} out of {df.shape[0]} samples')

    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Calculate the mean and standard deviation for the column (protein values)
        col_mean = df[col].mean(skipna=True)
        col_std = df[col].std(skipna=True)
        # Identify NaN positions in the column
        nan_mask = df[col].isnull()
        num_nans = nan_mask.sum()
        # Generate enough random values to replace the NaNs 
        random_values = np.random.randn(num_nans)
        # the mean is shifted by the mean shift value multiplied by the standard deviation
        shifted_random_values = (col_mean+(mean_shift*col_std)) + (col_std*std_dev_shift) * random_values
        # Replace NaNs in the column with the generated random values
        df.loc[nan_mask, col] = shifted_random_values

    if perSample:
        df = df.T
    
    adata_copy.X = df.values
    logger.info(f'Number of missing values after imputation: {np.sum(np.isnan(adata_copy.X))}')
    logger.info("Imputation complete")
    return adata_copy
