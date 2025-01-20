import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Assert that imputated values are not negative, if so prompt user to change the mean_shift value

    logger.info("Starting imputation with Gaussian distribution version 2.1")

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
        col_stddev = df[col].std(skipna=True)
        # Identify NaN positions in the column
        nan_mask = df[col].isnull()
        num_nans = nan_mask.sum() 
        # Generate random values from a normal distribution       
        shifted_random_values = np.random.normal(
            loc=(col_mean + (mean_shift * col_stddev)), 
            scale=(col_stddev * std_dev_shift), 
            size=num_nans)
        # Replace NaNs in the column with the generated random values
        df.loc[nan_mask, col] = shifted_random_values

    if perSample:
        df = df.T
    
    adata_copy.X = df.values
    logger.info(f'Number of missing values after imputation: {np.sum(np.isnan(adata_copy.X))}')
    logger.info("Imputation complete")
    return adata_copy

def impute_single_debugging(array, mean_shift, std_dev_shift, report_stats=True):
    
    array_log2 = np.log2(array)
    mean_log2 = np.nanmean(array_log2)
    stddev_log2 = np.nanstd(array_log2)
    nans = np.isnan(array_log2)
    num_nans = np.sum(nans)

    shifted_random_values_log2 = np.random.normal(
        loc=(mean_log2 + (mean_shift * stddev_log2)), 
        scale=(stddev_log2 * std_dev_shift), 
        size=num_nans)
    
    if report_stats:
        logger.debug(f"mean: {mean_log2}")
        logger.debug(f"stddev: {stddev_log2}")
        logger.debug(f"Coefficient of variation: {np.nanstd(array)/np.nanmean(array)}")
        logger.debug(f"Min  : {np.nanmin(array_log2)}")
        logger.debug(f"Max  : {np.nanmax(array_log2)}")

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    _,fixed_bins,_ = plt.hist(array_log2, bins=30)
    data = np.concatenate([array_log2, shifted_random_values_log2])
    groups = ['Raw'] * len(array_log2) + ['Imputed'] * len(shifted_random_values_log2)

    sns.boxplot(x=data, y=groups, ax=ax_box, palette=['b', 'r'], orient='h')
    sns.histplot(x=array_log2, bins=fixed_bins, kde=False, ax=ax_hist, color='b', alpha=0.8)
    sns.histplot(x=shifted_random_values_log2, bins=fixed_bins, kde=False, ax=ax_hist, color='r', alpha=0.5)

    ax_box.set(yticks=[])
    ax_box.set(xticks=[])
    ax_hist.set(yticks=[], ylabel="")

    plt.tight_layout()
    plt.show()