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

def impute_single(array, mean_shift, std_dev_shift, report_stats=True):
    
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