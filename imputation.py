#basic python
import os
import sys
from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
from tabulate import tabulate
import shutil
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#dataframe management
import numpy as np
import pandas as pd
import scanpy as sc
sc.settings.verbosity = 1
import anndata as ad
from sklearn.utils import shuffle

#math
import pingouin as pg
import statsmodels.api as sm
import statsmodels.stats.multitest
from scipy.stats import zscore
from scipy.stats import pearsonr
from scipy.stats import spearmanr 
from scipy.stats import ttest_ind
from scipy.sparse import csr_matrix
from scipy.stats import shapiro

#imputation
# from missingpy import MissForest

#plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from adjustText import adjust_text
sc.set_figure_params(dpi=150)

def imputation_gaussian(adata, mean_shift=-1.8, std_dev_shift=0.3, perSample=False):
    """
    Created by Jose Nimo on 2023-07-20
    Modified by Jose Nimo on 2023-08-16

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
    adata_copy = adata.copy()
    df = pd.DataFrame(data = adata_copy.X, columns = adata_copy.var.index, index = adata_copy.obs_names)

    if perSample:
        print(" --- --- --- Imputation with Gaussian distribution per sample is running... --- --- --- ")
        df = df.T
    else:
        print(" --- --- --- Imputation with Gaussian distribution per protein is running... --- --- --- ")

    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Calculate the mean and standard deviation for the column (protein values)
        col_mean = df[col].mean()
        col_std = df[col].std()
        # Identify NaN positions in the column
        nan_mask = df[col].isnull()
        num_nans = nan_mask.sum()
        # Generate random values from the Gaussian distribution proportional to original distribution
        # the mean is shifted by the mean shift value multiplied by the standard deviation
        random_values = np.random.randn(num_nans)
        shifted_random_values = (col_mean+(mean_shift*col_std)) + (col_std*std_dev_shift) * random_values
        # Replace NaNs in the column with the generated random values
        df.loc[nan_mask, col] = shifted_random_values

    if perSample:
        df = df.T

    adata_copy.X = df.values

    #mean number of missing values per row (sample)
    df = pd.DataFrame(data=adata.X, columns=adata.var_names, index=adata.obs_names)
    print('Mean number of missing values per sample: ', round(df.isnull().sum(axis=1).mean(),2), 'out of ', df.shape[1], ' proteins')
    print(" --- --- --- Imputation with Gaussian distribution is done! --- --- --- ")

    return adata_copy
