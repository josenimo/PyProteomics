from datetime import datetime
date = datetime.now().strftime("%Y%m%d")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import statsmodels.stats.multitest

# factor in a way that we can test the arrays with different tests


def Ttest_adata(adata, grouping, group1, group2, FDR_threshold=0.05):
    """
    Created by Jose Nimo on 2023-07-20
    Modified by Jose Nimo on 2023-10-02

    Description:
        This function performs a t-test for all columns of a annadata object, between two groups. 
        The groups are defined by a categorical column in adata.obs
        The t test is performed using pingouin.ttest, two-sided, not paired
        The p-values are corrected for multiple testing using statsmodels.stats.multitest.multipletests
        The correction method is Benjamini-Hochberg, with a threshold of 0.05 (default)

    Variables:
        adata: AnnData object
        grouping: str, column header in adata.obs, categorizing different groups to test
        group1: str, value in grouping column to be tested against
        group2: str, value in grouping column to be test against group 1
        FDR_threshold: float, default=0.05
            The threshold for the FDR correction

    """

    #TODO assert that adata slicing returns a non-empty object
    
    #setup
    adata_copy = adata.copy()
    t_values = []
    p_values = []
    FC = []
    
    #loop through each protein
    for column in adata_copy.var.index:
        array_1 = np.array(adata_copy[adata_copy.obs[grouping] == group1].X[:, adata_copy.var.index == column].flatten(), dtype=np.float64)
        array_2 = np.array(adata_copy[adata_copy.obs[grouping] == group2].X[:, adata_copy.var.index == column].flatten(), dtype=np.float64)
        result = pg.ttest(x=array_1, y=array_2, paired=False, alternative="two-sided", correction=False, r=0.707)
        t_values.append(result.iloc[0,0])
        p_values.append(result.iloc[0,3])
        FC.append(np.mean(array_1) - np.mean(array_2))

    #add to adata object
    adata_copy.var["t_val"] = t_values
    adata_copy.var["p_val"] = p_values
    adata_copy.var["log2_FC"] = FC
    
    #correct for multiple testing
    result_BH        = statsmodels.stats.multitest.multipletests(adata_copy.var["p_val"].values, alpha=FDR_threshold, method='fdr_bh')
    
    adata_copy.var["significant_BH"] = result_BH[0]
    adata_copy.var["p_val_corr_BH"] = result_BH[1]
    adata_copy.var['-log10(p_val_corr)_BH'] = -np.log10(adata_copy.var['p_val_corr_BH'])
    
    print("----- ----- Ttest_adata ----- -----")
    print("Testing for differential expression between {} and {}".format(group1, group2))
    print("Using pingouin.ttest to perform t-test, two-sided, not paired")
    print("Using statsmodels.stats.multitest.multipletests to correct for multiple testing")
    print("Using Benjamini-Hochberg for FDR correction, with a threshold of {}".format(FDR_threshold))
    print("The test found {} proteins to be significantly differentially expressed".format(np.sum(adata_copy.var["significant_BH"])))
    print("----- ----- Ttest_adata ----- -----\n")
    return adata_copy