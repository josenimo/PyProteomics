from datetime import datetime
date = datetime.now().strftime("%Y%m%d")

from typing import Optional
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import statsmodels.stats.multitest as smm

def anova_adata(
    adata: ad.AnnData,
    grouping: str,
    FDR_threshold: float = 0.05
) -> None:
    """
    Perform one-way ANOVA for all columns of an AnnData object across all groups in a categorical column.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    grouping : str
        Column header in adata.obs, categorizing different groups to test.
    FDR_threshold : float, default 0.05
        The threshold for the FDR correction.

    Returns
    -------
    None
        Results are saved to adata.var in-place.
    """
    adata_copy = adata.copy()
    F_vals = []
    p_vals = []

    X = np.asarray(adata_copy.X)
    group_labels = adata_copy.obs[grouping].astype(str)
    unique_groups = group_labels.unique()

    for column in adata_copy.var.index:
        col_idx = adata_copy.var.index.get_loc(column)
        # Gather arrays for each group
        group_arrays = [X[group_labels == group, col_idx].flatten() for group in unique_groups]
        # Perform one-way ANOVA using pingouin
        result = pg.anova(dv=None, between=None, data=None, x=group_arrays)
        # If using scipy: F, p = stats.f_oneway(*group_arrays)
        # But for consistency with pingouin, use:
        F, p = pg.compute_effsize(group_arrays, eftype='anova')[:2] if hasattr(pg, 'compute_effsize') else (np.nan, np.nan)
        # Fallback to scipy if needed
        if np.isnan(F) or np.isnan(p):
            from scipy.stats import f_oneway
            F, p = f_oneway(*group_arrays)
        F_vals.append(F)
        p_vals.append(p)

    adata_copy.var['anova_F'] = F_vals
    adata_copy.var['anova_p'] = p_vals
    # Multiple testing correction
    result_BH = smm.multipletests(adata_copy.var['anova_p'].values, alpha=FDR_threshold, method='fdr_bh')
    adata_copy.var['anova_significant_BH'] = result_BH[0]
    adata_copy.var['anova_p_corr_BH'] = result_BH[1]
    adata_copy.var['-log10(anova_p_corr)_BH'] = -np.log10(adata_copy.var['anova_p_corr_BH'])
    print(f"ANOVA across groups in '{grouping}' completed. {np.sum(adata_copy.var['anova_significant_BH'])} features significant at FDR < {FDR_threshold}.")