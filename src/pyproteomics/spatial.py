from tqdm import tqdm
from loguru import logger
import numpy as np
import pandas as pd
import libpysal as lps
from libpysal.weights import DistanceBand
from esda import moran, geary

def run_spatial_autocorrelation(adata, method="moran", x_y=['x_centroid', 'y_centroid'], k=8, threshold=10):
    """
    Compute spatial autocorrelation statistics (Moran's I or Geary's C) for each gene in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix where observations are cells and variables are genes.
    method : str, optional (default: 'moran')
        Which spatial statistic to compute. Options are 'moran' or 'geary'.
    x_y : list of str, optional (default: ['x_centroid', 'y_centroid'])
        List containing the column names in `adata.obs` that correspond to the x and y spatial coordinates.
    k : int, optional (default: 8)
        Number of nearest neighbors to use for Moran's I (ignored if method='geary').
    threshold : float, optional (default: 10)
        Distance threshold for neighbors for Geary's C (ignored if method='moran').

    Returns
    -------
    None
        Adds new columns to `adata.var` with the computed statistic and p-values.
    """

    logger.info(f"Starting spatial autocorrelation calculation using method: {method.upper()}")
    logger.info(f"adata shape: obs={adata.n_obs}, var={adata.n_vars}")
    coords = adata.obs[x_y].values

    if method.lower() == "moran":
        logger.info(f"Building KNN graph with k={k}")
        w = lps.weights.KNN.from_array(coords, k=k)
        w.transform = "r"
    elif method.lower() == "geary":
        logger.info(f"Building DistanceBand graph with threshold={threshold}")
        w = DistanceBand.from_array(coords, threshold=threshold, binary=True)
    else:
        raise ValueError("Method must be 'moran' or 'geary'.")

    results = []
    failed_genes = []

    logger.info(f"Starting calculation for {adata.n_vars} genes")
    for gene in tqdm(adata.var.index, desc=f"Running {method.title()}", leave=True):
        feature_values = adata[:, gene].X.flatten()
        try:
            if method.lower() == "moran":
                result = moran.Moran(feature_values, w)
            elif method.lower() == "geary":
                result = geary.Geary(feature_values, w)
            results.append(result)
        except Exception as e:
            results.append(np.nan)
            failed_genes.append(gene)
            logger.warning(f"Failed processing gene {gene}: {e}")

    if failed_genes:
        logger.warning(f"{len(failed_genes)} genes failed during {method.upper()} calculation")

    if method.lower() == "moran":
        adata.var['Moran_I'] = pd.Series(
            [r.I if pd.notna(r) else np.nan for r in results], 
            index=adata.var.index
        )
        adata.var['Moran_p_sim'] = pd.Series(
            [r.p_sim if pd.notna(r) else np.nan for r in results], 
            index=adata.var.index
        )
        adata.var['Moran_Zscore'] = pd.Series(
            [r.z_sim if pd.notna(r) else np.nan for r in results], 
            index=adata.var.index
        )

    elif method.lower() == "geary":
        adata.var[f'Geary_C_k{threshold}'] = pd.Series(
            [r.C if pd.notna(r) else np.nan for r in results], 
            index=adata.var.index
        )
        adata.var[f'Geary_p_sim_k{threshold}'] = pd.Series(
            [r.p_sim if pd.notna(r) else np.nan for r in results], 
            index=adata.var.index
        )

    logger.success(f"Finished spatial autocorrelation ({method.upper()}) computation.")