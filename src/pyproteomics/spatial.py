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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
import libpysal as lps
from libpysal.weights import DistanceBand
from esda import moran, geary


def run_spatial_autocorrelation_v2(
    adata,
    method="moran",
    x_y=['x_centroid', 'y_centroid'],
    k=8,
    threshold=10,
    island_threshold=0.05,  # 5% of samples being islands triggers search
    search_thresholds=None
):
    """
    Compute spatial autocorrelation statistics (Moran's I or Geary's C) for each gene in an AnnData object,
    with automatic threshold search if too many islands (disconnected samples) are detected.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix where observations are cells and variables are genes.
    method : str, optional
        Spatial statistic to compute: 'moran' or 'geary'.
    x_y : list of str, optional
        Names of columns in `adata.obs` containing spatial coordinates.
    k : int, optional
        Number of neighbors for Moran's I (ignored for Geary's C).
    threshold : float, optional
        Distance threshold for neighbors for Geary's C (ignored for Moran's I).
    island_threshold : float, optional
        If more than this fraction of samples are islands (no neighbors), hyperparameter search is triggered.
    search_thresholds : list or None
        Threshold values to test if hyperparameter search is needed. Defaults to np.linspace(10, 100, 10).

    Returns
    -------
    None
    """

    logger.info(f"Starting spatial autocorrelation: {method.upper()}")
    logger.info(f"adata shape: obs={adata.n_obs}, var={adata.n_vars}")
    coords = adata.obs[x_y].values

    if method.lower() == "moran":
        logger.info(f"Building KNN graph with k={k}")
        w = lps.weights.KNN.from_array(coords, k=k)
        w.transform = "r"
    elif method.lower() == "geary":
        logger.info(f"Building DistanceBand graph with threshold={threshold}")
        w = DistanceBand.from_array(coords, threshold=threshold, binary=True)

        # Check islands
        n_islands = sum(1 for neighbors in w.neighbors.values() if len(neighbors) == 0)
        frac_islands = n_islands / w.n
        logger.info(f"Detected {n_islands} islands ({frac_islands:.2%} of samples)")

        if frac_islands > island_threshold:
            logger.warning(f"Too many islands (> {island_threshold:.0%}). Running hyperparameter search...")
            if search_thresholds is None:
                search_thresholds = np.linspace(threshold, threshold * 5, 10)
            
            results = []
            for t in tqdm(search_thresholds, desc="Searching thresholds", leave=True):
                temp_w = DistanceBand.from_array(coords, threshold=t, binary=True)
                temp_n_islands = sum(1 for neighbors in temp_w.neighbors.values() if len(neighbors) == 0)
                temp_frac_islands = temp_n_islands / temp_w.n
                results.append((t, temp_frac_islands))
            
            thresholds, frac_islands_list = zip(*results)

            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(thresholds, np.array(frac_islands_list)*100, marker='o')
            ax.axhline(y=island_threshold*100, color='red', linestyle='--', label=f'Island Threshold ({island_threshold:.0%})')
            ax.set_xlabel('Distance Threshold')
            ax.set_ylabel('Percentage of Islands (%)')
            ax.set_title('Island percentage vs Distance Threshold')
            ax.legend()
            plt.tight_layout()
            plt.show()

            logger.info("Threshold search completed. Check the plot to select a better threshold.")
            return

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