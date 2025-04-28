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


def hyperparameter_search_threshold(adata, x_y=['x_centroid', 'y_centroid'], threshold_range=np.arange(1, 100, 1), loguru_logger=None, return_df=False):
    """
    Perform a hyperparameter search over a range of threshold values to determine the number of connected nodes and average neighbors
    for different threshold values.

    Parameters:
    - adata : AnnData object
        Spatially indexed data.
    - x_y : list of str, optional (default=['x_centroid', 'y_centroid'])
        Column names in adata.obs representing the spatial coordinates.
    - threshold_range : array-like, optional (default=np.arange(1, 100, 1))
        Range of threshold values to test.
    - loguru_logger : logger, optional
        Loguru logger for logging information during the process.

    Returns:
    - threshold_stats : pandas.DataFrame
        Dataframe containing statistics for each threshold value:
        - 'threshold': Threshold value.
        - 'num_connected_nodes': Number of nodes that are connected (not islands).
        - 'avg_neighbors': Average number of neighbors per sample.
    - fig, ax : matplotlib Figure and Axes
        Plot showing the relationship between threshold, number of connected nodes, and average neighbors.
    """

    # Initialize a list to store the stats for each threshold
    stats = []

    coords = adata.obs[x_y].values

    for threshold in threshold_range:
        # Compute the spatial weights for each threshold
        w = DistanceBand.from_array(coords, threshold=threshold, binary=True)

        # Calculate the number of connected nodes (not islands)
        num_connected_nodes = sum([len(w.neighbors[i]) > 0 for i in range(len(coords))])

        # Calculate the average number of neighbors
        avg_neighbors = np.mean([len(w.neighbors[i]) for i in range(len(coords))])

        stats.append({
            'threshold': threshold,
            'num_connected_nodes': num_connected_nodes,
            'avg_neighbors': avg_neighbors
        })

        # Log the threshold being processed if logger is provided
        if loguru_logger:
            loguru_logger.info(f"Processed threshold {threshold}, connected nodes: {num_connected_nodes}, avg neighbors: {avg_neighbors}")

    # Convert the stats list to a DataFrame
    threshold_stats = pd.DataFrame(stats)

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot number of connected nodes on primary axis
    ax1.plot(threshold_stats['threshold'], threshold_stats['num_connected_nodes'], 'b-', label='Connected Nodes')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Number of Connected Nodes', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot average number of neighbors on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(threshold_stats['threshold'], threshold_stats['avg_neighbors'], 'r-', label='Avg Neighbors')
    ax2.set_ylabel('Average Number of Neighbors', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set title and show plot
    plt.title("Hyperparameter Search: Threshold vs Connected Nodes and Avg Neighbors")
    fig.tight_layout()
    plt.show()

    if return_df:
        return threshold_stats