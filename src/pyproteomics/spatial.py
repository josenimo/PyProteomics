import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
from libpysal.weights import DistanceBand
import networkx as nx
import esda
from libpysal.weights import DistanceBand, KNN
from tqdm import tqdm

def spatial_autocorrelation(
    adata,
    method="moran",
    x_y=['x_centroid', 'y_centroid'],
    k=8,
    threshold=10,
    island_threshold=0.1,
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

    Returns
    -------
    None
    """

    logger.info(f"Starting spatial autocorrelation: {method.upper()}")
    logger.info(f"adata shape: obs={adata.n_obs}, var={adata.n_vars}")
    coords = adata.obs[x_y].values

    if method.lower() == "moran":
        logger.info(f"Building KNN graph with k={k}")
        w = KNN.from_array(coords, k=k)
        w.transform = "r"
    elif method.lower() == "geary":
        logger.info(f"Building DistanceBand graph with threshold={threshold}")
        w = DistanceBand.from_array(coords, threshold=threshold, binary=True)

        # Check islands
        n_islands = sum(1 for neighbors in w.neighbors.values() if len(neighbors) == 0)
        frac_islands = n_islands / w.n
        logger.info(f"Detected {n_islands} islands ({frac_islands:.2%} of samples)")

        if frac_islands > island_threshold:
            logger.warning(f"Too many islands (> {island_threshold:.0%}). Consider adjusting the threshold.")
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
                result = esda.moran.Moran(feature_values, w)
            elif method.lower() == "geary":
                result = esda.geary.Geary(feature_values, w)
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


def hyperparameter_search_threshold(adata, x_y=['x_centroid', 'y_centroid'], threshold_range=np.arange(1, 100, 1), loguru_logger=None, return_df=False, plot_network_at=None):
    """
    Perform a hyperparameter search over a range of threshold values to determine the number of connected nodes and average neighbors
    for different threshold values, and optionally plot the network of connected nodes at a given threshold.

    Parameters:
    - adata : AnnData object
        Spatially indexed data.
    - x_y : list of str, optional (default=['x_centroid', 'y_centroid'])
        Column names in adata.obs representing the spatial coordinates.
    - threshold_range : array-like, optional (default=np.arange(1, 100, 1))
        Range of threshold values to test.
    - loguru_logger : logger, optional
        Loguru logger for logging information during the process.
    - plot_network_at : int or None, optional (default=None)
        The threshold value at which to plot the network of connected nodes. If None, no plot is generated.

    Returns:
    - threshold_stats : pandas.DataFrame
        Dataframe containing statistics for each threshold value:
        - 'threshold': Threshold value.
        - 'num_connected_nodes': Number of nodes that are connected (not islands).
        - 'avg_neighbors': Average number of neighbors per sample.
    - fig, ax : matplotlib Figure and Axes
        Plot showing the relationship between threshold, number of connected nodes (percentage), and average neighbors.
    """
    
    # Initialize a list to store the stats for each threshold
    stats = []

    coords = adata.obs[x_y].values
    total_nodes = len(coords)  # Total number of nodes

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

        # Optionally plot the graph network for a particular threshold
        if plot_network_at == threshold:
            plot_graph_network(w, coords, threshold)

    # Convert the stats list to a DataFrame
    threshold_stats = pd.DataFrame(stats)

    # Plot the results (connected nodes as percentage)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot percentage of connected nodes on primary axis
    threshold_stats['connected_percentage'] = (threshold_stats['num_connected_nodes'] / total_nodes) * 100
    ax1.plot(threshold_stats['threshold'], threshold_stats['connected_percentage'], 'b-', label='Connected Nodes (%)')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Percentage of Connected Nodes', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot average number of neighbors on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(threshold_stats['threshold'], threshold_stats['avg_neighbors'], 'r-', label='Avg Neighbors')
    ax2.set_ylabel('Average Number of Neighbors', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set title and show plot
    plt.title("Hyperparameter Search: Threshold vs Connected Nodes (%) and Avg Neighbors")
    fig.tight_layout()

    if return_df:
        threshold_stats


def plot_graph_network(w, coords, threshold):
    """
    Plot the graph of connected nodes for a given threshold.

    Parameters:
    - w : libpysal.weights.DistanceBand object
        The distance band weights object.
    - coords : array-like
        The coordinates of the points.
    - threshold : float
        The threshold used to create the DistanceBand object.
    """
    # Create a network graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(len(coords)))

    # Add edges based on the distance threshold
    for i in range(len(coords)):
        for neighbor in w.neighbors[i]:
            G.add_edge(i, neighbor)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}  # Positions for nodes based on coordinates
    nx.draw(G, pos, with_labels=False, node_size=30, node_color='blue', alpha=0.5, edge_color='gray', width=0.5)
    plt.title(f"Graph of Connected Nodes at Threshold {threshold}")
    plt.show()