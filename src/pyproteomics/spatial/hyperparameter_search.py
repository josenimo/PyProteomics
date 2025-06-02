import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from pyproteomics.plotting.plot_graph_network import plot_graph_network

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