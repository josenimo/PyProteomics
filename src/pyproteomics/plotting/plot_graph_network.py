import matplotlib.pyplot as plt
import networkx as nx

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