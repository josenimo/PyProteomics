import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_dual_axis_boxplots(
    adata_obs,
    feature_key="RCN",
    feature_1="Proteins.Identified",
    feature_2="Precursors.Identified",
    ylabel1="Proteins Identified",
    ylabel2="Precursors Identified",
    offset=0.1,
    width=0.2,
    point_alpha=0.2,
    box1_color='skyblue',
    box2_color='lightcoral',
    median_color='black',
    scatter_color='black',
    tick1_color='blue',
    tick2_color='red',
    figsize=(6, 6),
    show_plot=True
):
    """
    Generates a dual-axis plot with boxplots and stripplots for two features
    grouped by a specified feature key.

    Args:
        adata_obs (pd.DataFrame): DataFrame typically derived from an AnnData object's
                                  observation metadata (adata.obs). Expected to contain
                                  columns specified in feature_key, feature_1, and feature_2.
        feature_key (str): Column name to group by (e.g., "RCN").
        feature_1 (str): Column name for the first feature to plot on the left y-axis.
        feature_2 (str): Column name for the second feature to plot on the right y-axis.
        ylabel1 (str): Label for the left y-axis.
        ylabel2 (str): Label for the right y-axis.
        offset (float): Offset for positioning the boxplots side-by-side.
        width (float): Width of the boxplots.
        point_alpha (float): Alpha transparency for the scatter plot points.
        box1_color (str): Face color for the boxplots of feature_1.
        box2_color (str): Face color for the boxplots of feature_2.
        median_color (str): Color of the median line in boxplots.
        scatter_color (str): Color of the points in stripplots.
        tick1_color (str): Color of the left y-axis tick labels and axis label.
        tick2_color (str): Color of the right y-axis tick labels and axis label.
        figsize (tuple): Figure size (width, height).
        show_plot (bool): If True, displays the plot. Otherwise, a figure and axes
                          objects are returned.

    Returns:
        tuple or None: If show_plot is False, returns the figure and axes (ax1, ax2).
                       Otherwise, returns None.
    """

    # Prepare data
    df = adata_obs.copy()
    df = df[[feature_key, feature_1, feature_2]]
    df = df.melt(id_vars=feature_key, var_name="variable", value_name="value")

    groups = df[feature_key].unique()
    # Sort groups if they are sortable (e.g., numeric or naturally sortable strings)
    try:
        groups = sorted(groups)
    except TypeError:
        pass # Cannot sort, use original order
    x_base = np.arange(len(groups))
    group_to_x = {group: i for i, group in enumerate(groups)}

    # Split data
    df1 = df[df["variable"] == feature_1]
    df2 = df[df["variable"] == feature_2]

    # Start plot
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Boxplots + Stripplots: loop through groups
    for group in groups:
        x_pos = group_to_x[group]
        x1_box = x_pos - offset
        x2_box = x_pos + offset

        # Get data
        y1 = df1[df1[feature_key] == group]["value"].dropna() # dropna to avoid issues with boxplot
        y2 = df2[df2[feature_key] == group]["value"].dropna() # dropna to avoid issues with boxplot

        if not y1.empty:
            # Boxplots for feature 1
            ax1.boxplot(y1, positions=[x1_box], widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor=box1_color, alpha=0.6),
                        medianprops=dict(color=median_color),
                        showfliers=False)
            # Stripplots for feature 1
            ax1.scatter(np.random.normal(x1_box, 0.03, size=len(y1)), y1,
                        color=scatter_color, alpha=point_alpha, s=10, zorder=3) # zorder to draw on top

        if not y2.empty:
            # Boxplots for feature 2
            ax2.boxplot(y2, positions=[x2_box], widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor=box2_color, alpha=0.6),
                        medianprops=dict(color=median_color),
                        showfliers=False)
            # Stripplots for feature 2
            ax2.scatter(np.random.normal(x2_box, 0.03, size=len(y2)), y2,
                        color=scatter_color, alpha=point_alpha, s=10, zorder=3) # zorder to draw on top

    # Formatting
    ax1.set_xticks(x_base)
    ax1.set_xticklabels(groups)
    ax1.set_xlabel(feature_key) # Added x-axis label based on feature_key
    ax1.set_ylabel(ylabel1, color=tick1_color)
    ax2.set_ylabel(ylabel2, color=tick2_color)
    ax1.tick_params(axis='y', labelcolor=tick1_color)
    ax2.tick_params(axis='y', labelcolor=tick2_color)

    # Adjust x-axis limits to give some padding
    ax1.set_xlim(x_base[0] - width - offset*2, x_base[-1] + width + offset*2)

    # Ensure y-axes start from 0 if data is non-negative, or adjust as needed
    if not df1["value"].empty:
        ax1_min = df1["value"].min()
        ax1_max = df1["value"].max()
        ax1.set_ylim(min(0, ax1_min - (ax1_max - ax1_min) * 0.05), ax1_max + (ax1_max - ax1_min) * 0.05)

    if not df2["value"].empty:
        ax2_min = df2["value"].min()
        ax2_max = df2["value"].max()
        ax2.set_ylim(min(0, ax2_min - (ax2_max - ax2_min) * 0.05), ax2_max + (ax2_max - ax2_min) * 0.05)


    plt.title(f"{feature_1} and {feature_2} by {feature_key}") # Added a title
    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    else:
        return fig, ax1, ax2