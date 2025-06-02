import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def plot_volcano_v2(adata, 
                    x="log2_FC", 
                    y="-log10(p_val_corr)_BH", 
                    significant=True, 
                    FDR=None, 
                    tag_top=None, 
                    group1=None, 
                    group2=None,
                    return_fig=False):
    """
    Plot a volcano plot from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        An AnnData object with statistical test results stored in `adata.var`.
    x : str
        Column name in `adata.var` for the x-axis (e.g., log2 fold change).
    y : str
        Column name in `adata.var` for the y-axis (e.g., -log10 p-value).
    significant : bool
        Whether to highlight significant points based on FDR threshold.
    FDR : float or None
        Threshold for corrected p-value (e.g., 0.05). Required if `significant=True`.
    tag_top : int or None
        Number of top hits (by `y`) to label with text, regardless of significance.
    group1 : str or None
        Name of the first group (used for x-axis label annotation).
    group2 : str or None
        Name of the second group (used for x-axis label annotation).
    return_fig : bool
        If True, returns the matplotlib `fig` object for further modification.

    Returns
    -------
    matplotlib.figure.Figure or None
        The `fig` object if `return_fig=True`, otherwise None.
    """
    adata_copy = adata.copy()
    df = adata_copy.var

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=df[x], y=df[y], ax=ax, color='gray', s=20, edgecolor=None)

    if group1 is not None and group2 is not None:
        ax.set_xlabel(f"Difference in mean protein expression (log2)\n{group1} (right) vs {group2} (left)")
    else:
        ax.set_xlabel(x)

    ax.set_ylabel("-log10 corrected p-value BH")

    # Highlight significant points
    if significant:
        if FDR is None:
            raise ValueError("FDR must be specified if significant=True.")
        sig_df = df[df["p_val_corr_BH"] < FDR]
        ax.scatter(sig_df[x], sig_df[y], color="red", s=20)

    # Label top N points by y-axis, regardless of significance
    if tag_top:
        tag_df = df.sort_values(by=y, ascending=False).head(tag_top)

        df_left = tag_df[tag_df[x] < 0]
        df_right = tag_df[tag_df[x] > 0]

        texts_left = [
            ax.text(row[x], row[y], idx, ha='right', va='center', fontsize=8)
            for idx, row in df_left.iterrows()
        ]
        texts_right = [
            ax.text(row[x], row[y], idx, ha='left', va='center', fontsize=8)
            for idx, row in df_right.iterrows()
        ]

        adjust_text(
            texts_left + texts_right,
            ax=ax,
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=0.5)
        )

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.2)
    ax.grid(False)

    if return_fig:
        return fig
    else:
        plt.show()
