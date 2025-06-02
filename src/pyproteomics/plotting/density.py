import pandas as pd
import seaborn as sns

def density_plots(adata, color_by):
    
    adata_copy = adata.copy()
    assert color_by in adata_copy.obs.columns, f"{color_by} not found in adata.obs"

    df = pd.DataFrame(data=adata_copy.X, columns=adata_copy.var_names, index=adata_copy.obs[color_by])
    df.reset_index(inplace=True)
    df = pd.melt(df, id_vars=color_by, var_name="Protein", value_name="Abundance")  
    sns.kdeplot(data=df, x="Abundance",hue=color_by, multiple="layer", common_norm=False)

    """
    Example code:
    density_plots(adata, "sample_id")
    """