import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import numpy as np

def rankplot(adata, condition, protein):

    rankdata = adata.copy()

    for type in condition:
        rankdata.var[f"mean_{type}"] = np.nanmean(rankdata.layers["log2_trans"][rankdata.obs["condition"] == type], axis=0)
        rankdata.var = rankdata.var.sort_values(by=f"mean_{type}", ascending=False)
        rankdata.var[f"rank_{type}"] = range(0, len(rankdata.var))

    if len(protein)>0:
        for p in protein:
            protein_filtered = rankdata.var.loc[rankdata.var["PG.Genes"] == p]
            for type in condition:
                plt.annotate(f"{p} in {type}", 
                            xy=(protein_filtered[f"rank_{type}"], protein_filtered[f"mean_{type}"]), 
                            xytext=(protein_filtered[f"rank_{type}"]+200, protein_filtered[f"mean_{type}"]),
                            arrowprops=dict(arrowstyle = "-"))
            
        for type in condition:
            plt.scatter(rankdata.var[f"rank_{type}"], rankdata.var[f"mean_{type}"], s=2, label=f"{type}")
    
    else:
        for type in condition:
            plt.scatter(rankdata.var[f"rank_{type}"], rankdata.var[f"mean_{type}"], s=0.5, label=f"Cell type {type}")

    plt.title('Rank Plot')
    plt.xlabel('Protein Rank')
    plt.ylabel('-log2(Mean Intensity)')
    plt.legend()
    plt.grid()

def rankplot_byJose(adata, key_adataobs, val_adataobs, key_adatavar, label_adatavar ):
    
    adata_copy = adata.copy()
    
    assert key_adatavar in adata_copy.var.columns, f"{key_adatavar} not in adata.var"

    if key_adataobs is not None:
        assert key_adataobs in adata_copy.obs.columns, f"{key_adataobs} not in adata.obs"
        adata_copy = adata_copy[adata_copy.obs[key_adataobs] == val_adataobs].copy()

    df_tmp = pd.DataFrame(data=adata_copy.X, columns=adata_copy.var[key_adatavar])
    df_tmp = df_tmp.T
    df_tmp['mean'] = df_tmp.mean(axis=1)
    df_tmp.sort_values(by='mean', ascending=False, inplace=True)
    df_tmp.reset_index(inplace=True)
    df_tmp['rank'] = df_tmp.index +1
        
    #plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_tmp, x='rank', y='mean', hue='mean', palette='flare', ax=ax, linewidth=0)
    ax.get_legend().remove()

    if label_adatavar is not None:
        
        texts = []
        for gene in label_adatavar:
            x = df_tmp.loc[df_tmp[key_adatavar] == gene, 'rank']
            y = df_tmp.loc[df_tmp[key_adatavar] == gene, 'mean']
            texts.append(ax.text(x, y, gene, ha='center', va='center'))
        adjust_text(texts, expand=(1, 3), arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.show()


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


def coefficient_of_variation(adata, group_by):
    adata_copy = adata.copy()
    assert group_by in adata_copy.obs.columns, f"{group_by} not found in adata.obs"

    # Temporary DataFrame to store CV values for plotting
    df_tmp = pd.DataFrame()

    for group in adata.obs[group_by].unique():
        
        # Extract group-specific data
        adata_group = adata_copy[adata_copy.obs[group_by] == group].copy()
        
        # Warn if less than 3 samples in the group
        print(f"this group has {adata_group.shape[0]} samples")
        if adata_group.shape[0] < 3:
            print(f"{group} in dataset has less than 3 samples, leading to poor statistics")
        
        # Calculate mean, std, and cv for each feature in the group
        means = np.mean(adata_group.X, axis=0) #does this ignore NaNs? 
        stds = np.std(adata_group.X, axis=0)
        cvs = stds / means  # CV = std / mean

        # Store results in adata_copy.var
        adata_copy.var[f"{group}_mean"] = means
        adata_copy.var[f"{group}_std"] = stds
        adata_copy.var[f"{group}_cv"] = cvs

        # Append to df_tmp for plotting
        group_df = pd.DataFrame({f"{group}_cv": cvs, group_by: group})
        df_tmp = pd.concat([df_tmp, group_df], ignore_index=True)

    # Plot using seaborn
    df_tmp = df_tmp.melt(id_vars=group_by, var_name='metric', value_name='cv')
    sns.boxplot(data=df_tmp, y="cv", hue=group_by, width=0.3)