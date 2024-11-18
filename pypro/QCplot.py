import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



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