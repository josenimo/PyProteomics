import pandas as pd
import numpy as np
import seaborn as sns

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