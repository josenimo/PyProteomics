def filter_invalid_proteins(adata, grouping, threshold):
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-03-28
    Retired on 28.03.2024

    Description:
        Filter out proteins that have a NaN proportion above the threshold, for each group in the grouping variable.
    Variables:
        adata: anndata object
        grouping: string, name of the column in adata.obs that contains the groups
        threshold: float, between 0 and 1, proportion of valid values above which a protein is considered valid
    Returns:
        adata: anndata object, filtered
    """

    #TODO save dataframe with all the proteins, a column stating if they are valid or not, and the proportion of NaNs for each group


    print(f"------------------Filter Invalid Proteins ------------------")
    print(f"Keeping proteins that have at least {threshold} valid values for any group in {grouping}")
    
    #global variable to store the names of the proteins that should be kept
    valid_protein_names = []
    # dataframe to store the report
    df_report = pd.DataFrame(columns=['Group', 'Invalid_proteins', 'Valid_proteins'])

    for group in adata.obs[grouping].unique():
        # filter adata by group
        adata_group = adata[adata.obs[grouping] == group]
        # array from adata
        protein_data_group = adata_group.X
        # create a dataframe with the nan proportions for each protein
        # 0.6 means that 60% of the samples are NaNs
        nan_proportions = pd.DataFrame(np.isnan(protein_data_group).mean(axis=0), columns=['nan_proportion'])
        # get the proteins that have a NaN proportion below the threshold
        nan_proportions_T_F = nan_proportions['nan_proportion'] <= (1.0-threshold)
        #dataframe for reporting the results
        df_report.loc[len(df_report)] = [group,(len(nan_proportions_T_F) - nan_proportions_T_F.sum()), (nan_proportions_T_F.sum())]
        # get the proteins that should be kept
        valid_proteins_group = adata.var[nan_proportions_T_F.values]
        # add the proteins to the global variable
        valid_protein_names.extend(list(set(valid_proteins_group.index)))
        
    # Get the unique set of filtered proteins across groups
    valid_proteins_unique = list(set(valid_protein_names))
    # filter proteins
    adata = adata[:, valid_proteins_unique]
    print(tabulate(df_report, headers='keys', tablefmt='psql', showindex=False))
    print(f"The output object has {adata.shape[1]} proteins")
    return adata