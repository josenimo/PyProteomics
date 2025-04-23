import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def coefficient_of_variation(df, axis=0, nan_policy='propagate'):
    """
    Calculate the coefficient of variation (CV = std / mean) along a specified axis of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - axis (int): 0 column-wise CV, 1 row-wise CV.
    - nan_policy (str): {'propagate', 'raise', 'omit'}
        - 'propagate': returns NaN if NaN is present
        - 'raise': raises ValueError if NaN is present
        - 'omit': ignores NaNs in the calculation

    Returns:
    - pd.Series: CV values for each row or column.

    Raises:
    - ValueError: if nan_policy='raise' and NaNs are present
    """
    if nan_policy not in {'propagate', 'raise', 'omit'}:
        raise ValueError("nan_policy must be 'propagate', 'raise', or 'omit'")

    if nan_policy == 'raise' and df.isna().any().any():
        raise ValueError("NaN values found in DataFrame and nan_policy is set to 'raise'")

    if nan_policy == 'omit':
        mean = df.mean(axis=axis, skipna=True)
        std = df.std(axis=axis, skipna=True)
    else:  # 'propagate'
        mean = df.mean(axis=axis, skipna=False)
        std = df.std(axis=axis, skipna=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std / mean
        cv.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cv

def bootstrap_variability(
    dataframe,
    n_bootstrap=100,
    subset_sizes=[10, 50, 100],
    summary_func=np.mean,
    return_raw=False,
    return_summary=True,
    plot=True,
    random_seed=42,
    nan_policy="omit",
):

    """
    Evaluate the variability of feature-level coefficient of variation (CV) via bootstrapping.

    This function samples subsets from the input DataFrame and computes the CV (standard deviation divided by mean)
    of each feature (column) for each bootstrap replication. For each subset size, the function aggregates the CVs
    across bootstraps and then summarizes them with a user-specified statistic (e.g., mean, median). Optionally,
    the function can generate a violin plot of the summarized CVs across different subset sizes, and it returns the
    bootstrapped raw CVs and/or the summarized results.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data (features in columns, samples in rows).
    n_bootstrap : int, optional (default=100)
        Number of bootstrap replicates to perform for each subset size.
    subset_sizes : list of int, optional (default=[10, 50, 100])
        List of subset sizes (number of rows to sample) to use during the bootstrapping.
    summary_func : callable, optional (default=np.mean)
        Function to aggregate the per-feature CVs across bootstraps. For example, np.mean, np.median, etc.
    return_raw : bool, optional (default=True)
        If True, returns the raw bootstrapped CVs in long format.
    return_summary : bool, optional (default=True)
        If True, returns a summary DataFrame where the per-feature bootstrapped CVs have been aggregated using
        `summary_func` for each subset size.
    plot : bool, optional (default=True)
        If True, displays a violin plot of the summarized CVs (one aggregated value per feature) across subset sizes.
    random_seed : int or None, optional (default=42)
        Seed for the random number generator, ensuring reproducibility.
    nan_policy : str, optional (default="omit")
        How to handle NaN values. Options are:
            - "omit": ignore NaNs during calculations,
            - "raise": raise an error if NaNs are encountered,
            - "propagate": allow NaNs to propagate in the output.

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        Depending on the flags `return_raw` and `return_summary`, the function returns:
            - If both are True: a tuple (raw_df, summary_df)
              * raw_df: DataFrame in long format with columns "feature", "cv", "subset_size", and "bootstrap_id".
              * summary_df: DataFrame with the aggregated CV (using `summary_func`) per feature and subset size,
                with columns "subset_size", "feature", and "cv_summary".
            - If only one of the flags is True, only that DataFrame is returned.
            - If neither is True, returns None.

    Raises
    ------
    ValueError
        If any of the specified subset sizes is larger than the number of rows in `dataframe`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(np.random.randn(100, 5))  # 100 samples, 5 features
    >>> raw_results, summary_results = bootstrap_variability(df, subset_sizes=[10, 20, 50])
    >>> summary_results.head()
         subset_size feature  cv_summary
    0           10       A    0.123456
    1           10       B    0.098765
    2           20       A    0.110987
    3           20       B    0.102345
    4           50       A    0.095432
    """

    # Safety checks
    if max(subset_sizes) > dataframe.shape[0]:
        raise ValueError("A subset size is larger than the number of rows in the dataframe.")
    rng = np.random.default_rng(seed=random_seed)
    
    all_feature_results = []

    for size in tqdm(subset_sizes, desc="Subset sizes"):
        feature_cv_list = []
        for i in tqdm(range(n_bootstrap), desc=f"Bootstraps (n={size})", leave=False):
            subset = dataframe.sample(n=size, replace=False, random_state=rng.integers(0, 1e9))
            cv = coefficient_of_variation(subset, axis=0, nan_policy=nan_policy)  # Series
            feature_cv_list.append(cv.rename(f"bootstrap_{i+1}"))

        # Combine all bootstraps into a DataFrame (features as rows, bootstraps as columns)
        feature_cvs_df = pd.concat(feature_cv_list, axis=1)
        feature_cvs_df['subset_size'] = size
        feature_cvs_df['feature'] = feature_cvs_df.index

        # Melt into long format: one row per feature-bootstrap
        melted = feature_cvs_df.drop(columns=['subset_size', 'feature']).T.melt(
            var_name='feature', value_name='cv'
        )
        melted['subset_size'] = size
        melted['bootstrap_id'] = melted.index % n_bootstrap + 1  # Optional: give bootstrap ID
        all_feature_results.append(melted)

    # Combine all subset sizes
    results_df = pd.concat(all_feature_results, ignore_index=True)

    # Summarize
    summary_df = (
        results_df.groupby(['subset_size', 'feature'])['cv']
        .agg(summary_func)
        .reset_index()
        .rename(columns={'cv': 'cv_summary'})
    )

    if plot:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=summary_df, x="subset_size", y="cv_summary")
        plt.title("Bootstrap variability across subset sizes")
        plt.xlabel("Subset size")
        plt.ylabel(f"{summary_func.__name__.capitalize()} CV per feature")
        plt.tight_layout()
        plt.show()

    if return_raw and return_summary:
        return results_df, summary_df
    elif return_summary:
        return summary_df
    elif return_raw:
        return results_df
    else:
        return None