import numpy as np

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