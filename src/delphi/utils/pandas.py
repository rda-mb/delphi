from datetime import timedelta

from pandas import DataFrame, Series


def trim_initial_zeros(df: DataFrame, target_col: str, wait_years: int = 2) -> DataFrame:
    """Trim the initial zeros from the target column in the raw data.
    If there are more than 2 years of zeros at the beginning of the target column,
    the data before the first non-zero value is discarded.

    Args:
        df (DataFrame): DataFrame containing covariates and target series
        target_col (str): target column name
        wait_years (int): Number of years containing zeros as threshold for trimming.
    """
    first_date = df.index[0]
    # get the index of the first non-zero value in the target column
    first_non_zero_date = (df[target_col] != 0).idxmax()
    # if more than 2 years of zeros to begin with, discard this data
    if first_non_zero_date - first_date > timedelta(days=365 * wait_years):
        # trim the data before the date
        df = df.loc[first_non_zero_date:]
    return df


def trim_low_correlation_variables(
    df: DataFrame, target: Series, thresh: float = 0.0
) -> DataFrame:
    """Discards variables from a dataframe
    that correlate weakly to a given Series. Uses Pearson's correlation.

    Args:
        df (DataFrame): Covariate DataFrame
        target (Series): target Series
        thresh (float, optional): discard columns where correlation < thresh. Defaults to 0.0.

    Returns:
        DataFrame: Dataframe trimmed from low correlation variables.
    """
    if not thresh:
        return df
    # correlations between Dataframe and target series
    corrs = df.corrwith(target)
    # columns with correlation above cutoff value
    variables_above_cutoff = corrs[corrs.abs() > thresh].index
    return df[variables_above_cutoff]
