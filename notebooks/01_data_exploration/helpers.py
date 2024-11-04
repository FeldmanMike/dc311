"""
Helper functions for notebook `01_data_exploration.ipynb`
"""

import pandas as pd


def check_if_two_fields_are_one_to_one(
    df: pd.DataFrame, field_one: str, field_two: str
):
    """
    Check if the relationship between two fields in a dataframe is one-to-one.

    Args:
        df: A pandas DataFrame
        field_one: A field in the dataframe
        field_two: Another field in the dataframe

    Returns:
        None
    """
    check_if_two_fields_are_one_to_one_helper(df, field_one, field_two)
    print("----------------------------------------------------")
    check_if_two_fields_are_one_to_one_helper(df, field_two, field_one)
    return None


def check_if_two_fields_are_one_to_one_helper(
    df: pd.DataFrame, field_one: str, field_two: str
):
    """
    Check if the relationship between two fields in a dataframe is one-to-one.

    Args:
        df: A pandas DataFrame
        field_one: A field in the dataframe
        field_two: Another field in the dataframe

    Returns:
        None
    """
    field_one_gb = df.groupby(field_one)[field_two].nunique()
    df_filt = field_one_gb[field_one_gb > 1]
    if df_filt.empty:
        print(f"All values of `{field_one}` have at most one `{field_two}`.")
    else:
        print(
            f"There are {len(df_filt)} values of `{field_one}` that have "
            f"more than one `{field_two}`:"
        )
        print(df_filt)


def summarize_relationship_between_target_and_variable(
    df: pd.DataFrame, ind_var: str, descriptor_var: str = None
) -> pd.DataFrame:
    """
    Calculate metrics to summarize relationship between the number of days needed to
    resolve a DC 311 case, and another categorical independent variable.

    Args:
        df: A pandas DataFrame
        ind_var: Name of categorical independent variable
        descriptor_var: Name of another independent variable to append to final
            dataframe that is associated with ind_var (e.g., a service code
            description variable if ind_var is service code)

    Returns:
        A DataFrame where metrics have been calculated for each value that the
        independent variable takes on
    """
    agg_df = (
        df.groupby(ind_var)
        .agg(
            median_days_to_resolve=("days_to_resolve", "median"),
            std_dev_days_to_resolve=("days_to_resolve", "std"),
            avg_days_to_resolve=("days_to_resolve", "mean"),
            count_records=("objectid", "count"),
        )
        .reset_index()
    )

    if descriptor_var:
        agg_df = agg_df.merge(
            df[[ind_var, descriptor_var]], on=ind_var, how="left"
        ).drop_duplicates()

    return agg_df
