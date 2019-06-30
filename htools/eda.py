import pandas as pd


def ends(df, n=5):
    """Show both the head and tail of a pandas dataframe.

    Sample usage:
    pd.DataFrame.ends = ends
    df.ends(3)
    df.sort_values('x').ends()

    :param df: pandas DataFrame
    :param n: Number of rows to display in head and tail, respectively. The
    total number of rows returned will therefore be 2*n.
    :return: pandas DataFrame
    """
    return pd.concat([df.head(n), df.tail(n)], axis=0)
