import operator
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


def filter_by_count(df, col, value, method):
    """Filter a dataframe to return a subset of rows determined by their
     value_counts(). For example, we can return rows with users who appear
     at least 5 times in the dataframe, or with users who appear less than 10
     times, or who appear greater than or equal to 100 times.
     """
    operation = {'=': operator.eq,
                 '>': operator.gt,
                 '<': operator.lt,
                 '>=': operator.ge,
                 '<=': operator.le
                }
    counts = df[col].value_counts().loc[lambda x: operation[method](x, value)]
    return df[df[col].isin(counts.index)]


def top_cats(df, col, cats=None, val=None):
    """Filter a df to only include the most common categories.

    Parameters
    -----------
    df: pd.DataFrame
    col: str
        Name of column to filter on.
    cats: int
        Optional - # of categories to include (i.e. top 5 most common
        categories).
    val: int
        Optional - Value count threshold to include (i.e. all categories that
        occur at least 10 times).
    """
    if cats is not None:
        top = df[col].value_counts(ascending=False).head(cats).index
        return df[df[col].isin(top)]
    if val is not None:
        return df.groupby(col).filter(lambda x: len(x) >= val)
