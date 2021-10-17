from functools import partial
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from numbers import Number
import numpy as np
import operator
import pandas as pd
import pandas_flavor as pf
from sklearn.model_selection import KFold

from htools.core import spacer
from htools import set_module_global


@pf.register_series_method
@pf.register_dataframe_method
def ends(df, n=2):
    """Display the first and last few rows of a dataframe.

    Parameters
    -----------
    n: int
        Number of rows to return in the head and tail, respectively. The total
        number of rows returned will be equal to 2*n.

    Returns
    --------
    pd.DataFrame
    """
    assert n > 0, 'n must be positive.'

    if df.shape[0] < 2 * n:
        return df
    return pd.concat((df.head(n), df.tail(n)), axis=0)


@pf.register_dataframe_method
def filter_by_count(df, col, method, value, norm=False):
    """Filter a dataframe to return a subset of rows determined by their
    value_counts(). For example, we can return rows with users who appear
    at least 5 times in the dataframe, or with users who appear less than 10
    times, or who appear exactly once.

    Parameters
    -----------
    col: str
        Name of dataframe column to filter by.
    method: str
        Symbol specifying which operation to use for filtering.
        One of ('=', '<', '>', '<=', '>=').
    value: int, float
        Numeric value that each row in `col` will be compared against.
    norm: bool
        If True, filtering will occur on normalized values (so the value should
        be a float between 0 and 1).

    Returns
    --------
    pd.DataFrame

    Examples
    ---------
    Return rows containing users who appear at least 5 times:
    df.filter_by_count('user_id', '>=', 5)

    Return rows containing users who appear only once:
    df.filter_by_count('user_id', '=', 1)

    Return rows containing users who make up less than 20% of rows:
    df.filter_by_count('user_id', '<', .2, True)
    """
    operation = {'=': operator.eq,
                 '>': operator.gt,
                 '<': operator.lt,
                 '>=': operator.ge,
                 '<=': operator.le
                }
    counts = df[col].value_counts(norm).loc[lambda x:
                                            operation[method](x, value)]
    return df[df[col].isin(counts.index)]


@pf.register_dataframe_method
def grouped_mode(df, xs, y):
    """Return the most common value in column y for each value or combination
    of values of xs. Note that this can be slow, especially when passing in
    multiple x columns.

    Parameters
    -----------
    xs: list[str]
        One or more column names to group by.
    y: str
        Column to calculate the modes from.

    Returns
    --------
    pd.Series
    """
    return df.dropna(subset=[y])\
             .groupby(xs)[y]\
             .agg(lambda x: pd.Series.mode(x)[0])


@pf.register_dataframe_method
def impute(df, col, fill_val=None, method='mean', inplace=False, dummy=True):
    """Fill null values in the specified column, then optionally add an
    additional column specifying whether the first column was originally null.
    This can be useful in certain machine learning problems if the fact that a
    value is missing may indicate something about the example.

    For instance, we might try to predict student test scores, where one
    feature column records the survey results of asking the student's parent to
    rate their satisfaction with the teacher on a scale of 1-5. If the value is
    missing, that means the parent didn't take the survey, and therefore may
    not be very involved with the student's academics. This could be highly
    relevant information that we don't want to discard, which we would if we
    simply imputed the missing value and made no record of it.

    Parameters
    -----------
    col: str
        Name of df column to fill null values for.
    fill_val: str, int, float, None
        If specified, this constant value will be used to impute missing
        values. If None, the `method` argument will be used to compute a fill
        value.
    method: str
        One of ('mean', 'median', 'mode'). This will only be used when fill_val
        is None. More complex methods, such as building a model to predict the
        missing values based on other features, must be done manually.
    inplace: bool
        Specify whether to perform the operation in place (default False).
    dummy: bool
        Specify whether to add a dummy column recording whether the value was
        initially null (default True).

    Returns
    --------
    pd.DataFrame
    """
    if not inplace:
        df = df.copy()

    # If adding a dummy column, it must be created before imputing null values.
    if dummy:
        df[col + '_isnull'] = df[col].isnull() * 1

    # Mode returns a series, mean and median return primitives.
    if fill_val is None:
        fill_val = getattr(df[col], method)()
        if method == 'mode':
            fill_val = fill_val[0]
    df[col].fillna(fill_val, inplace=True)

    if not inplace:
        return df


@pf.register_dataframe_method
def target_encode(df, x, y, n=5, stat='mean', shuffle=True, state=None,
                  inplace=False, df_val=None):
    """Compute target encoding based on one or more feature columns.

    Parameters
    -----------
    x: str, list[str]
        Name of columns to group by.
    y: str
        Name of target variable column.
    n: int
        Number of folds for regularized version. Must be >1.
    stat: str
        Specifies the type of aggregation to use on the target column.
        Typically this would be mean or occasionally median, but all the
        standard dataframe aggregation functions are available:
        ('mean', 'median', 'min', 'max', 'std', 'var', 'skew').
    shuffle: bool
        Specifies whether to shuffle the dataframe when creating folds of the
        data. This would be important, for instance, if the dataframe is
        ordered by a user_id, where each user has multiple rows. Here, a lack
        of shuffling means that all of a user's rows are likely to end up in
        the same fold. This effectively eliminates the value of creating the
        folds in the first place.
    state: None, int
        If state is an integer and shuffle is True, the folds produced by
        KFold will be repeatable. If state is None (the default) and shuffle
        is True, shuffling will be different every time.
    inplace: bool
        Specifies whether to do the operation in place. The inplace version
        does not return anything. When inplace==False, the dataframe is
        returned.
    df_val: None, pd.DataFrame
        Validation set (optional). If provided, naive (i.e. un-regularized)
        target encoding will be performed using the labels from the original
        (i.e. training) df. NOTE: Inplace must be True when passing in df_val,
        because we only return the original df.

    Returns
    --------
    pd.DataFrame or None
    """
    assert df_val is None or inplace, 'To encode df_val, inplace must be True.'
    # Prevents SettingWithCopy warning, which is not actually an issue here.
    pd.options.mode.chained_assignment = None

    if not inplace:
        df = df.copy()
    new_col = f"{'_'.join(x)}__{stat}_enc"
    global_agg = getattr(df[y], stat)()
    df[new_col] = global_agg

    def indexer(row):
        """Map a dataframe row to its grouped target value. When we group by
        multiple columns, our groupby object `enc` will require a tuple index.

        Note: When benchmarking function speed, it was slightly faster when
        leaving the if statement inside this function. Not sure if this is a
        coincidence but it at least seems like it's not hurting performance.
        """
        key = row[0] if len(x) == 1 else tuple(row)
        return enc.get(key, global_agg)

    # Compute target encoding on n-1 folds and map back to nth fold.
    for train_idx, val_idx in KFold(n, shuffle, state).split(df):
        enc = getattr(df.iloc[train_idx, :].groupby(x)[y], stat)()
        mapped = df.loc[:, x].iloc[val_idx].apply(indexer, axis=1)
        df.loc[:, new_col].iloc[val_idx] = mapped
    df[new_col].fillna(global_agg, inplace=True)

    # Encode validation set in place if it is passed in. No folds are used.
    if df_val is not None:
        enc = getattr(df.groupby(x)[y], stat)()
        df_val[new_col] = df_val[x].apply(indexer, axis=1).fillna(global_agg)

    if not inplace:
        return df


@pf.register_dataframe_method
def top_categories(df, col, n_categories=None, threshold=None):
    """Filter a dataframe to return rows containing the most common categories.
    This can be useful when a column has many possible values, some of which
    are extremely rare, and we want to consider only the ones that occur
    relatively frequently.

    The user can either specify the number of categories to include or set
    a threshold for the minimum number of occurrences. One of `categories` and
    `threshold` should be None, while the other should be an integer.

    Parameters
    -----------
    col: str
        Name of column to filter on.
    n_categories: int, None
        Optional - # of categories to include (i.e. top 5 most common
        categories).
    threshold: int, None
        Optional - Value count threshold to include (i.e. all categories that
        occur at least 10 times).

    Returns
    --------
    pd.DataFrame
    """
    assert bool(n_categories) + bool(threshold) == 1

    if n_categories is not None:
        top = df[col].value_counts(ascending=False).head(n_categories).index
        return df[df[col].isin(top)]
    if threshold is not None:
        return df.groupby(col).filter(lambda x: len(x) >= threshold)


@pf.register_series_method
def vcounts(df_col, **kwargs):
    """Return both the raw and normalized value_counts of a series.

    Parameters
    -----------
    Most parameters in value_counts() are available (i.e. `sort`, `ascending`,
    `dropna`), with the obvious exception of `normalize` since that is handled
    automatically.

    Returns
    --------
    pd.DataFrame

    Examples
    ---------
    df.colname.vcounts()
    """
    if 'normalize' in kwargs.keys():
        del kwargs['normalize']

    counts = df_col.value_counts(**kwargs)
    normed_counts = df_col.value_counts(normalize=True, **kwargs)

    # Pandas seems to have problem merging on bool col/index. Could use
    # pd.concat but unsure if order is consistent in case of ties.
    if counts.name is None:
        counts.name = 'raw'
        normed_counts.name = 'normed'

    df = pd.merge(counts, normed_counts,
                  how='left', left_index=True, right_index=True,
                  suffixes=['_raw_count', '_normed_count'])\
           .reset_index()

    col_name = '_'.join(df.columns[1].split('_')[:-2])
    return df.rename({'index': col_name}, axis=1)


@pf.register_series_method
@pf.register_dataframe_method
def pprint(df, truncate_text=True):
    """Display a dataframe of series as a rendered HTML table in
    Jupyter notebooks. Useful when printing multiple outputs in a cell.

    Parameters
    ----------
    truncate_text: bool
        If True, long strings will be truncated to maintain a reasonable
        row height. If False, the whole string will be displayed, which can
        results in massive outputs.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)
    if truncate_text:
        display(df)
    else:
        display(HTML(df.to_html()))


@pf.register_series_method
@pf.register_dataframe_method
def lambda_sort(df, func, **kwargs):
    """Sort a DataFrame or Series by a function that takes itself as input.
    For example, we can sort by the absolute value of a column or the sum of
    2 different columns.
    Parameters
    -----------
    func: function
       Callable function or lambda expression to sort by.
       (eg: lambda x: abs(x))
    **kwargs: additional keyword args will be passed to the sort_values()
       method.

    Returns
    --------
    pd.DataFrame

    Examples
    ---------
    >>> df = pd.DataFrame(np.arange(8).reshape((4, 2)), columns=['a', 'b'])
    >>> df.loc[3, 'a'] *= -1
    >>> df

        a  b
    0    0  1
    1    2  3
    2    4  6
    3   -6  7

    >>> df.lambda_sort(lambda x: x.a * x.b)

        a  b
    3   -6  7
    2    0  1
    1    2  3
    0    4  5
    """
    col = 'lambda_col'
    df = df.copy()
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)
    df[col] = func(df)
    return df.sort_values(col, **kwargs).drop(col, axis=1)


@pf.register_dataframe_method
def coalesce(df, cols):
    """Create a column where each row contains the first non-null value for
    that row from a list of columns.

    Parameters
    ----------
    cols: list[str]
        Names of columns to coalesce over.

    Returns
    -------
    pd.Series
    """
    new_col = df[cols[0]].copy()
    i = 1
    while new_col.isnull().sum() > 0 and i < len(cols):
        next_col = cols[i]
        new_col.fillna(df[next_col], inplace=True)
        i += 1
    return new_col


@pf.register_series_method
def stringify(list_col, join=True, ignore_terms=None, greedy_ignore=False,
              null=''):
    """Converts a df column of lists, possibly containing np.nan's, to strings.

    Parameters
    -----------
    join: bool
        If True, create the string by joining all items in each list/row. If
        False, simply return the first item in each list. Default True.
    ignore_terms: list, set
        Terms to drop from the column. If None, all terms will be retained.
        Ex: {'width=device-width'}
    greedy_ignore: bool
        If True, ignore_terms can be a list of prefixes. In other words,
        we will remove any strings in the list column that start with one of
        the ignore_terms even (as opposed to requiring an exact match.)
    null: str
        The value to replace null values with. For many pandas string methods,
        it is beneficial for this to be a string rather than np.nan.

    Returns
    -------
    pd.Series
    """
    ignore_terms = dict.fromkeys(ignore_terms or [])

    def process(x, join, ignore_terms, greedy_ignore, null):
        # Handles both np.nan and empty lists.
        if not isinstance(x, list) or not x:
            return null

        # Dict instead of set to maintain order
        # (dict-key operations would still change order).
        x = dict.fromkeys(map(str, x))
        if greedy_ignore:
            x = (term for term in x if not term.startswith(tuple(ignore_terms)))
        else:
            x = (term for term in x if term not in ignore_terms)

        # Return a string.
        if join:
            return ' '.join(x)
        else:
            return next(x)

    return list_col.map(partial(process, join=join, ignore_terms=ignore_terms,
                                greedy_ignore=greedy_ignore, null=null))


@pf.register_series_method
def is_list_col(col):
    """Determine whether a column is a list column. These are columns
    resulting from the protobuf format, where we end up with a situation
    where rows either contains lists or np.nan.

    Parameters
    -----------
    col: pd.Series
        The column to evaluate.

    Returns
    --------
    bool
    """
    # Filter out nulls first otherwise type could be np.nan instead of list.
    no_nulls = col.dropna()
    return not no_nulls.empty and isinstance(no_nulls.iloc[0], list)


@pf.register_dataframe_method
@pf.register_series_method
def verbose_plot(df, nrows=None, **kwargs):
    """Plot data and also print it out as a table. For example, this is
    nice when finding quantiles, where it's often helpful to plot them for a
    quick visual snapshot but also to examine the table of values for more
    details.

    Parameters
    ----------
    nrows: int or None
        If provided, will truncate the printed table. Otherwise all rows will
        be shown.
    kwargs: any
        Arguments to pass to the plot method.

    Returns
    -------
    None

    Examples
    --------
    df.bids.quantile(np.arange(0, 1, .1)).verbose_plot(kind='bar', color='red')
    """
    df.plot(**kwargs)
    df.head(nrows).pprint()
    plt.show()


@pf.register_dataframe_method
def fuzzy_groupby(df, col, model=None, cats=5, fuzzy_col=True,
                  globalize_vecs='_vecs', **kwargs):
    """Pandas method to group by a string column using fuzzy matching using
    SentenceTransformers. Categories can be passed in or selected automatically
    as the most common values in the data.

    Note: this method will try to import things from the sentence_transformers
    package. I chose not to include that in the htools dependencies since this
    is sort of niche/experimental functionality and I don't want to require
    transformers for all installs.

    Parameters
    ----------
    col: str
        Column name to group by.
    model: SentenceTransformer or None
        By default, we use the 'all-MiniLM-L6-v2' model. This maps each piece
        of text to a 384 dimensional vector. In theory, this should work with
        texts of varying lengths (sentences, paragraphs, etc.) and allow us to
        compare them without issue, though YMMV.
    cats: Iterable[str] or int
        Each item in col will be mapped to one of these categories based on the
        cosine similarity between it and each category. If an integer is
        provided, this defaults to the n most common items in col.
        Depending on your data, there's no guarantee these will be particularly
        good choices. Clustering algorithms might be a better starting
        place but I still have to think about how that would work (with some
        of sentence_transformers' clustering methods, not everything is
        assigned to a cluster).
    fuzzy_col: bool or str
        If Truthy, we add a new column to df containing the cat each item was
        mapped to. A string argument will be used as the new col name while a
        value of True will default to f'{col}_fuzzy'.
    globalize_vecs: str
        If True, store resulting vectors (technically a rank 2 tensor) in a
        global variable with this name. We do this rather than returning them
        because a common workflow is something like:
        df.fuzzy_groupby('color').age.mean()
        Having fuzzy_groupby return a tuple would therefore be somewhat
        awkward.
    kwargs: any
        Additional kwargs to pass to model.encode() when creating vectors.

    Returns
    -------
    pd.DataFrameGroupBy
    """
    from sentence_transformers.util import cos_sim

    if not model:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    if isinstance(cats, Number):
        cats = df[col].value_counts().head(cats).index.tolist()
    vals = df[col].unique()
    cat_vecs = model.encode(cats, **kwargs)
    val_vecs = model.encode(vals, **kwargs)
    sim_mat = cos_sim(val_vecs, cat_vecs)
    w2cat = {val: cats[i] for val, i in zip(vals, sim_mat.argmax(1))}
    assigned_cats = df[col].map(w2cat)
    if fuzzy_col:
        title = (fuzzy_col if isinstance(fuzzy_col, str) else f'{col}_fuzzy')
        df[title] = assigned_cats
    if globalize_vecs:
        set_module_global('__main__', globalize_vecs, val_vecs)
    return df.groupby(assigned_cats)


def anti_join(df_left, df_right, left_on=None, right_on=None, **kwargs):
    """Remove rows present in rhs from lhs.

    Parameters
    ----------
    df_left: pd.DataFrame
        The dataframe containing the initial rows to subtract from.
    df_right: pd.DataFrame
        The dataframe containing rows to "subtract" from `df_left`.
    left_on: str or list[str]
        Column name(s) to join on.
    right_on: str or list[str]
        If not provided, the default is to use the same values as `left_on`.
    kwargs: any
        Will be passed to the merge operation. Ex: `left_index=True`.

    Returns
    -------
    pd.DataFrame: Contains all rows that are present in `df_left` but not in
        `df_right`.
    """
    right = df_right.copy()
    right['XX_RHS_XX'] = True
    merged = df_left.merge(right, how='left', left_on=left_on,
                           right_on=right_on or left_on,
                           suffixes=['', '_rhs'], **kwargs)
    return merged.loc[merged.XX_RHS_XX.isnull(), df_left.columns]


def highlight_rows(row, fn, highlight_color='yellow', default_color='white'):
    """Use with pd.style.apply to highlight certain rows. `row` will be passed
    automatically through the pandas method.

    Parameters
    ----------
    row: row of a pandas DataFrame
        Never passed in manually by user.
    fn: function
        Takes row as input and returns True if it should be highlighted, False
        otherwise.
    highlight_color: str
        Color to display highlighted rows in.
    default_color: str
        Color to display un-highlighted rows in.

    Examples
    --------
    highlight_shared = partial(highlight_rows, fn=lambda x: x.cat in names)
    df.style.apply(highlight_shared, axis=1)
    """
    color = highlight_color if fn(row) else default_color
    return [f'background-color: {color}'] * len(row.values)


def paginate(d, seen=()):
    """Page through a dict of dataframes, showing for each column:
    name, # unique, % null, standard deviation, min, max, mean, 1 non-null row

    Parameters
    ----------
    d: dict[pd.DataFrame]
    seen: Iterable
        If provided, this should be a list-like container of strings (or
        whatever type you use as keys in `d`). These are the names of the df's
        you've already explored. A common workflow is to have a line like
        `seen = paginate(name2df, seen)`. This way if you choose to exit the
        loop for some additional EDA, you can easily pick up where you left
        off.
    """
    seen = set(seen)
    for i, (k, v) in enumerate(d.items(), 1):
        if k in seen: continue
        print(spacer())
        print(f'{i}. {k}')
        print(v.shape)
        nulls = pd.DataFrame(v.isnull().mean().T)
        # Get 1 non-null example for each column.
        examples = pd.concat([v[col].dropna().head(1).reset_index(drop=True)
                              for col in v], axis=1).T
        stats = v.select_dtypes([np.number, bool]) \
            .agg([np.mean, np.std, min, max]).T
        pd.DataFrame(v.nunique().sort_index()) \
            .merge(nulls, left_index=True, right_index=True) \
            .merge(stats, how='left', left_index=True, right_index=True) \
            .merge(examples, left_index=True, right_index=True) \
            .rename(columns={'0_x': 'n_unique', '0_y': 'pct_null',
                             0: 'example'}) \
            .fillna('') \
            .pprint(False)
        while True:
            cmd = input('Press <ENTER> to continue or press e to exit.')
            if cmd == '':
                seen.add(k)
                break
            elif cmd == 'e':
                return seen
    return seen


