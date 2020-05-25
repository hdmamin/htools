from bz2 import BZ2File
from collections import Counter, Sequence, Iterable, \
    Mapping
import gc
from email.mime.text import MIMEText
import inspect
from itertools import chain
import json
import os
from pathlib import Path
import pickle
from random import choice
import re
import smtplib
import sys

from htools.config import get_credentials, get_default_user


class InvalidArgumentError(Exception):
    pass


def hdir(obj, magics=False, internals=False):
    """Print object methods and attributes, by default excluding magic methods.

    Parameters
    -----------
    obj: any type
        The object to print methods and attributes for.
    magics: bool
        Specifies whether to include magic methods (e.g. __name__, __hash__).
        Default False.
    internals: bool
        Specifies whether to include internal methods (e.g. _dfs, _name).
        Default False.

    Returns
    --------
    dict
        Keys are method/attribute names, values are strings specifying whether
        the corresponding key is a 'method' or an 'attr'.
    """
    output = dict()
    for attr in dir(obj):
        # Exclude magics or internals if specified.
        if (not magics and attr.startswith('__')) or \
           (not internals and re.match('_[^_]', attr)):
            continue

        # Handle rare case where attr can't be invoked (e.g. df.sparse on a
        # non-sparse Pandas dataframe).
        try:
            is_method = callable(getattr(obj, attr))
        except Exception:
            continue

        # Update output to specify whether attr is callable.
        if is_method:
            output[attr] = 'method'
        else:
            output[attr] = 'attribute'
    return output


def tdir(obj, **kwargs):
    """A variation of the built in `dir` function that shows the
    attribute names as well as their types. Methods are excluded as they can
    change the object's state.

    Parameters
    ----------
    obj: any type
        The object to examine.
    kwargs: bool
        Additional arguments to be passed to hdir. Options are `magics` and
        `internals`. See hdir documentation for more information.

    Returns
    -------
    dict[str, type]: Dictionary mapping the name of the object's attributes to
    the corresponding types of those attributes.
    """
    return {k: type(getattr(obj, k))
            for k, v in hdir(obj, **kwargs).items() if v == 'attribute'}


def hasarg(func, arg):
    """Checks if a function has a given argument. Works with args and kwargs as
    well if you exclude the stars. See example below.

    Parameters
    ----------
    func: function
    arg: str
        Name of argument to look for.

    Returns
    -------
    bool

    Example
    -------
    def foo(a, b=6, *args):
        return

    >>> hasarg(foo, 'b')
    True
    >>> hasarg(foo, 'args')
    True
    >>> hasarg(foo, 'c')
    False
    """
    return arg in inspect.signature(func).parameters


def quickmail(subject, message, to_email, from_email=None):
    """Send an email.

    Parameters
    -----------
    from_email: str
        Gmail address being used to send email.
    to_email: str
        Recipient's email.
    subject: str
        Subject line of email.
    message: str
        Body of email.

    Returns
    --------
    None
    """
    # Load source email address.
    from_email = from_email or get_default_user()
    if not from_email:
        return None

    # Load email password.
    password = get_credentials(from_email)
    if not password:
        return None

    # Create message instance.
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Access server and send email.
    server = smtplib.SMTP(host='smtp.gmail.com', port=587)
    server.starttls()
    server.login(user=from_email, password=password)
    server.send_message(msg)
    print(f'Email sent to {to_email}.')


def hsplit(text, sep, group=True, attach=True):
    """Flexible string splitting that retains the delimiter rather, unlike
    the built-in str.split() method.

    Parameters
    -----------
    text: str
        The input text to be split.
    sep: str
        The delimiter to be split on.
    group: bool
        Specifies whether to group consecutive delimiters together (True),
        or to separate them (False).
    attach: bool
        Specifies whether to attach the delimiter to the string that preceeds
        it (True), or to detach it so it appears in the output list as its own
        item (False).

    Returns
    --------
    list[str]

    Examples
    ---------
    text = "Score -- Giants win 6-5"
    sep = '-'

    # Case 0.1: Delimiters are grouped together and attached to the preceding
    word.
    >> hsplit(text, sep, group=True, attach=True)
    >> ['Score --', ' Giants win 6-', '5']

    # Case 0.2: Delimiters are grouped together but are detached from the
    preceding word, instead appearing as their own item in the output list.
    >> hsplit(text, sep, group=True, attach=False)
    >> ['Score ', '--', ' Giants win 6', '-', '5']

    Case 1.1: Delimiters are retained and attached to the preceding string.
    If the delimiter occurs multiple times consecutively, only the first
    occurrence is attached, and the rest appear as individual items in the
    output list.
    >> hsplit(text, sep, group=False, attach=True)
    >> ['Score -', '-', ' Giants win 6-', '5']

    # Case 1.2: Delimiters are retained but are detached from the preceding
    string. Each instance appears as its own item in the output list.
    >> hsplit(text, sep, group=False, attach=False)
    >> ['Score ', '-', '-', ' Giants win 6', '-', '5']
    """
    sep_re = re.escape(sep)
    regex = f'[^{sep_re}]*{sep_re}*'

    ##########################################################################
    # Case 0: Consecutive delimiters are grouped together.
    ##########################################################################
    if group:
        # Subcase 0.1
        if attach:
            return [word for word in re.findall(regex, text)][:-1]

        # Subcase 0.2
        else:
            return [word for word in re.split(f'({sep_re}+)', text) if word]

    ##########################################################################
    # Case 1: Consecutive delimiters are NOT grouped together.
    ##########################################################################
    words = text.split(sep)

    # Subcase 1.1
    if attach:
        return [word for word in re.findall(regex[:-1]+'?', text) if word]

    # Subcase 1.2
    return [word for word in chain(*zip(words, [sep]*len(words))) if word][:-1]


def rmvars(*args):
    """Wrapper to quickly free up memory by deleting global variables. Htools
    3.0 does not provide a way to do this for local variables.

    Parameters
    ----------
    args: str
        One or more variable names to delete. Do not pass in the variable
        itself.

    Returns
    -------
    None
    """
    for arg in args:
        del globals()[arg]
    gc.collect()


def print_object_sizes(space, limit=None, exclude_underscore=True):
    """Print the object names and sizes of the currently defined objects.

    Parameters
    -----------
    space: dict
        locals(), globals(), or vars()
    limit: int or None
        Optionally limit the number of objects displayed (default None for no
        limit).
    exclude_underscore: bool
        Determine whether to exclude objects whose names start with an
        underscore (default True).
    """
    var_size = [(var, sys.getsizeof(obj)) for var, obj in space.items()]
    for var, size in sorted(var_size, key=lambda x: -x[1])[:limit]:
        if not var.startswith('_') or not exclude_underscore:
            print(var, size)


def eprint(arr, indent=2, spacing=1):
    """Enumerated print. Prints an iterable with one item per line accompanied
    by a number specifying its index in the iterable.

    Parameters
    -----------
    arr: iterable
        The object to be iterated over.
    indent: int
        Width to assign to column of integer indices. Default is 2, meaning
        columns will line up as long as <100 items are being printed, which is
        the expected use case.
    spacing: int
        Line spacing. Default of 1 will print each item on a new line with no
        blank lines in between. Spacing of 2 will double space output, and so
        on for larger values.

    Returns
    --------
    None
    """
    for i, x in enumerate(arr):
        print(f'{i:>{indent}}: {x}', end='\n'*spacing)


def _read_write_args(path, mode):
    """Helper for `save` and `load` functions.

    Parameters
    ----------
    path: str
        Path to read/write object from/to.
    mode: str
        'w' for writing files (as in `save`), 'r' for reading files
        (as in `load`).

    Returns
    -------
    tuple: Function to open file, mode to open file with (str), object to open
        file with.
    """
    ext = path.rpartition('.')[-1]
    if ext not in {'json', 'pkl', 'txt', 'zip'}:
        raise InvalidArgumentError(
            'Invalid extension. Make sure your filename ends with '
            '.json, .pkl, or .zip.'
        )

    # Store in dict to make it easier to add additional formats in future.
    ext2data = {
        'json': (open, '', json),
        'pkl': (open, 'b', pickle),
        'zip': (BZ2File, '', pickle),
    }
    opener, mode_suffix, saver = ext2data[ext]
    return opener, mode + mode_suffix, saver


def save(obj, path, mode_pre='w', verbose=True):
    """Wrapper to save data as text, pickle (optionally zipped), or json.

    Parameters
    -----------
    obj: any
        Object to save. This will be pickled/jsonified/zipped inside the
        function - do not convert it before-hand.
    path: str
        File name to save object to. Should end with .txt, .pkl, .zip, or
        .json depending on desired output format. If .zip is used, object will
        be zipped and then pickled.
    mode_pre: str
        Determines whether to write or append text. One of ('w', 'a').
    verbose: bool
        If True, print a message confirming that the data was pickled, along
        with its path.

    Returns
    -------
    None
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    if verbose: print(f'Writing data to {path}.')
    if path.suffix == '.txt':
        with path.open(mode_pre) as f:
            f.write(obj)
    else:
        opener, mode, saver = _read_write_args(str(path), mode_pre)
        with opener(path, mode) as f:
            saver.dump(obj, f)


def load(path, verbose=True):
    """Wrapper to load text files or pickled (optionally zipped) or json data.

    Parameters
    ----------
    path : str
        File to load. File type will be inferred from extension. Must be one of
        '.txt', '.json', '.pkl', or '.zip'.
    verbose : bool, optional
        If True, will print message stating where object was loaded from.

    Returns
    -------
    object: The Python object that was pickled to the specified file.
    """
    path = Path(path)
    if path.suffix == '.txt':
        return path.read_text()

    opener, mode, saver = _read_write_args(str(path), 'r')
    with opener(path, mode) as f:
        data = saver.load(f)
    if verbose: print(f'Object loaded from {path}.')
    return data


def dict_sum(*args):
    """Given two or more dictionaries with numeric values, combine them into a
    single dictionary. For keys that appear in multiple dictionaries, their
    corresponding values are added to produce the new value.

    This differs from combining two dictionaries in the following manner:

    {**d1, **d2}

    The method shown above will combine the keys but will retain the value
    from d2, rather than adding the values from d1 and d2.

    Parameters
    -----------
    *args: dicts
        2 or more dictionaries with numeric values.

    Returns
    --------
    dict: Contains all keys which appear in any of the dictionaries that are
          passed in. The corresponding values from each dictionary containing a
          given key are summed to produce the new value.

    Examples
    ---------
    >>> d1 = {'a': 1, 'b': 2, 'c': 3}
    >>> d2 = {'a': 10, 'c': -20, 'd': 30}
    >>> d3 = {'c': 10, 'd': 5, 'e': 0}
    >>> dict_sum(d1, d2)

    {'a': 11, 'b': 2, 'c': -7, 'd': 35, 'e': 0}
    """
    keys = {key for d in args for key in d.keys()}
    return {key: sum(d.get(key, 0) for d in args)
            for key in keys}


def _select_mapping(items, keep=(), drop=()):
    """Helper function for `select`.

    Parameters
    ----------
    items: Mapping
        Dict (or similar mapping) to select/drop from.
    keep: Iterable[str]
        Sequence of keys to keep.
    drop: Iterable[str]
        Sequence of keys to drop. You should specify either `keep` or `drop`,
        not both.

    Returns
    -------
    Dict
    """
    if keep:
        return {k: items[k] for k in keep}
    return {k: v for k, v in items.items() if k not in set(drop)}


def _select_sequence(items, keep=(), drop=()):
    """Helper function for `select` that works on sequences (basically
    collections that support enumeration).

    Parameters
    ----------
    items: Sequence
        List, tuple, or iterable sequence of some sort to select items from.
    keep: Iterable[str]
        Sequence of indices to keep.
    drop: Iterable[str]
        Sequence of indices to drop. You should specify either `keep` or
        `drop`, not both.

    Returns
    -------
    Same type as `items` (usually a list or tuple).
    """
    type_ = type(items)
    if keep:
        return type_(x for i, x in enumerate(items) if i in set(keep))
    return type_(x for i, x in enumerate(items) if i not in set(drop))


def select(items, keep=(), drop=()):
    """Select a subset of a data structure. When used on a mapping (e.g. dict),
    you can specify a list of keys to include or exclude. When used on a
    sequence like a list or tuple, specify indices instead of keys.

    Parameters
    ----------
    items: abc.Sequence or abc.Mapping
        The dictionary to select items from.
    keep: Iterable[str]
        Sequence of keys to keep.
    drop: Iterable[str]
        Sequence of keys to drop. You should specify either `keep` or `drop`,
        not both.

    Returns
    -------
    dict: Dictionary containing only the specified keys (when passing in
        `keep`), or all keys except the specified ones (when passing in
        `drop`).
    """
    if bool(keep) + bool(drop) != 1:
        raise InvalidArgumentError('Specify exactly one of `keep` or `drop`.')

    if isinstance(items, Mapping):
        return _select_mapping(items, keep, drop)
    elif isinstance(items, Sequence):
        return _select_sequence(items, keep, drop)
    else:
        raise InvalidArgumentError('`items` must be a Mapping or Sequence.')


def differences(obj1, obj2, methods=False, **kwargs):
    """Find the differences between two objects (generally of the same type -
    technically this isn't enforced but we do require that the objects have
    the same set of attribute names so a similar effect is achieved. Actual
    type checking was causing problems comparing multiple Args instances,
    presumably because each Args object is defined when called).

    This is a way to get more detail beyond whether two objects are equal or
    not.

    Parameters
    -----------
    obj1: any
        An object.
    obj2: any, usually the same type as obj1
        An object.
    methods: bool
        If True, include methods in the comparison. If False, only attributes
        will be compared. Note that the output may not be particularly
        interpretable when using method=True; for instance when comparing two
        strings consisting of different characters, we get a lot of output
        that looks like this:

        {'islower': (<function str.islower()>, <function str.islower()>),
        'isupper': (<function str.isupper()>, <function str.isupper()>),...
        'istitle': (<function str.istitle()>, <function str.istitle()>)}

        These attributes all reflect the same difference: if obj1 is 'abc'
        and obj2 is 'def', then
        'abc' != 'def' and
        'ABC' != 'DEF' abd
        'Abc' != 'Def'.

        When method=False, we ignore all of these, such that
        differences('a', 'b') returns {}. Therefore, it is important to
        carefully consider what differences you care about identifying.

    **kwargs: bool
        Can pass args to hdir to include magics or internals.

    Returns
    --------
    dict[str, tuple]: Maps attribute name to a tuple of values, where the
        first is the corresponding value for obj1 and the second is the
        corresponding value for obj2.
    """
    # May built-in comparison functionality. Keep error handling broad.
    try:
        if obj1 == obj2:
            return {}
    except Exception:
        pass

    attr1, attr2 = hdir(obj1, **kwargs), hdir(obj2, **kwargs)
    assert attr1.keys() == attr2.keys(), 'Objects must have same attributes.'

    diffs = {}
    for (k1, v1), (k2, v2) in zip(attr1.items(), attr2.items()):
        # Only compare non-callable attributes.
        if not (methods or v1 == 'attribute'):
            continue

        # Comparisons work differently for arrays/tensors than other objects.
        val1, val2 = getattr(obj1, k1), getattr(obj2, k2)
        try:
            equal = (val1 == val2).all()
        except AttributeError:
            equal = val1 == val2

        # Store values that are different for obj1 and obj2.
        if not equal:
            diffs[k1] = (val1, val2)

    return diffs


def catch(func, *args, verbose=False):
    """Error handling for list comprehensions. In practice, it's recommended
    to use the higher-level robust_comp() function which uses catch() under the
    hood.

    Parameters
    -----------
    func: function
    *args: any type
        Arguments to be passed to func.
    verbose: bool
        If True, print the error message should one occur.

    Returns
    --------
    any type: If the function executes successfully, its output is returned.
        Otherwise, return None.

    Examples
    ---------
    [catch(lambda x: 1 / x, i) for i in range(3)]
    >>> [None, 1.0, 0.5]

    # Note that the filtering method shown below also removes zeros which is
    # okay in this case.
    list(filter(None, [catch(lambda x: 1 / x, i) for i in range(3)]))
    >>> [1.0, 0.5]
    """
    try:
        return func(*args)
    except Exception as e:
        if verbose: print(e)
        return


def safe_map(func, seq):
    """This addresses the issue of error handling in map() or list
    comprehension operations by simply skipping any items that throw an error.
    Note that values of None will be removed from the resulting list.

    Parameters
    ----------
    func: function
        Function to apply to each item in seq.
    seq: generator, iterator
        The sequence to iterate over. This could also be a generator, list,
        set, etc.

    Returns
    -------
    list

    Examples
    --------
    # Notice that instead of throwing an error when dividing by zero, that
    # entry was simply dropped.
    >>> safe_map(lambda x: x/(x-2), range(4))
    [-0.0, -1.0, 3.0]
    """
    return list(
        filter(lambda x: x is not None, (catch(func, obj) for obj in seq))
    )


def flatten(nested):
    """Flatten a nested sequence where the sub-items can be sequences or
    primitives. This differs slightly from itertools chain methods because
    those require all sub-items to be sequences. Here, items can be primitives,
    sequences, nested sequences, or any combination of these. Any iterable
    items aside from strings will be completely un-nested, so use with caution
    (e.g. a torch Dataset would be unpacked into separate items for each
    index). This also returns a list rather than a generator.

    Parameters
    ----------
    nested: sequence (list, tuple, set)
        Sequence where some or all of the items are also sequences.

    Returns
    -------
    list: Flattened version of `nested`.
    """
    def _walk(nested):
        for group in nested:
            if isinstance(group, Iterable) and not isinstance(group, str):
                yield from _walk(group)
            else:
                yield group
    return list(_walk(nested))


class BasicPipeline:
    """Create a simple unidirectional pipeline of functions to apply in order
    with optional debugging output.
    """

    def  __init__(self, *funcs):
        """
        Parameters
        ----------
        *funcs: function(s)
            One or more functions to apply in the specified order.
        """
        self.funcs = funcs

    def __call__(self, x, verbose=False, attr=''):
        """Apply the pipeline of functions to x.

        Parameters
        ----------
        x: any
            Object to operate on.
        verbose: bool
            If True, print x (or an attribute of x) after each step.
        attr: str
            If specified and verbose is True, will print this attribute of x
            after each function is applied.

        Returns
        -------
        output of last func in self.funcs
        """
        for func in self.funcs:
            x = func(x)
            if verbose: print(repr(getattr(x, attr, x)))
        return x

    def __repr__(self):
        return f'BasicPipeline({", ".join(f.__name__ for f in self.funcs)})'


def pipe(x, *funcs, verbose=False, attr=''):
    """Convenience function to apply many functions in order to some object.
    This lets us replace messy notation where it's hard to keep parenthesis
    straight:

    list(parse_processed_text(tokenize_rows(porter_stem(strip_html_tags(
         text)))))

    with:

    pipe(text, strip_html_tags, porter_stem, tokenize_rows,
         parse_processed_text, list)

    or if we have a list  of functions:

    pipe(x, *funcs)

    Parameters
    ----------
    x: any
        Object to apply functions to.
    *funcs: function(s)
        Functions in the order you want to apply them. Use functools.partial
        to specify other arguments.
    verbose: bool
        If True, print x (or an attribute of x) after each step.
    attr: str
        If specified and verbose is True, will print this attribute of x
        after each function is applied.

    Returns
    -------
    output of last func in *funcs
    """
    return BasicPipeline(*funcs)(x, verbose=verbose, attr=attr)


def vcounts(arr, normalize=True):
    """Equivalent of pandas_htools vcounts method that we can apply on lists
    or arrays. Basically just a wrapper around Counter but with optional
    normalization.

    Parameters
    ----------
    arr: Iterable
        Sequence of values to count. Typically a list or numpy array.
    normalize: bool
        If True, counts will be converted to percentages.

    Returns
    -------
    dict: Maps unique items in `arr` to the number of times (or % of times)
        that they occur in `arr`.
    """
    counts = dict(Counter(arr))
    if normalize:
        length = len(arr)
        counts = {k: v/length for k, v in counts.items()}
    return counts


def item(it, random=True, try_values=True):
    """Get an item from an iterable (e.g. dict, set, torch DataLoader).
    This is a quick way to access an item for iterables that don't support
    indexing, or do support indexing but require us to know a key.

    Parameters
    ----------
    it: Iterable
        Container that we want to access a value from.
    random: bool
        If True, pick a random value from `it`. Otherwise just return the first
        value.
    try_values: bool
        If True, will check if `it` has a `values` attribute and will operate
        on that if it does. We often want to see a random value from a dict
        rather than a key. If we want both a key and value, we could set
        try_values=False and pass in d.items().

    Returns
    -------
    any: An item from the iterable.
    """
    if try_values and hasattr(it, 'values'): it = it.values()
    if random:
        return choice(list(it))
    return next(iter(it))


def lmap(fn, *args):
    """Basically a wrapper for `map` that returns a list rather than a
    generator. This is such a common pattern that I think it deserves its own
    function (think of it as a concise alternative to a list comprehension).
    One slight difference is that we use *args instead of passing in an
    iterable. This adds a slight convenience for the intended use case (fast
    prototyping). See the `Examples` for more on this.

    Parameters
    ----------
    args: any

    Returns
    -------
    list

    Examples
    --------
    Consider these three equivalent syntax options:

    lmap(fn, x, y)
    [fn(obj) for obj in (x, y)]
    list(map(fn, (x, y))

    When quickly iterating, option 1 saves a bit of typing. The extra
    parentheses that options 2 and 3 require to put x and y in a temporary
    data structure can get messy as we add more complex logic.
    """
    return list(map(fn, args))


def attrmap(attr, *args):
    """More convenient syntax for quick data exploration. Get an attribute
    value for multiple objects.

    Parameters
    ----------
    attr: str
        Name of attribute to retrieve for each object.
    args: any
        Objects (usually of same type) to retrieve attributes for.

    Returns
    -------
    list: Result for each object.

    Examples
    --------
    df1 = pd.DataFrame(np.random.randint(0, 10, (4, 5)))
    df2 = pd.DataFrame(np.random.randint(0, 3, (4, 5)))
    df3 = pd.DataFrame(np.random.randint(0, 3, (2, 3)))

    >>> attrmap('shape', df1, df2, df3)
    [(4, 5), (4, 5), (2, 3)]

    net = nn.Sequential(...)
    >>> attrmap('shape', *net.parameters())
    [torch.Size([5, 3]),
     torch.Size([16, 4]),
     torch.Size([16, 3]),
     torch.Size([16])]
    """
    return [getattr(arg, attr) for arg in args]


def identity(x):
    """Returns the input argument. Sometimes it is convenient to have this if
    we sometimes apply a function to an item: rather than defining a None
    variable, sometimes setting it to a function, then checking if it's None
    every time we're about to call it, we can set the default as identity and
    safely call it without checking.

    Parameters
    ----------
    x: any

    Returns
    -------
    x: Unchanged input.
    """
    return x


def max_key(d, fn=identity):
    """Find the maximum value in a dictionary and return the associated key.
    If we want to compare values using something other than their numeric
    values, we can specify a function. For example, with a dict mapping strings
    to strings, fn=len would return the key with the longest value.

    Parameters
    ----------
    d: dict
        Values to select from.
    fn: callable
        Takes 1 argument (a single value from d.values()) and returns a number.
        This will be used to sort the items.

    Returns
    -------
    A key from dict `d`.
    """
    return max(d.items(), key=lambda x: fn(x[1]))[0]


def cd_root(root_subdir='notebooks'):
    """Run at start of Jupyter notebook to enter project root.

    Parameters
    ----------
    root_subdir: str
        Name of a subdirectory contained in the project root directory.
        If not found in the current working directory, this will move
        to the parent directory.

    Examples
    --------
    Sample file structure (abbreviated):
    my_project/
        py/
            fetch_raw_data.py
        notebooks/
            nb01_eda.ipynb

    Running cd_root() from nb01_eda.ipynb will change the working
    directory from notebooks/ to my_project/, which is typically the
    same directory we'd run scripts in py/ from. This makes converting
    from notebooks to scripts easier.
    """
    if root_subdir not in next(os.walk('.'))[1]:
        os.chdir('..')
    print('Current directory:', os.getcwd())


SENTINEL = object()
