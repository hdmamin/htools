from bz2 import BZ2File
from collections import Counter, namedtuple, UserDict, Sequence, Iterable, \
    Mapping
from email.mime.text import MIMEText
from fuzzywuzzy import fuzz, process
import inspect
from itertools import chain
import json
import os
from pathlib import Path
import pickle
import re
import smtplib
import sys

from htools.config import get_credentials, get_default_user


def Args(**kwargs):
    """Wrapper to easily create a named tuple of arguments. Functions sometimes
    return multiple values, and we have a few options to handle this: we can
    return them as a regular tuple, but it is often convenient to be able to
    reference items by name rather than position. If we want the output to be
    mutable, we can return a dictionary, but this still requires more space
    than a tuple and bracket notation is arguably less convenient than dot
    notation. We can create a new namedtuple inside the function, but this
    kind of seems like overkill to create a new type of namedtuple for each
    function.

    Instead, this lets us create a namedtuple of Args on the fly just as easily
    as making a dictionary.

    Parameters
    ----------

    Examples
    --------
    def math_summary(x, y):
        sum_ = x + y
        prod = x * y
        diff = x - y
        quotient = x / y
        return Args(sum=sum_,
                    product=prod,
                    difference=diff,
                    quotient=quotient)

    >>> results = math_summary(4, 2)
    >>> results.product

    8

    >>> results.quotient

    2

    >>> results

    Args(sum=6, product=8, difference=2, quotient=2)
    """
    args = namedtuple('Args', kwargs.keys())
    return args(*kwargs.values())


class InvalidArgumentError(Exception):
    pass


class FuzzyKeyDict(dict):
    """Dictionary that will try to find similar keys if a key is missing and
    return their corresponding values. This could be useful when working with
    embeddings, where we could try mapping missing words to a combination of
    existing words.

    Examples
    --------
    d = FuzzyKeyDict(limit=3, verbose=True)
    d['dog'] = 0
    d['cat'] = 1
    d['alley cat'] = 2
    d['pig'] = 3
    d['cow'] = 4
    d['cowbell'] = 5
    d['baby cow'] = 6

    # Keys and similarity scores for the most similar keys.
    >>> d.similar_keys('house cat')
    [('alley cat', 56), ('cat', 50), ('cowbell', 25)]

    # "house cat" not in dict so we get the values for the most similar keys.
    >>> d['house cat']
    [2, 1, 5]

    # "cat" is in dict so output is an integer rather than a list.
    >>> d['cat']
    1
    """

    def __init__(self, data=None, limit=3):
        """
        Parameters
        ----------
        data: Iterable (optional)
            Sequence of pairs, such as a dictionary or a list of tuples. If
            provided, this will be used to populate the FuzzyKeyDict.
        limit: int
            Number of similar keys to find when trying to retrieve the value
            for a missing key.
        """
        if isinstance(data, Mapping):
            for k, v in data.items():
                self[k] = v
        elif isinstance(data, Iterable):
            for k, v in data:
                self[k] = v
        self.limit = limit

    def __getitem__(self, key):
        """
        Returns
        -------
        any or list[any]: If key is present in dict, the corresponding value
            is returned. If not, the n closest keys are identified and their
            corresponding values are returned in a list (where n is defined
            by the `limit` argument specified in the constructor). Values are
            sorted in descending order by the neighboring keys' similarity to
            the missing key in.
        """
        try:
            return super().__getitem__(key)
        except KeyError:
            return [self[k] for k in self.similar_keys(key)]

    def similar_keys(self, key, return_distances=False):
        pairs = process.extract(key, self.keys(), limit=self.limit,
                                scorer=fuzz.ratio)
        if return_distances:
            return pairs
        return [p[0] for p in pairs]


class LambdaDict(UserDict):
    """Create a default dict where the default function can accept parameters.
    Whereas the defaultdict in Collections can set the default as int or list,
    here we can pass in any function where the key is the parameter.
    """

    def __init__(self, default_function):
        """
        Parameters
        ----------
        default_function: function
            When referencing a key in a LambdaDict object that has not been
            added yet, the value will be the output of this function called
            with the key passed in as an argument.
        """
        super().__init__()
        self.f = default_function

    def __missing__(self, key):
        self[key] = self.f(key)
        return self[key]


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
    """Checks if a function has a given argument.
    Works with args and kwargs as well if you exclude the
    stars. See example below.
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
    if ext not in {'json', 'pkl', 'zip'}:
        raise InvalidArgumentError(
            'Invalid extension. Make sure your filename ends with .json, '
            '.pkl, or .zip.'
        )
        
    # Store in dict to make it easier to add additional formats in future.
    ext2data = {'pkl': (open, 'b', pickle), 
                'zip': (BZ2File, '', pickle), 
                'json': (open, '', json)}
    opener, mode_suffix, saver = ext2data[ext]
    return opener, mode + mode_suffix, saver


def save(obj, path, verbose=True):
    """Wrapper to save data as pickle (optionally zipped) or json.

    Parameters
    -----------
    obj: any
        Object to save. This will be pickled/jsonified/zipped inside the
        function - do not convert it before-hand.
    path: str
        File name to save object to. Should end with .pkl, .zip, or
        .json depending on desired output format. If .zip is used, object will
        be zipped and then pickled.
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
        path.write_text(obj)
    else:
        opener, mode, saver = _read_write_args(str(path), 'w')
        with opener(path, mode) as f:
            saver.dump(obj, f)


def load(path, verbose=True):
    """Wrapper to load pickled (optionally zipped) or json data.
    
    Parameters
    ----------
    path : str
        File to load. File type will be inferred from extension.
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
    """Find the differences between two objects of the same type. This is a
    way to get more detail beyond whether two objects are equal or not.

    Parameters
    -----------
    obj1: any type
        An object.
    obj2: same type as obj1
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
    if obj1 == obj2:
        return {}

    assert type(obj1) == type(obj2), 'Objects must be the same type.'
    attr1, attr2 = hdir(obj1, **kwargs), hdir(obj2, **kwargs)
    assert attr1.keys() == attr2.keys(), 'Objects must have same attributes.'

    diffs = {}
    for (k1, v1), (k2, v2) in zip(attr1.items(), attr2.items()):
        # Only compare non-callable attributes.
        if not (methods or v1 == 'attribute'):
            continue

        # Comparisons work differently for numpy arrays.
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
    sequences, nested sequences, or any combination of these. This also returns
    a list rather than a generator.

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
            try:
                yield from _walk(group)
            except TypeError:
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


SENTINEL = object()
