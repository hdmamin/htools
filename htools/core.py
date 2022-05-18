from bz2 import BZ2File
from collections import Counter, Sequence, Iterable, \
    Mapping
from functools import partial
import gc
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from inspect import signature, getattr_static, ismethod, getmembers, getmodule
from itertools import chain
import json
import mimetypes
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
from random import choice, choices
import re
import smtplib
from string import ascii_letters
from subprocess import run, check_output
import sys
import time
from tqdm.auto import tqdm
import wordninja as wn

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
    return arg in signature(func).parameters


# def quickmail(subject, message, to_email, from_email=None,
#               attach_paths=(), verbose=True, password=None):
#     """Send an email.
#
#     Parameters
#     -----------
#     subject: str
#         Subject line of email.
#     message: str
#         Body of email.
#     to_email: str
#         Recipient's email. This can also be a verizon phone number in the form
#         3332221111@vtext.com (notice no extra leading 1), in which case this
#         will send an sms. In theory, you should be able to send mms
#         (i.e. include image(s)) by using a format like 3332221111@vzwtext.com,
#         but this doesn't seem to be working as of 2/10/22.
#     from_email: str
#         Gmail address being used to send email.
#     attach_paths: str or listlike
#         Paths to files to attach. Currently supports text (.txt, .md, etc.;
#         as of 2/11/22, gmail blocks executable attachments like .py)
#         and image (.jpg, .png, etc.) files.
#
#     Returns
#     --------
#     None
#     """
#     # Load email username. Error handling takes place in config functions.
#     from_email = from_email or get_default_user()
#     if not from_email: return None
#
#     # Load email password.
#     password = password or get_credentials(from_email)
#     if not password: return None
#
#     # Create message and add text if specified.
#     msg = MIMEMultipart()
#     msg['Subject'] = subject
#     msg['From'] = from_email
#     msg['To'] = to_email
#     if message: msg.attach(MIMEText(message))
#
#     # Load and attach file(s). Paths must be strings, not Paths, for mimetypes.
#     for path in map(str, tolist(attach_paths)):
#         ftype = mimetypes.guess_type(path)[0].split('/')[0]
#         if ftype == 'text':
#             mime_cls = MIMEText
#             mode = 'r'
#             encoder = identity
#             kwargs = {}
#         elif ftype == 'image':
#             mime_cls = MIMEImage
#             mode = 'rb'
#             encoder = encoders.encode_base64
#             kwargs = {'name': os.path.basename(path)}
#         else:
#             raise ValueError('Attached file should be a text or image file. '
#                              f'We parsed your file as type {ftype}.')
#         with open(path, mode) as f:
#             attachment = mime_cls(f.read(), **kwargs)
#         # show up as an attachment rather than an embedded object, but
#         # sometimes the later might be preferable.
#         attachment.add_header('Content-Disposition', 'attachment',
#                               filename=os.path.basename(path))
#         encoder(attachment)
#         msg.attach(attachment)
#
#     # Access server and send email.
#     server = smtplib.SMTP(host='smtp.gmail.com', port=587)
#     server.starttls()
#     server.login(user=from_email, password=password)
#     server.sendmail(from_email, to_email, msg.as_string())
#     if verbose: print(f'Email sent to {to_email}.')

def quickmail(subject, message, to_email, from_email=None,
              attach_paths=(), verbose=True, password=None):
    """Send an email.

    Parameters
    -----------
    subject: str
        Subject line of email.
    message: str
        Body of email.
    to_email: str
        Recipient's email. This can also be a verizon phone number in the form
        3332221111@vtext.com (notice no extra leading 1), in which case this
        will send an sms. In theory, you should be able to send mms
        (i.e. include image(s)) by using a format like 3332221111@vzwtext.com,
        but this doesn't seem to be working as of 2/10/22.
    from_email: str
        Gmail address being used to send email.
    attach_paths: str or listlike
        Paths to files to attach. Currently supports text (.txt, .md, etc.;
        as of 2/11/22, gmail blocks executable attachments like .py)
        and image (.jpg, .png, etc.) files.

    Returns
    --------
    None
    """
    # Load email username. Error handling takes place in config functions.
    from_email = from_email or get_default_user()
    if not from_email: return None

    # Load email password.
    password = password or get_credentials(from_email)
    if not password: return None

    # Create message and add text if specified.
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    if message: msg.attach(MIMEText(message))

    # Load and attach file(s). Paths must be strings, not Paths, for mimetypes.
    for path in map(str, tolist(attach_paths)):
        ftype = mimetypes.guess_type(path)[0].split('/')[0]
        if ftype == 'text':
            mime_cls = MIMEText
            mode = 'r'
            encoder = identity
            kwargs = {}
        elif ftype == 'image':
            mime_cls = MIMEImage
            mode = 'rb'
            encoder = encoders.encode_base64
            kwargs = {'name': os.path.basename(path)}
        else:
            raise ValueError('Attached file should be a text or image file. '
                             f'We parsed your file as type {ftype}.')
        with open(path, mode) as f:
            attachment = mime_cls(f.read(), **kwargs)
        # show up as an attachment rather than an embedded object, but
        # sometimes the later might be preferable.
        attachment.add_header('Content-Disposition', 'attachment',
                              filename=os.path.basename(path))
        encoder(attachment)
        msg.attach(attachment)

    # Access server and send email.
    hosts = {'gmail.com': 'smtp.gmail.com',
             'outlook.com': 'smtp-mail.outlook.com'}
    try:
        host = hosts[from_email.split('@')[-1]]
    except KeyError:
        raise ValueError(f'Unrecognized host {from_email.split("@")[-1]}. '
                         f'We currently support {hosts.keys()}.')
    server = smtplib.SMTP(host=host, port=587)
    server.starttls()
    server.login(user=from_email, password=password)
    server.sendmail(from_email, to_email, msg.as_string())
    if verbose: print(f'Email sent to {to_email}.')


def hsplit(text, sep, group=True, attach=True):
    """Flexible string splitting that retains the delimiter rather, unlike
    the built-in str.split() method.

    NOTE: I recently observed behavior suggesting separators with special
    characters (e.g. "\n") may not work as expected for some settings. It
    should work when group=True and attach=True though since I rewrote that
    with new logic without the re module.

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
            return _grouped_split(text, sep)

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


def _grouped_split(text, sep):
    """Hsplit helper for case where group=True and attach=True (see hsplit
    docs). Old re.find() method didn't work right when sep had special
    characters (e.g. "\n").
    """
    res = []
    toks = text.split(sep)
    max_idx = len(toks) - 1
    for i, tok in enumerate(toks):
        if tok:
            if i < max_idx: tok += sep
            res.append(tok)
        elif i < max_idx:
            if res:
                res[-1] += sep
            else:
                res.append(sep)
    return res


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

    # Store in dict to make it easier to add additional formats in future.
    ext2data = {
        'json': (open, '', json),
        'pkl': (open, 'b', pickle),
        'zip': (BZ2File, '', pickle),
    }
    if ext not in ext2data:
        raise InvalidArgumentError(
            'Invalid extension. Make sure your filename ends with '
            '.json, .pkl, or .zip.'
        )

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
        File name to save object to. Should end with .txt, .sh, md, .pkl, .zip,
        or .json depending on desired output format. If .zip is used, object
        will be zipped and then pickled. (.sh and .md will be treated
        identically to .txt.)
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
    if path.suffix[1:] in ('txt', 'sh', 'md', 'py'):
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
        '.txt', '.sh', 'md', '.json', '.pkl', or '.zip'.
    verbose : bool, optional
        If True, will print message stating where object was loaded from.

    Returns
    -------
    object: The Python object that was pickled to the specified file.
    """
    path = Path(path)
    if path.suffix[1:] in ('txt', 'sh', 'md', 'py'):
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

    def __init__(self, *funcs):
        """
        Parameters
        ----------
        *funcs: function(s)
            One or more functions to apply in the specified order.
        """
        # Make `funcs` mutable. Could use @htools.meta.delegate('funcs')
        # but not sure if that would cause circular import issues. Check later.
        self.funcs = list(funcs)

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
        # Try to display each item in the form that was likely passed in: for
        # functions, this is the name, but for callable classes this is
        # the str representation of the object, not the class itself.
        names = ',\n\t'.join(str(f) if hasattr(f, '__call__') else func_name(f)
                             for f in self.funcs)
        return f'{type(self).__name__}(\n\t{names}\n)'


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
    if random: return choice(list(it))
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


def amap(attr, *args):
    """More convenient syntax for quick data exploration. Get an attribute
    value for multiple objects. Name is short for "attrmap".

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

    >>> amap('shape', df1, df2, df3)
    [(4, 5), (4, 5), (2, 3)]

    net = nn.Sequential(...)
    >>> amap('shape', *net.parameters())
    [torch.Size([5, 3]),
     torch.Size([16, 4]),
     torch.Size([16, 3]),
     torch.Size([16])]
    """
    return [getattr(arg, attr) for arg in args]


def smap(*x):
    """Get shape of each array/tensor in a list or tuple.

    Parameters
    ----------
    *x: np.arrays or torch.tensors
        We use star unpacking here to create a consistent interface with amap()
        and lmap().

    Returns
    -------
    list: Shape of each array/tensor in input.
    """
    return amap('shape', *x)


def sleepy_range(*args, wait=1, wait_before=True):
    """Convenience function: we often want to create a loop that mimics doing
    some time intensive thing on each iteration. This is just like the built-in
    range function (not technically a function!) but with a sleep period baked
    in, making it particularly useful for list comprehensions where this would
    be tricky otherwise. Note: unlike range, calling this is destructive.
    See examples.

    Parameters
    ----------
    args: int
        Passed on to range().
    wait: int or float
        Number of seconds to wait on each iteration. Remember this is a keyword
        only argument for compatibility with the range interface.
    wait_before: bool
        Determines whether to sleep before or after yielding the number.
        Defaults to before to mimic "doing work" before producing some result.

    Examples
    --------
    # Takes 6 seconds to create this list.
    >>> [i for i in sleepy_range(3, wait=2)]
    [0, 1, 2]

    >>> srange = sleepy_range(0, 6, 2, wait_before=False)
    >>> for i in srange:
    >>>     print(i)
    0
    2
    4
    >>> for i in srange:
    >>>     print(i)

    # Notice this cannot be used again without manually calling sleepy_range.
    """
    for i in range(*args):
        if wait_before: time.sleep(wait)
        yield i
        if not wait_before: time.sleep(wait)


def venumerate(iterable, start=0, freq=1, print_before=True,
               message_format='{}'):
    """Verbose enumerate: simple convenience function that's a drop-in
    replacement for enumerate. It prints updates as we iterate over some
    object. TQDM progress bar may not be available in some cases (e.g. we
    don't know the length of the interval, or possible some cases using
    concurrency?), and this function gives us some way to keep an eye on
    progress. Mainly intended as a convenience for list comprehensions, since
    in a standard for loop we could easily add this logic.

    Parameters
    ----------
    iterable: Iterable
        The object to iterate over.
    start: int
        Passed on to enumerate - the first index to use when counting.
    freq: int
        Frequency with which to print updates (i.e. updates are printed when
        i is divisible by freq).
    print_before: bool
        Specifies whether to print the message before yielding the i'th value
        or after.
    message_format: str
        Used to format the message that will be displayed when i is divisible
        by freq. Defaults to just printing i.
    """
    for i, x in enumerate(iterable, start=start):
        if i % freq == 0 and print_before: print(message_format.format(i))
        yield i, x
        if i % freq == 0 and not print_before: print(message_format.format(i))


def method_of(meth):
    """Retrieve the class a method belongs to. This will NOT work on
    attributes. Also, this won't help if your goal is to retrieve an instance:
    this returns the type of the instance. Not thoroughly tested but it seems
    to work regardless of whether you pass in meth from an instance or a class
    (the output is the same in both cases).

    Parameters
    ----------
    meth: MethodType
        The method to retrieve the class of.

    Returns
    -------
    type: The class which defines the method in question.

    Examples
    --------
    class Foo:
        def my_method(self, x):
            return x*2

    f = Foo()
    assert method_of(Foo.my_method) == method_of(f.my_method) == Foo
    """
    cls, name = meth.__qualname__.split('.')
    return dict(getmembers(getmodule(meth)))[cls]


def hasstatic(cls, meth_name):
    """Check if a class possesses a staticmethod of a given name. Similar to
    hasattr. Note that isinstance(cls.meth_name, staticmethod) would always
    return False: we must use getattr_static or cls.__dict__[meth_name]
    to potentially return True.

    Parameters
    ----------
    cls: Type or any
        A class or an instance (seems to work on both, though more extensive
        testing may be needed for more complex scenarios).
    meth_name: str
        Name of method to check. If the class/instance does not contain any
        attribute with this name, function returns False.

    Returns
    -------
    bool: True if `cls` has a staticmethod with name `meth_name`.
    """
    return isinstance(getattr_static(cls, meth_name, None), staticmethod)


def isstatic(meth):
    """Companion to hasstatic that checks a method itself rather than a class
    and method name. It does use hasstatic under the hood.
    """
    # First check isn't required but I want to avoid reaching the hackier bits
    # of code if necessary. This catches regular methods and attributes.
    if ismethod(meth) or not callable(meth): return False
    parts = getattr(meth, '__qualname__', '').split('.')
    if len(parts) != 2: return False
    cls = method_of(meth)
    return hasstatic(cls, parts[-1])


def has_classmethod(cls, meth_name):
    """Check if a class has a classmethod with a given name.
    Note that isinstance(cls.meth_name, classmethod) would always
    return False: we must use getattr_static or cls.__dict__[meth_name]
    to potentially return True.

    Parameters
    ----------
    cls: type or obj
        This is generally intended to be a class but it should work on objects
        (class instances) as well.
    meth_name: str
        The name of the potential classmethod to check for.

    Returns
    -------
    bool: True if cls possesses a classmethod with the specified name.
    """
    return isinstance(getattr_static(cls, meth_name), classmethod)


def is_classmethod(meth):
    """Companion to has_classmethod that checks a method itself rather than a
    class and a method name. It does use has_classmethod under the hood.
    """
    if not ismethod(meth): return False
    parts = getattr(meth, '__qualname__', '').split('.')
    if len(parts) != 2: return False
    cls = method_of(meth)
    return has_classmethod(cls, parts[-1])


def parallelize(func, items, total=None, chunksize=1_000, processes=None):
    """Apply a function to a sequence of items in parallel. A progress bar
    is included.

    Parameters
    ----------
    func: function
        This will be applied to each item in `items`.
    items: Iterable
        Sequence of items to apply `func` to.
    total: int or None
        This defaults to the length of `items`. In the case that items is a
        generator, this lets us pass in the length explicitly. This lets tdqm
        know how quickly to advance our progress bar.
    chunksize: int
        Positive int that determines the size of chunks submitted to the
        process pool as separate tasks. Multiprocessing's default is 1 but
        larger values should speed things up, especially with long sequences.
    processes: None
        Optionally set number of processes to run in parallel.

    Returns
    -------
    list
    """
    total = total or len(items)
    with Pool(processes) as p:
        res = list(tqdm(p.imap(func, items, chunksize=chunksize),
                        total=total))
    return res


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


def always_true(x, *args, **kwargs):
    """Similar to `identity` but returns True instead of x. I'm tempted to name
    this `true` but I fear that will cause some horrible bugs where I
    accidentally use this when I want to use True.
    """
    return True


def ifnone(arg, backup):
    """Shortcut to provide a backup value if an argument is None. Commonly used
    for numpy arrays since their truthiness is ambiguous.

    Parameters
    ----------
    arg: any
        We will check if this is None.
    backup: any
        This will be returned if arg is None.

    Returns
    -------
    Either `arg` or `backup` will be returned.
    """
    return arg if arg is not None else backup


def listlike(x):
    """Checks if an object is a list/tuple/set/array etc. Strings and
    mappings (e.g. dicts) are not considered list-like.
    """
    return isinstance(x, Iterable) and not isinstance(x, (str, Mapping))


def tolist(x, length_like=None, length=None,
           error_message='x length does not match desired length.'):
    """Helper to let a function accept a single value or a list of values for
    a certain parameter.

    WARNING: if x is a primitive and you specify a length (either via
    `length_like` or `length`, the resulting list will contain multiple
    references to the same item). This is mostly intended for use on lists of
    floats or ints so I don't think it's a problem, but keep this in mind when
    considering using this on mutable objects.

    Parameters
    ----------
    x: Iterable
        Usually either a list/tuple or a primitive.
    length_like: None or object
        If provided, we check that x is the same length. If x is a primitive,
        we'll make it the same length.
    length: None or int
        Similar to `length_like` but lets us specify the desired length
        directly. `length_like` overrides this, though you should only provide
        one or the other.
    error_message: str
        Displayed in the event that a desired length is specified and x is
        list-like and does not match that length. You can pass in your own
        error message if you want something more specific to your current use
        case.

    Returns
    -------
    list

    Examples
    --------
    def train(lrs):
        lrs = tolist(lrs)
        ...

    We can now pass in a single learning rate or multiple.
    >>> train(3e-3)
    >>> train([3e-4, 3e-3])
    """
    if length_like is not None: length = len(length_like)

    # Case 1. List-like x
    if listlike(x):
        if length:
            assert len(x) == length, error_message
        return list(x)

    # Case 2. Dict-like x
    if isinstance(x, Mapping):
        raise ValueError('x must not be a mapping. It should probably be a '
                         'primitive (str, int, etc.) or a list-like object '
                         '(tuple, list, set).')

    # Case 3. Primitive x
    return [x] * (length or 1)


def xor_none(*args, n=1):
    """Checks that exactly 1 (or n) of inputs is not None. Useful for
    validating optional function arguments (for example, ensuring the user
    specifies either a directory name or a list of files but not both.

    Parameters
    ----------
    args: any
    n: int
        The desired number of non-None elements. Usually 1 but we allow the
        user to specify other values.

    Returns
    -------
    None: This will raise an error if the condition is not satisfied. Do not
    use this as an if condition (e.g. `if xor_none(a, b): print('success')`.
    This would always evaluate to False because the function doesn't explicitly
    return a value so we get None.
    """
    if sum(bool(arg is not None) for arg in args) != n:
        raise ValueError(f'Exactly {n} of args must be not None.')


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


def is_builtin(x, drop_callables=True):
    """Check if an object is a Python built-in object.

    Parameters
    ----------
    x: object
    drop_callables: bool
        If True, return False for callables (basically functions, methods, or
        classes). These typically will return True otherwise since they are of
        class `type` or `builtin_function_or_method`.

    Returns
    -------
    bool: True if `x` is a built-in object, False otherwise.
    """
    def _builtin(x, drop_callables):
        if callable(x) and drop_callables:
            return False
        return x.__class__.__module__ == 'builtins'
    builtin = partial(_builtin, drop_callables=drop_callables)
    # Check mapping first because mappings are iterable.
    if isinstance(x, Mapping):
        return builtin(x) and all(builtin(o) for o in flatten(x.items()))
    elif isinstance(x, Iterable):
        return builtin(x) and all(builtin(o) for o in flatten(x))
    return builtin(x)


def hashable(x):
    """Check if an object is hashable. Hashable objects will usually be
    immutable though this is not guaranteed.

    Parameters
    ----------
    x: object
        The item to check for hashability.

    Returns
    -------
    bool: True if `x` is hashable (suggesting immutability), False otherwise.
    """
    try:
        _ = hash(x)
        return True
    except TypeError:
        return False


def fgrep(text, term, window=25, with_idx=False, reverse=False):
    """Search a string for a given term. If found, print it with some context.
    Similar to `grep -C 1 term text`. `fgrep` is short for faux grep.

    Parameters
    ----------
    text: str
        Text to search.
    term: str
        Term to look for in text.
    window: int
        Number of characters to display before and after the matching term.
    with_idx: bool
        If True, return index as well as string.
    reverse: bool
        If True, reverse search direction (find last match rather than first).

    Returns
    -------
    str or tuple[int, str]: The desired term and its surrounding context.
        If the term isn't present, an empty string is returned. If
        with_idx=True, a tuple of (match index, string with text) is returned.
    """
    idx = text.rfind(term) if reverse else text.find(term)
    if idx == -1:
        res = ''
    else:
        res = text[max(idx-window, 0):idx+window]
    return (idx, res) if with_idx else res


def spacer(char='-', n_chars=79, newlines_before=1, newlines_after=1):
    """ Get string to separate output when printing output for multiple items.

    Parameters
    ----------
    char: str
        The character that will be printed repeatedly.
    n_chars: int
        The number of times to repeat `char`. We expect that `char` is a
        single character so this will be the total line length.
    newlines_before: int
        Number of newline characters to add before the spacer.
    newlines_after: int
        Number of newline characters to add after the spacer.

    Returns
    -------
    str
    """
    return '\n'*newlines_before + char * n_chars + '\n'*newlines_after


def func_name(func):
    """Usually just returns the name of a function. The difference is this is
    compatible with functools.partial, which otherwise makes __name__
    inaccessible.

    Parameters
    ----------
    func: callable
        Can be a function, partial, or callable class.
    """

    assert callable(func), 'Input must be callable.'

    try:
        res = func.__name__
    except AttributeError:
        if isinstance(func, partial):
            return func_name(func.func)
        else:
            return func.__class__.__name__
    except Exception as e:
        raise e
    return res


def snake2camel(text):
    """Convert snake case to camel case. This assumes the input is valid snake
    case (if you have some weird hybrid of snake and camel case, for instance,
    you'd want to do some preprocessing first).

    Parameters
    ----------
    text: str
        Snake case string, e.g. vader_sentiment_score.

    Returns
    -------
    str: `text` converted to camel case, e.g. vaderSentimentScore.
    """
    res = []
    prev = ''
    for char in text:
        if char != '_':
            # Check if res is empty because of case with leading underscore.
            res.append(char.upper() if prev == '_' and res else char)
        prev = char
    return ''.join(res)


def camel2snake(text):
    """Convert camel case to snake case. This assumes the input is valid camel
    case (if you have some weird hybrid of camel and snake case, for instance,
    you'd want to do some preprocessing first).

    Parameters
    ----------
    text: str
        Camel case string, e.g. vaderSentimentScore.

    Returns
    -------
    str: `text` converted to snake case, e.g. vader_sentiment_score.
    """
    res = []
    for char in text:
        if char.islower():
            res.append(char)
        else:
            res.extend(['_', char.lower()])
    return ''.join(res)


def to_snake(text):
    """Experimental feature: tries to convert any common format to snake case.
    This hasn't been extensively tested but it seems to work with snake case
    (no change), camel case, upper camel case, words separated by
    hyphens/dashes/spaces, and combinations of the above. It may occasionally
    split words that should not be split, though this should be rare if names
    use actual English words (this might not work so well on fastai-style
    variable names (very short, e.g. "tfms" for "transforms"), but the intended
    use case is mostly for fixing column names in pandas.

    Parameters
    ----------
    text: str

    Returns
    -------
    str: Input text converted to snake case.
    """
    return '_'.join(wn.split(text.lower()))


def to_camel(text):
    """Experimental feature: tries to convert any common format to camel case.
    This hasn't been extensively tested but it seems to work with camel case
    (no change), snake case, upper camel case, words separated by
    hyphens/dashes/spaces, and combinations of the above. It may occasionally
    split words that should not be split, though this should be rare if names
    use actual English words (this might not work so well on fastai-style
    variable names (very short, e.g. "tfms" for "transforms"), but the intended
    use case is mostly for fixing column names in pandas.

    Parameters
    ----------
    text: str

    Returns
    -------
    str: Input text converted to snake case.
    """
    return ''.join(w.title() if i > 0 else w
                   for i, w in enumerate(wn.split(text.lower())))


def kwargs_fallback(self, *args, assign=False, **kwargs):
    """Use inside a method that accepts **kwargs. Sometimes we want to use
    an instance variable for some computation but want to give the user the
    option to pass in a new value to the method (often ML hyperparameters) to
    be used instead. This function makes that a little more convenient.

    Parameters
    ----------
    self: object
        The class instance. In most cases users will literally pass `self` in.
    args: str
        One or more names of variables to use this procedure on.
    assign: bool
        If True, any user-provided kwargs will be used to update attributes of
        the instance. If False (the default), they will be used in computation
        but won't change the state of the instance.
    kwargs: any
        Just forward along the kwargs passed to the method.

    Returns
    -------
    list or single object: If more than one arg is specified, a list of values
        is returned. For just one arg, a single value will be returned.

    Examples
    --------
    class Foo:

        def __init__(self, a, b=3, c=('a', 'b', 'c')):
            self.a, self.b, self.c = a, b, c

        def walk(self, d, **kwargs):
            a, c = kwargs_fallback(self, 'a', 'c', **kwargs)
            print(self.a, self.b, self.c)
            print(a, c, end='\n\n')

            b, c = kwargs_fallback(self, 'b', 'c', assign=True, **kwargs)
            print(self.a, self.b, self.c)
            print(b, c)

    # Notice the first `kwargs_fallback` call doesn't change attributes of f
    # but the second does. In the first block of print statements, the variable
    # `b` does not exist yet because we didn't include it in *args.
    >>> f = Foo(1)
    >>> f.walk(d=0, b=10, c=100)
    1 3 ('a', 'b', 'c')
    1 100

    1 10 100
    10 100
    """
    res = []
    for arg in args:
        # Don't just use `kwargs.get(arg) or ...` because this doesn't work
        # well when we pass in a numpy array or None.
        val = kwargs[arg] if arg in kwargs else getattr(self, arg)
        res.append(val)
        if assign: setattr(self, arg, val)
    return res if len(res) > 1 else res[0]


def cd_root(root_subdir='notebooks', max_depth=4):
    """Run at start of Jupyter notebook to enter project root.

    Parameters
    ----------
    root_subdir: str
        Name of a subdirectory contained in the project root directory.
        If not found in the current working directory, this will move
        to the parent directory repeatedly until it is found. Choose carefully:
        if you have multiple directories with the same name in your directory
        structure (e.g. ~/htools/lib/htools), 'htools' would be a bad choice
        if you want to end up in ~).
    max_depth: int
        Max number of directory levels to traverse. Don't want to get stuck in
        an infinite loop if we make a mistake.

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
    changes = 0
    start_dir = os.getcwd()
    while root_subdir not in next(os.walk('.'))[1]:
        if changes >= max_depth:
            os.chdir(start_dir)
            raise RuntimeError('Exceeded max_depth. Check that your '
                               'root_subdir is <= max_depth directories away.')
        os.chdir('..')
        changes += 1
    print('Current directory:', os.getcwd())


def ngrams(word, n=3, step=1, drop_last=False):
    """To get non-overlapping sequences, pass in same value for `step` as `n`.
    """
    stop = max(1, step+len(word)-n)
    ngrams_ = []
    for i in range(0, stop, step):
        ngrams_.append(word[i:i+n])
    if drop_last and len(ngrams_[-1]) < n: ngrams_ = ngrams_[:-1]
    return ngrams_


def shell(cmd, return_output=True):
    """Execute shell command (between subprocess and os, there's ~5 different
    ways to do this and I always forget which I want. This is just a way for me
    to choose once and not have to decide again. There are rare situations
    where we may need a different function (subprocess.run is blocking; if we
    want to launch a process and continue the script without waiting for
    completion, we can use subprocess.check_call).

    Parameters
    ----------
    cmd: str
        Example: 'ls *.csv'
    return_output: bool
        If True, return the output of the command: e.g. if cmd is
        'pip show requests', this would return a string containing information
        about the version of the requests library you have installed. If False,
        we return a tuple of (return code (0/1), stderr, stdout). I've noticed
        the latter 2 are usually None though - need to read more into
        subprocess docs to figure out why this is happening.

    Returns
    -------
    tuple: returncode (int), stderr, stdout. I believe stderr and stdout are
    None if nothing is returned and str otherwise.
    """
    parts = cmd.split()
    if return_output:
        return check_output(parts).decode()
    res = run(parts)
    return res.returncode, res.stderr, res.stdout


def set_summary(x1, x2, info=('first_only', 'second_only')):
    """Summarize set comparison between two iterables (they will be converted
    to sets internally).

    Parameters
    ----------
    info: Iterable[str]
        Determines what info to return. 'first_only' returns items only in the
        first iterable, 'second_only' returns items only in the second, 'and'
        returns items in both, and 'or' returns items in either.

    Returns
    -------
    dict[str, set]: Maps str in `info` to set of items.
    """
    s1, s2 = set(x1), set(x2)
    res = {'and': s1 & s2,
           'or': s1 | s2,
           'first_only': s1 - s2,
           'second_only': s2 - s1}
    for k, v in res.items():
        print(f'{k}: {len(v)} items')
    return select(res, keep=list(info))


def random_str(length, lower=False, valid=tuple(ascii_letters + '0123456789')):
    """Generate random string of alphanumeric characters.

    Parameters
    ----------
    length: int
        Number of characters in output string.
    lower: bool
        If True, the output will be lowercased.
    valid: Iterable
        List-like container of valid characters (strings) to sample from.

    Returns
    -------
    str: `length` characters long.
    """
    text = ''.join(choices(valid, k=length))
    return text.lower() if lower else text


SENTINEL = object()

