import bz2
from collections import namedtuple
from contextlib import contextmanager
from email.mime.text import MIMEText
from functools import wraps, partial
import inspect
from itertools import chain
import os
import pickle
import re
import signal
import smtplib
import sys
import time

from htools.config import get_credentials, get_default_user


class AutoInit:
    """Mixin class where child class has a long list init arguments that will
    be assigned to the same name. Note that *args are not supported in the
    init method because each attribute that is defined in the resulting object
    must have a name. A variable length list of args can still be passed in as
    a single argument, of course, without the use of star unpacking.

    Examples
    --------
    Without AutoInit:

    class Child:
        def __init__(self, name, age, sex, hair, height, weight, grade, eyes):
            self.name = name
            self.age = age
            self.sex = sex
            self.hair = hair
            self.height = height
            self.weight = weight
            self.grade = grade
            self.eyes = eyes
        def __repr__(self):
            return f'Child(name={self.name}, age={self.age}, sex={self.sex}, '\
                   f'hair={self.hair}, weight={self.weight}, '\
                   f'grade={self.grade}, eyes={self.eyes})'

    With AutoInit:

    class Child(AutoInit):
        def __init__(self, name, age, sex, hair, height, weight, grade, eyes):
            super().__init__(locals())

    Note that we could also use the following method, though this is less
    informative when constructing instances of the child class and does not
    have the built in __repr__ that comes with AutoInit:

    class Child:
        def __init__(self, **kwargs):
            self.__dict__ = kwargs
    """
    def __init__(self, child_args):
        self._auto_init(child_args)

    def _auto_init(self, child_args):
        """Initialize variables.
        
        Parameters
        ----------
        child_args : dict
            Arguments passed to child class.
        """
        child_args.update(child_args.pop('kwargs', {}))
        self.__dict__ = {k: v for k, v in child_args.items()
                         if k != 'self' and not k.startswith('__')}
        self._init_keys = set(self.__dict__.keys())

    def __repr__(self):
        """Returns string representation of child class including variables
        used in init method. For the example in the class docstring, this would
        return:

        child = Child('Henry', 8, 'm', 'brown', 52, 70, 3, 'green')
        Child(name='Henry', age=8, sex='m', hair='brown', height=52, 
              weight=70, grade=3, eyes='green')
        
        Returns
        -------
        [type]
            [description]
        """
        args = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items()
                         if k in self._init_keys)
        return f'{self.__class__.__name__}({args})'


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


class TimeExceededError(Exception):
    pass


def timebox_handler(time, frame):
    raise TimeExceededError('Time limit exceeded.')


@contextmanager
def timebox(time):
    """Try to execute code for specified amount of time before throwing error.
    If you don't want to throw an error, use with a try/except block.

    Parameters
    ----------
    time: int
        Max number of seconds before throwing error.

    Examples
    --------
    with time_box(5) as tb:
        x = computationally_expensive_code()

    More permissive version:
    x = step_1()
    with timebox(5) as tb:
        try:
            x = slow_step_2()
        except TimeExceededError:
            pass
    """
    try:
        signal.signal(signal.SIGALRM, timebox_handler)
        signal.alarm(time)
        yield
    finally:
        signal.alarm(0)


def timeboxed(time):
    """Decorator version of timebox. Try to execute decorated function for
    `time` seconds before throwing exception.

    Parameters
    ----------
    time: int
        Max number of seconds before throwing error.

    Examples
    --------
    @timeboxed(5)
    def func(x, y):
        # If function does not complete within 5 seconds, will throw error.
    """
    def intermediate_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timebox(time) as tb:
                return func(*args, **kwargs)
        return wrapper
    return intermediate_wrapper


class cached_property:
    """Decorator for computationally expensive methods that should only be
    computed once (i.e. they take zero arguments aside from self and are slow
    to execute). Lowercase name is used for consistency with more decorators.
    Heavily influenced by example in `Python Cookbook` by David Beazley and
    Brian K. Jones. Note that, as with the @property decorator, no parentheses
    are used when calling the decorated method.

    Examples
    --------
    class Vocab:

        def __init__(self, tokens):
            self.tokens = tokens

        @cached_property
        def embedding_matrix(self):
            print('Building matrix...')
            # Slow computation to build and return a matrix of word embeddings.
            return matrix

    # First call is slow.
    >>> v = Vocab(tokens)
    >>> v.embedding_matrix

    Building matrix...
    [[.03, .5, .22, .01],
     [.4, .13, .06, .55]
     [.77, .14, .05, .9]]

    # Second call accesses attribute without re-computing 
    # (notice no "Building matrix" message).
    >>> v = Vocab(tokens)
    >>> v.embedding_matrix

    [[.03, .5, .22, .01],
     [.4, .13, .06, .55]
     [.77, .14, .05, .9]]
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        """This method is called when the variable being accessed is not in the
        instance's state dict. The next time the attribute is accessed, the 
        computed value will be in the state dict so this method (and the method
        in the instance itself) is not called again unless the attribute is 
        deleted.
        """
        # When attribute accessed as class method, instance is None.
        if instance is None:
            return self

        # When accessed as instance method, call method on instance as usual.
        # Then set instance attribute and return value.
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def typecheck(func_=None, **types):
    """Decorator to enforce type checking for a function or method. There are 
    two ways to call this: either explicitly passing argument types to the
    decorator, or letting it infer them using type annotations in the function
    that will be decorated. We allow multiple both usage methods since older
    versions of Python lack type annotations, and also because I feel the 
    annotation syntax can hurt readability.
    

    Parameters
    ----------
    func_: function
        The function to decorate. When using decorator with 
        manually-specified types, this is None. Underscore is used so that
        `func` can still be used as a valid keyword argument for the wrapped 
        function.
    types: type
        Optional way to specify variable types.
        
    Examples
    --------
    In the first example, we specify types directly in the decorator. Notice
    that they can be single types or tuples of types. You can choose to 
    specify types for all arguments or just a subset.
    
    @typecheck(x=float, y=(int, float), iters=int, verbose=bool)
    def process(x, y, z, iters=5, verbose=True):
        print(f'z = {z}')
        for i in range(iters):
            if verbose: print(f'Iteration {i}...')
            x *= y
        return x
    
    >>> process(3.1, 4.5, 0, 2.0)
    TypeError: iters must be <class 'int'>, not <class 'float'>.
    
    >>> process(3.1, 4, 'a', 1, False)
    z = a
    12.4
    
    Alternatively, you can let the decorator infer types using annotations
    in the function that is to be decorated. The example below behaves 
    equivalently to the explicit example shown above. Note that annotations
    regarding the returned value are ignored.
    
    @typecheck
    def process(x:float, y:(int, float), z, iters:int=5, verbose:bool=True):
        print(f'z = {z}')
        for i in range(iters):
            if verbose: print(f'Iteration {i}...')
            x *= y
        return x
        
    >>> process(3.1, 4.5, 0, 2.0)
    TypeError: iters must be <class 'int'>, not <class 'float'>.
    
    >>> process(3.1, 4, 'a', 1, False)
    z = a
    12.4
    """
    # Case 1: Pass keyword args to decorator specifying types.
    if not func_:
        return partial(typecheck, **types)
    # Case 2: Infer types from annotations. Skip if Case 1 already occurred.
    elif not types:
        types = {k: v.annotation 
                 for k, v in inspect.signature(func_).parameters.items()
                 if not v.annotation == inspect._empty}
    
    @wraps(func_)
    def wrapped(*args, **kwargs):
        fargs = inspect.signature(wrapped).bind(*args, **kwargs).arguments
        for k, v in types.items():
            if k in fargs and not isinstance(fargs[k], v):
                raise TypeError(
                    f'{k} must be {str(v)}, not {type(fargs[k])}.'
                )
        return func_(*args, **kwargs)
    return wrapped


class LambdaDict(dict):
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


def hmail(subject, message, to_email, from_email=None):
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
    None.
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


def htimer(func):
    """Provide conservative time estimate for a function to run. Behavior may
    not be interpretable for recursive functions.

    Parameters
    -----------
    func: function
        The function to time.

    Examples
    ---------
    import time

    @htimer
    def count_to(x):
        for i in range(x):
            time.sleep(0.5)

    >>> count_to(10)
    [TIMER]: function <count_to> executed in roughly 5.0365 seconds
    (conservatively).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        print(f'\n[TIMER]: function <{func.__name__}> executed in roughly '
              f'{round(time.time() - start, 4)} seconds (conservatively).\n')
        return output
    return wrapper


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


def save(obj, path, compress=True, verbose=True):
    """Wrapper to quickly save a pickled object.

    Parameters
    -----------
    obj: any
        Object to pickle.
    path: str
        File name to save pickled object to.
    compress: bool
        If True, will compress the pickled object using the bz library (in 
        this case, path should end with a .zip extension). If False, object
        will not be compressed and path should end with a .pkl extension.
    verbose: bool
        If True, print a message confirming that the data was pickled, along
        with its path.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if compress:
        with bz2.BZ2File(path, 'w') as f:
            pickle.dump(obj, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    if verbose:
        print(f'Data written to {path}.')


def load(path, decompress=True, verbose=True):
    """Wrapper to load a pickled object.
    
    Parameters
    ----------
    path : str
        File to load.
    decompress : bool, optional
        Pass in True if object is zipped, False otherwise.
    verbose : bool, optional
        If True, will print message stating where object was loaded from.
    
    Returns
    -------
    object: The Python object that was pickled to the specified file.
    """
    if decompress:
        with bz2.BZ2File(path, 'r') as f:
            data = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    if verbose:
        print(f'Object loaded from {path}.')
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
    """Error handling for list comprehensions.

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

    list(filter(None, [catch(lambda x: 1 / x, i) for i in range(3)]))
    >>> [1.0, 0.5]
    """
    try:
        return func(*args)
    except Exception as e:
        if verbose:
            print(e)
        return


def flatten(nested):
    """Flatten a nested sequence where the sub-items can be sequences or 
    primitives. This differs slightly from itertools chain methods because
    those require all sub-items to be sequences. This also returns a list 
    rather than a generator.

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
                yield from group
            except TypeError:
                yield group
    return list(_walk(nested))
