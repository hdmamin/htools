from abc import ABC, abstractmethod
from contextlib import contextmanager, redirect_stdout
from copy import copy, deepcopy
from functools import wraps, partial, update_wrapper
import inspect
import io
import logging
import os
from pathlib import Path
import signal
import sys
import time
from tqdm.auto import tqdm
import types
import warnings
from weakref import WeakSet

from htools import hdir, load, save, identity


class AutoInit:
    """Mixin class where child class has a long list of init arguments where
    the parameter name and the class attribute will be the same. Note that
    *args are not supported in the init method because each attribute that is
    defined in the resulting object must have a name. A variable length list
    of args can still be passed in as a single argument, of course, without the
    use of star unpacking.

    This updated version of AutoInit is slightly more user friendly than in V1
    (no more passing locals() to super()) but also slower and probably requires
    more testing (all because of the frame hack in the init method). Note that
    usage differs from the AutoInit present in htools<=2.0.0, so this is a
    breaking change.

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
            super().__init__()

    Note that we could also use the following method, though this is less
    informative when constructing instances of the child class and does not
    have the built in __repr__ that comes with AutoInit:

    class Child:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    """

    def __init__(self):
        # Calculate how many frames to go back to get child class.
        frame_idx = type(self).__mro__.index(AutoInit)
        attrs = {k: v for k, v in sys._getframe(frame_idx).f_locals.items()
                 if not k.startswith('__')}
        attrs.pop('self')
        bound = inspect.signature(self.__class__.__init__)\
                       .bind_partial(**attrs)

        # Flatten dict so kwargs are not listed as their own argument.
        bound.arguments.update(
            bound.arguments.pop('kwargs', {}).get('kwargs', {})
        )
        self._init_keys = set(bound.arguments.keys())
        for k, v in bound.arguments.items():
            setattr(self, k, v)

    def __repr__(self):
        """Returns string representation of child class including variables
        used in init method. For the example in the class docstring, this would
        return:

        child = Child('Henry', 8, 'm', 'brown', 52, 70, 3, 'green')
        Child(name='Henry', age=8, sex='m', hair='brown', height=52,
              weight=70, grade=3, eyes='green')

        Returns
        -------
        str
        """
        fstrs = (f'{k}={repr(getattr(self, k))}' for k in self._init_keys)
        return f'{self.__class__.__name__}({", ".join(fstrs)})'


def auto_repr(cls):
    """Class decorator that provides __repr__ method automatically
    based on __init__ parameters. This aims to provide a simpler alternative
    to AutoInit that does not require access to the arguments passed to
    __init__. Attributes will only be included in the repr if they are in
    the class dict and appear in __init__ as a named parameter (with the
    same name).

    Examples
    --------
    @auto_repr
    class Foo:
        def __init__(self, a, b=6, c=None, p=0.5, **kwargs):
            self.a = a
            self.b = b
            # Different name to demonstrate that cat is not included in repr.
            self.cat = c
            # Property is not stored in class dict, not included in repr.
            self.p = p

        @property
        def p(self):
            return self._p

        @p.setter
        def p(self, val):
            if val > 0:
                self._p = val
            else:
                raise ValueError('p must be non-negative')

    >>> f = Foo(3, b='b', c='c')
    >>> f

    Foo(a=3, b='b')
    """

    def _repr(instance):
        args = dict(inspect.signature(instance.__init__).parameters)
        arg_strs = (f'{k}={repr(v)}' for k, v in instance.__dict__.items()
                    if k in args.keys())
        return f'{type(instance).__name__}({", ".join(arg_strs)})'

    cls.__repr__ = _repr
    return cls


def delegate(attr, iter_magics=False, skip=(), getattr_=True):
    """Decorator that automatically delegates attribute calls to an attribute
    of the class. This is a nice convenience to have when using composition.
    User can also choose to delegate magic methods related to iterables.

    Note: I suspect this could lead to some unexpected behavior so be careful
    using this in production.

    KNOWN ISSUE: Max recursion error when a class inherits from nn.Module and
    delegates to the actual model.

    Parameters
    ----------
    attr: str
        Name of variable to delegate to.
    iter_magics: bool
        If True, delegate the standard magic methods related to iterables:
        '__getitem__', '__setitem__', '__delitem__', and '__len__'.
    skip: Iterable[str]
        Can optionally provide a list of iter_magics to skip. This only has
        an effect when `iter_magics` is True. For example, you may want to be
        able to iterate over the class but no allow item deletion. In this case
        you should pass skip=('__delitem__').
    getattr_: bool
        If True, delegate non-magic methods. This means that if you try to
        access an attribute or method that the object produced by the decorated
        class does not have, it will look for it in the delegated object.

    Examples
    --------
    Example 1: We can use BeautifulSoup methods like `find_all` directly on
    the Page object. Most IDEs should let us view quick documentation as well.

    @delegate('soup')
    class Page:
        def __init__(self, url, logfile, timeout):
            self.soup = self.fetch(url, timeout=timeout)
        ...

    page = Page('http://www.coursera.org')
    page.find_all('div')

    Example 2: Magic methods except for __delitem__ are delegated.

    @delegate('data', True, skip=('__delitem__'))
    class Foo:
        def __init__(self, data, city):
            self.data = data
            self.city = city

    >>> f = Foo(['a', 'b', 'c'], 'San Francisco')
    >>> len(f)
    3

    >>> for char in f:
    >>>     print(char)
    a
    b
    c

    >>> f.append(3); f.data
    ['a', 'b', 'c', 3]

    >>> del f[0]
    TypeError: 'Foo' object doesn't support item deletion

    >>> f.clear(); f.data
    []
    """
    def wrapper(cls):
        def _delegate(self, attr):
            """Helper that retrieves object that an instance delegates to."""
            return getattr(self, attr)

        # Any missing attribute will be delegated.
        if getattr_:
            def _getattr(self, new_attr):
                return getattr(_delegate(self, attr), new_attr)

            cls.__getattr__ = _getattr

        # If specified, delegate magic methods to make cls iterable.
        if iter_magics:
            if '__getitem__' not in skip:
                def _getitem(self, i):
                    return _delegate(self, attr)[i]

                setattr(cls, '__getitem__', _getitem)

            if '__setitem__' not in skip:
                def _setitem(self, i, val):
                    _delegate(self, attr)[i] = val

                setattr(cls, '__setitem__', _setitem)

            if '__delitem__' not in skip:
                def _delitem(self, i):
                    del _delegate(self, attr)[i]

                setattr(cls, '__delitem__', _delitem)

            if '__len__' not in skip:
                def _len(self):
                    return len(_delegate(self, attr))

                setattr(cls, '__len__', _len)
        return cls

    return wrapper


class LoggerMixin:
    """Mixin class that configures and returns a logger.

    Examples
    --------
    class Foo(LoggerMixin):

        def __init__(self, a, log_file):
            self.a = a
            self.log_file = log_file
            self.logger = self.get_logger(log_file)

        def walk(self, location):
            self.logger.info(f'walk received argument {location}')
            return f'walking to {location}'
    """

    def get_logger(self, path=None, fmode='a', level='info',
                   fmt='%(asctime)s [%(levelname)s]: %(message)s'):
        """
        Parameters
        ----------
        path: str or None
            If provided, this will be the path the logger writes to.
            If left as None, logging will only be to stdout.
        fmode: str
            Logging mode when using a log file. Default 'a' for
            'append'. 'w' will overwrite the previously logged messages. Note:
            this only affects what happens when we create a new logger ('w'
            will remove any existing text in the log file if it exists, while
            'a' won't. But calling `logger.info(my_msg)` twice in a row with
            the same logger will always result in two new lines, regardless of
            mode.
        level: str
            Minimum level necessary to log messages.
            One of ('debug', 'info', 'warning', 'error')
        fmt: str
            Format that will be used for logging messages. This uses the
            logging module's formatting language, not standard Python string
            formatting.

        Returns
        -------
        logging.logger
        """
        # When working in Jupyter, need to reset handlers.
        # Otherwise every time we run a cell creating an
        # instance of the logged class, the list of handlers will grow.
        logger = logging.getLogger(type(self).__name__)
        logger.handlers.clear()
        logger.setLevel(getattr(logging, level.upper()))

        # handler.basicConfig() doesn't work in Jupyter.
        formatter = logging.Formatter(fmt)
        handlers = [logging.StreamHandler(sys.stdout)]
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handlers.append(logging.FileHandler(path, fmode))
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


@delegate('logger')
class MultiLogger(LoggerMixin):
    """Easy way to get a pre-configured logger. This can also be used to
    record stdout, either through the context manager provided by contextlib
    or the function decorator defined in this module.

    It delegates to its logger and should be used as follows when explicitly
    called by the user:

    logger = MultiLogger('train.log')
    logger.info('Starting model training.')

    Notice we call the `info` method rather than `write`.
    """

    def __init__(self, path, fmode='w', fmt='%(message)s'):
        """
        Parameters
        ----------
        path: str or Path
            The log file to save to. If None is provided, will only log to
            stdout.
        fmode: str
            One of ('a', 'w'). See `LoggerMixin` docstring: this only affects
            behavior on the first write.
        fmt: str
            Message format. See `LoggerMixin` docstring for details.
        """
        self.logger = self.get_logger(path, fmode, 'info', fmt)

    def write(self, buf):
        """Provided for compatibility with `redirect_stdout` to allow logging
        of stdout while still printing it to the screen. The user should never
        call this directly.
        """
        if buf != '\n':
            self.logger.info(buf)


def verbose_log(path, fmode='w', fmt='%(message)s'):
    """Decorator to log stdout to a file while also printing it to the screen.
    Commonly used for model training.

    Parameters
    ----------
    path: str or Path
        Log file.
    fmode: str
        One of ('a', 'w') for 'append' mode or 'write' mode. Note that 'w' only
        overwrites the existing file once when the decorated function is
        defined: subsequent calls to the function will not overwrite previously
        logged content.
    fmt: str
        String format for logging messages. Uses formatting specific to
        `logging` module, not standard Python string formatting.
    """
    def decorator(func):
        logger = MultiLogger(path, fmode, fmt)
        @wraps(func)
        def wrapper(*args, **kwargs):
            with redirect_stdout(logger):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class SaveableMixin:
    """Provide object saving and loading methods. If you want to be able to
    pass a file name rather than a full path to `save`, the object can define
    a `self.dir` attribute.
    """

    def save(self, path=None, fname=None):
        """Pickle object with optional compression.

        Parameters
        ----------
        path: str or Path
            Path to save object to.
        fname: str or Path
            If passed in, method will use this as a filename within the
            object's `dir` attribute.
        """
        assert not fname or not path, 'Can\'t pass in both fname and path.'
        path = path or Path(self.dir) / fname
        save(self, path)

    @classmethod
    def load(cls, path):
        """Load object from pickle file.

        Parameters
        ----------
        path: str or Path
            Name of file where object is stored.
        """
        return load(path)


def chainmethod(func):
    """Decorator for methods in classes that want to implement
    eager chaining. Chainable methods should be instance methods
    that change 1 or more instance attributes and return None. All this
    decorator does is ensure these methods are called on a deep copy of the
    instance instead of on the instance itself so that operations don't affect
    the original object. The new object is returned.

    Examples
    --------
    @auto_repr
    class EagerChainable:

        def __init__(self, arr, b=3):
            self.arr = arr
            self.b = b

        @chainmethod
        def double(self):
            self.b *= 2

        @chainmethod
        def add(self, n):
            self.arr = [x+n for x in self.arr]

        @chainmethod
        def append(self, n):
            self.arr.append(n)

    >>> ec = EagerChainable([1, 3, 5, -22], b=17)
    >>> ec

    EagerChainable(arr=[1, 3, 5, -22], b=17)

    >>> ec2 = ec.append(99).double().add(400)
    >>> ec2

    EagerChainable(arr=[401, 403, 405, 378, 499], b=34)

    >>> ec   # Remains unchanged.
    EagerChainable(arr=[1, 3, 5, -22], b=17)
    """
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        new_inst = deepcopy(instance)
        func(new_inst, *args, **kwargs)
        return new_inst
    return wrapper


def lazychain(func):
    """Decorator to register a method as chainable within a
    LazyChainable class.
    """
    func._is_chainable = True
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapped


class LazyChainMeta(type):
    """Metaclass to create LazyChainable objects."""

    def __new__(cls, name, bases, methods):
        new_methods = {}

        # Find chainable staticmethods and create public versions.
        for k, v in methods.items():
            try:
                func = v.__get__(1)
                assert func._is_chainable
            except:
                continue
            public_name = k.lstrip('_')

            # Get args and kwargs passed to staticmethod (except for instance).
            sig = inspect.signature(func)
            sig = sig.replace(parameters=list(sig.parameters.values())[1:])

            # Must use default args so they are evaluated within loop.
            def make_public_method(func=func, private_name=k,
                                   public_name=public_name, sig=sig):
                def public(inst, *args, **kwargs):
                    bound = sig.bind(*args, **kwargs).arguments
                    new_method = partial(getattr(inst, private_name), **bound)
                    inst.ops.append(new_method)
                    return inst

                public.__name__ = public_name
                return public

            new_methods[public_name] = make_public_method()

        return type.__new__(cls, name, bases, {**methods, **new_methods})


class LazyChainable(metaclass=LazyChainMeta):
    """Base class that allows children to lazily chain methods,
    similar to a Spark RDD.

    Chainable methods must be decorated with @staticmethod
    and @chainmethod and be named with a leading underscore. A public
    method without the leading underscore will be created, so don't
    overwrite this with another method. Chainable methods
    accept an instance of the same class as the first argument,
    process the instance in some way, then return it. A chain of
    commands will be stored until the exec() method is called.
    It can operate either in place or not.

    Examples
    --------
    class Sequence(LazyChainable):

        def __init__(self, numbers, counter, new=True):
            super().__init__()
            self.numbers = numbers
            self.counter = counter
            self.new = new

        @staticmethod
        @lazychain
        def _sub(instance, n):
            instance.counter -= n
            return instance

        @staticmethod
        @lazychain
        def _gt(instance, n=0):
            instance.numbers = list(filter(lambda x: x > n, instance.numbers))
            return instance

        @staticmethod
        @lazychain
        def _call(instance):
            instance.new = False
            return instance

        def __repr__(self):
            pre, suf = super().__repr__().split('(')
            argstrs = (f'{k}={repr(v)}' for k, v in vars(self).items())
            return f'{pre}({", ".join(argstrs)}, {suf}'


    >>> seq = Sequence([3, -1, 5], 0)
    >>> output = seq.sub(n=3).gt(0).call().exec()
    >>> output

    Sequence(ops=[], numbers=[3, 5], counter=-3, new=False)

    >>> seq   # Unchanged because exec was not in place.

    Sequence(ops=[], numbers=[3, -1, 5], counter=0, new=True)


    >>> output = seq.sub(n=3).gt(-1).call().exec(inplace=True)
    >>> output   # None because exec was in place.
    >>> seq   # Changed

    Sequence(ops=[], numbers=[3, -1, 5], counter=-3, new=False)
    """

    def __init__(self):
        self.ops = []

    def exec(self, inplace=False):
        new = deepcopy(self)
        for func in self.ops:
            new = func(copy(new))
        # Clear ops list now that chain is complete.
        new.ops.clear()
        if inplace:
            self.__dict__ = new.__dict__
        else:
            self.ops.clear()
            return new

    def __repr__(self):
        argstrs = (f'{k}={repr(v)}' for k, v in vars(self).items())
        return f'{type(self).__name__}({", ".join(argstrs)})'


def params(func):
    """Get parameters in a functions signature.

    Parameters
    ----------
    func: function

    Returns
    -------
    dict: Maps name (str) to Parameter.
    """
    return dict(inspect.signature(func).parameters)


def hasarg(func, arg):
    """Check if a function has a parameter with a given name. (Technically,
    hasparam might be a more appropriate name but hasarg lets us match the
    no-space convention of hasattr and getattr while maintaining readability.)

    Parameters
    ----------
    func: function
    arg: str
        The name of the parameter that you want to check for in func's
        signature.

    Returns
    -------
    bool: True if `func` has a parameter named `arg`.
    """
    return arg in params(func)


def bound_args(func, args, kwargs, collapse_kwargs=True):
    """Get the bound arguments for a function (with defaults applied). This is
    very commonly used when building decorators that log, check, or alter how
    a function was called.

    Parameters
    ----------
    func: function
    args: tuple
        Notice this is not *args. Just pass in the tuple.
    kwargs: dict
        Notice this is not **kwargs. just pass in the dict.
    collapse_kwargs: bool
        If True, collapse kwargs into the regular parameter dict. E.g.
        {'a': 1, 'b': True, 'kwargs': {'c': 'c_val', 'd': 0}} ->
        {'a': 1, 'b': True, 'c': 'c_val', 'd': 0}

    Returns
    -------
    OrderedDict[str, any]: Maps parameter name to passed value.
    """
    bound = inspect.signature(func).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    args = bound.arguments
    if not collapse_kwargs: return args
    args.update(args.pop('kwargs', {}))
    return args


def handle_interrupt(func=None, cbs=(), verbose=True):
    """Decorator that allows us to interrupt a function with ctrl-c. We can
    pass in callbacks that execute on function end. Keep in mind that local
    variables will be lost as soon as `func` stops running. If `func` is a
    method, it may be appropriate to update instance variables while running,
    which we can access because the instance will be the first element of
    `args` (passed in as `self`).

    Note: Kwargs are passed to callbacks as a single dict, not as **kwargs.

    Parameters
    ----------
    func: function
    cbs: Iterable[Callback]
        List of callbacks to execute when `func` completes. These will execute
        whether we interrupt or not.
    verbose: bool
        If True, print a message to stdout when an interrupt occurs.
    """
    if not func: return partial(handle_interrupt, cbs=cbs, verbose=verbose)
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_inputs = bound_args(func, args, kwargs, collapse_kwargs=False)
        try:
            res = func(*args, **kwargs)
        except KeyboardInterrupt:
            if verbose: print('KeyboardInterrupt. Aborting...')
            res = None
        finally:
            for cb in cbs:
                cb.on_end(func, func_inputs, res)
        return res
    return wrapper


@contextmanager
def assert_raises(error):
    """Context manager to assert that an error is raised. This can be nice
    if we don't want to clutter up a notebook with error messages.

    Parameters
    ----------
    error: class inheriting from Exception or BaseException
        The type of error to catch, e.g. ValueError.

    Examples
    --------
    # First example does not throw an error.
    >>> with assert_raises(TypeError) as ar:
    >>>     a = 'b' + 6

    # Second example throws an error.
    >>> with assert_raises(ValueError) as ar:
    >>>     a = 'b' + 6

    AssertionError: Wrong error raised. Expected PermissionError, got
    TypeError(can only concatenate str (not "int") to str)

    # Third example throws an error because the code inside the context manager
    # completed successfully.
    >>> with assert_raises(ValueError) as ar:
    >>>     a = 'b' + '6'

    AssertionError: No error raised, expected PermissionError.
    """
    try:
        yield
    except error as e:
        print(f'As expected, got {error.__name__}({e}).')
    except Exception as e:
        raise AssertionError(f'Wrong error raised. Expected {error.__name__},'
                             f' got {type(e).__name__}({e}).') from None
    else:
        raise AssertionError(f'No error raised, expected {error.__name__}.')


class TimeExceededError(Exception):
    pass


def timebox_handler(time, frame):
    raise TimeExceededError('Time limit exceeded.')


@contextmanager
def timebox(seconds, strict=True, freq=.1):
    """Try to execute code for specified amount of time before throwing error.
    If you don't want to throw an error, use with a try/except block.

    Parameters
    ----------
    seconds: float
        Max number of seconds before throwing error. This will be enforced with
        a relatively low level of precision.
    strict: bool
        If True, timeout will cause an error to be raised, halting execution of
        the entire program. If False, a warning message will be printed and
        the timeboxed operation will end, letting the program proceed to the
        next step.
    freq: float
        How often to update progress bar (measured in seconds).

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

    def update_custom_pbar(signum, frame):
        """Handler that is called every `freq` seconds. User never calls this
        directly.
        """
        pbar.update(n=freq)
        if time.time() - pbar.start_t >= seconds:
            raise TimeExceededError('Time limit exceeded.')

    pbar = tqdm(total=seconds, bar_format='{l_bar}{bar}|{n:.2f}/{total:.1f}s')
    try:
        signal.signal(signal.SIGALRM, update_custom_pbar)
        signal.setitimer(signal.ITIMER_REAL, freq, freq)
        yield
    except TimeExceededError as e:
        if strict: raise
        warnings.warn(e.args[0])
    finally:
        pbar.close()
        signal.alarm(0)


def timeboxed(time, strict=True, freq=.1):
    """Decorator version of timebox. Try to execute decorated function for
    `time` seconds before throwing exception.

    Parameters
    ----------
    time: float
        Max number of seconds before throwing error. This will be enforced with
        a relatively low level of precision.
    strict: bool
        If True, timeout will cause an error to be raised, halting execution of
        the entire program. If False, a warning message will be printed and
        the timeboxed operation will end, letting the program proceed to the
        next step.
    freq: float
        How often to update the progress bar (measured in seconds).

    Examples
    --------
    @timeboxed(5)
    def func(x, y):
        # If function does not complete within 5 seconds, will throw error.
    """
    def intermediate_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timebox(time, strict, freq) as tb:
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


class ReadOnly:
    """Descriptor to make an attribute read-only. This means that once a value
    has been set, the user cannot change or delete it. Note that read-only
    attributes must first be created as class variables (see example below).
    To allow more flexibility, we do allow the user to manually manipulate the
    instance dictionary.

    Examples
    --------
    class Dog:
        breed = ReadOnly()
        def __init__(self, breed, age):
            # Once breed is set in the line below, it cannot be changed.
            self.breed = breed
            self.age = age

    >>> d = Dog('dalmatian', 'Arnold')
    >>> d.breed

    'dalmatian'

    >>> d.breed = 'labrador'

    PermissionError: Attribute is read-only.

    >>> del d.breed

    PermissionError: Attribute is read-only.
    """

    def __init__(self):
        # self.name = None
        self.initialized = WeakSet()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        elif instance not in self.initialized:
            warnings.warn(
                f'Read-only attribute {self.name} has not been initialized.'
            )
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if instance not in self.initialized:
            instance.__dict__[self.name] = value
            self.initialized.add(instance)
        else:
            raise PermissionError('Attribute is read-only.')

    def __delete__(self, instance):
        raise PermissionError('Attribute is read-only.')


def validating_property(func, allow_del=False):
    """Factory that makes properties that perform some user-specified
    validation when setting values. The returned function must be used as a
    descriptor to create a class variable before setting the instance
    attribute.

    Parameters
    ----------
    func: function
        Function or lambda that accepts a single parameter. This will be used
        when attempting to set a value for the managed attribute. It should
        return True if the value is acceptable, False otherwise.
    allow_del: bool
        If True, allow the attribute to be deleted.

    Returns
    -------
    function: A property with validation when setting values. Note that this
        will be used as a descriptor, so it must create a class variable as
        shown below. In the example, also notice that the name passed to
        LengthyInt mustt match the name of the variable it is assigned to.

    Examples
    --------
    LengthyInt = validating_property(
        lambda x: isinstance(x, int) and len(str(int)) > 4
    )

    class Foo:
        long = LengthyInt('long')
        def __init__(self, a, long):
            self.a = a
            self.long = long

    >>> foo = Foo(3, 4)

    ValueError: Invalid value 4 for argument long.

    # No error on instantiation because the argument is a valid LengthyInt.
    >>> foo = Foo(3, 543210)
    >>> foo.long

    543210

    >>> foo = Foo(3, 'abc')
    ValueError: Invalid value 'abc' for argument long.
    """
    def prop(name):
        @property
        def method(instance):
            return instance.__dict__[name]

        @method.setter
        def method(instance, val):
            if func(val):
                instance.__dict__[name] = val
            else:
                raise ValueError(f'Invalid value {val} for argument {name}.')

        if allow_del:
            @method.deleter
            def method(instance):
                del instance.__dict__[name]
        return method
    return prop


class Callback(ABC):
    """Abstract base class for callback objects to be passed to @callbacks
    decorator. Children must implement on_begin and on_end methods. Both should
    accept the decorated function's inputs and output as arguments

    Often, we may want to use the @debug decorator on one or both of these
    methods. If both methods should perform the same steps, one shortcut
    is to implement a single undecorated __call__ method, then have the
    debug-decorated on_begin and on_end methods return self(inputs, output).
    """

    @abstractmethod
    def setup(self, func):
        """
        Parameters
        ----------
        func: function
            The function being decorated.

        """

    @abstractmethod
    def on_begin(self, func, inputs, output=None):
        """
        Parameters
        ----------
        func: function
            The function being decorated.
        inputs: dict
            Dictionary of bound arguments passed to the function being
            decorated with @callbacks.
        output: any
            Callbacks to be executed after the function call can pass the
            function output to the callback. The default None value will remain
            for callbacks that execute before the function.
        """

    @abstractmethod
    def on_end(self, func, inputs, output=None):
        """
        Parameters
        ----------
        func: function
            The function being decorated.
        inputs: dict
            Dictionary of bound arguments passed to the function being
            decorated with @callbacks.
        output: any
            Callbacks to be executed after the function call can pass the
            function output to the callback. The default None value will remain
            for callbacks that execute before the function.
        """

    def __repr__(self):
        return f'{type(self).__name__}()'


def callbacks(cbs):
    """Decorator that attaches callbacks to a function. Callbacks should be
    defined as classes inheriting from abstract base class Callback that
    implement on_begin and on_end methods. This allows us to store states
    rather than just printing outputs or relying on global variables.

    Parameters
    ----------
    cbs: list
        List of callbacks to execute before and after the decorated function.

    Examples
    --------
    @callbacks([PrintHyperparameters(), PlotActivationHist(),
                ActivationMeans(), PrintOutput()])
    def train_one_epoch(**kwargs):
        # Train model.
    """
    def decorator(func):
        for cb in cbs:
            cb.setup(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for cb in cbs:
                cb.on_begin(func, bound.arguments, None)
            out = func(*args, **kwargs)
            for cb in cbs:
                cb.on_end(func, bound.arguments, out)
            return out
        return wrapper
    return decorator


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
        Optional way to specify variable types. Use standard types rather than
        importing from the typing library, as subscripted generics are not
        supported (e.g. typing.List[str] will not work; typing.List will but at
        that point there is no benefit over the standard `list`).

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
    def wrapper(*args, **kwargs):
        fargs = inspect.signature(wrapper).bind(*args, **kwargs).arguments
        for k, v in types.items():
            if k in fargs and not isinstance(fargs[k], v):
                raise TypeError(
                    f'{k} must be {str(v)}, not {type(fargs[k])}.'
                )
        return func_(*args, **kwargs)
    return wrapper


def valuecheck(func):
    """Decorator that checks if user-specified arguments are acceptable.
    Because this re-purposes annotations to specify values rather than types,
    this can NOT be used together with the @typecheck decorator. Keep in mind
    that this tests for equality, so 4 and 4.0 are considered equivalent.

    Parameters
    ----------
    func: function
        The function to decorate. Use annotations to specify acceptable values
        as tuples, as shown below.

    Examples
    --------
    @valuecheck
    def foo(a, b:('min', 'max'), c=6, d:(True, False)=True):
        return d, c, b, a

    >>> foo(3, 'min')
    (True, 6, 'min', 3)

    >>> foo(True, 'max', d=None)
    ValueError: Invalid argument for parameter d. Value must be in
    (True, False).

    >>> foo('a', 'mean')
    ValueError: Invalid argument for parameter b. Value must be in
    ('min', 'max').
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        annos = {k: v.annotation for k, v in sig.parameters.items()}
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for k, v in bound.arguments.items():
            choices = annos[k]
            if choices == inspect._empty: continue
            if v not in choices:
                raise ValueError(f'Invalid argument for parameter {k}. '
                                 f'Value must be in {choices}.')
        return func(*args, **kwargs)

    return wrapper


def debug(func=None, prefix='', arguments=True):
    """Decorator that prints information about a function call. Often, this
    will only be used temporarily when debugging. Note that a wrapped function
    that accepts *args will display a signature including an 'args' parameter
    even though it isn't a named parameter, because the goal here is to
    explicitly show which values are being passed to which parameters. This
    does mean that the printed string won't be executable code in this case,
    but that shouldn't be necessary anyway since it would contain the same call
    that just occurred.

    The decorator can be used with or without arguments.

    Parameters
    ----------
    func: function
        Function being decorated.
    prefix: str
        A short string to prepend the printed message with. Ex: '>>>'
    arguments: bool
        If True, the printed message will include the function arguments.
        If False, it will print the function name but not its arguments.

    Examples
    --------
    Occasionally, you might pass arguments to different parameters than you
    intended. Throwing a debug_call decorator on the function helps you check
    that the arguments are matching up as expected. For example, the parameter
    names in the function below have an unexpected order, so you could easily
    make the following call and expect to get 8. The debug decorator helps
    catch that the third argument is being passed in as the x parameter.

    @debug
    def f(a, b, x=0, y=None, z=4, c=2):
        return a + b + c

    >>> f(3, 4, 1)
    CALLING f(a=3, b=4, x=1, y=None, z=4, c=2)
    9

    @debug(prefix='***', arguments=False)
    def f(a, b, x=0, y=None, z=4, c=2):
        return a + b + c

    >>> f(3, 4, 1)
    *** CALLING f()
    9
    """
    if not func:
        if prefix: prefix += ' '
        return partial(debug, prefix=prefix, arguments=arguments)

    @wraps(func)
    def wrapper(*args, **kwargs):
        out_fmt = '\n{}CALLING {}({})'
        arg_strs = ''
        if arguments:
            sig = bound_args(wrapper, args, kwargs, collapse_kwargs=True)
            if sig:
                first_key = next(iter(sig))
                # Remove self/cls arg from methods.
                if first_key in ('self', 'cls'):
                    del sig[first_key]
            arg_strs = (f'{k}={repr(v)}' for k, v in sig.items())

        # Print call message and return output.
        print(out_fmt.format(prefix, func.__qualname__, ', '.join(arg_strs)))
        return func(*args, **kwargs)

    return wrapper


def log_stdout(func=None, fname=''):
    """Decorator that logs all stdout produced by a function.

    Parameters
    ----------
    func: function
        If the decorator is used without parenthesis, the function will be
        passed in as the first argument. You never need to explicitly specify
        a function.
    fname: str
        Path to log file which will be created. If None is specified, the
        default is to write to ./logs/wrapped_func_name.log. If specified,
        this must be a keyword argument.

    Examples
    --------
    @log_stdout
    def foo(a, b=3):
        print(a)
        a *= b
        print(a)
        return a**b

    @log_stdout(fname='../data/mylog.log')
    def foo(a, b=3):
        ...
    """
    if not func:
        return partial(log_stdout, fname=Path(fname))
    if not fname:
        fname = Path(f'./logs/{func.__name__}.log')

    @wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs(fname.parent, exist_ok=True)
        with open(fname, 'w') as f:
            with redirect_stdout(f):
                out = func(*args, **kwargs)
        return out

    return wrapper


def return_stdout(func):
    """Decorator that returns printed output from the wrapped function. This
    may be useful if we define a function that only prints information and
    returns nothing, then later decide we want to access the printed output.
    Rather than re-writing everything, we can slap a @return_stdout decorator
    on top and leave it as is. This should not be used if the decorated
    function already returns something else since we will only return what is
    printed to stdout. For that use case, consider the `log_stdout` function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = io.StringIO()
        with redirect_stdout(res):
            func(*args, **kwargs)
        return res.getvalue()
    return wrapper


def log_cmd(path, mode='a'):
    """Decorator that saves the calling command for a python script. This is
    often useful for CLIs that train ML models. It makes it easy to re-run
    the script at a later date with the same or similar arguments.

    Parameters
    ----------
    path: str or Path
        Specifies file where output will be saved.
    mode: str
        Determines whether output should overwrite old file or be appended.
        One of ('a', 'w'). In most cases we will want append mode because we're
        tracking multiple trials.

    Examples
    --------
    ```
    # train.py
    import fire

    @log_cmd('logs/training_runs.txt')
    def train(lr, epochs, dropout, arch, data_version):
        # Train model

    if __name__ == '__main__':
        fire.Fire(train)
    ```

    $ python train.py --lr 3e-3 --epochs 50 --dropout 0.5 --arch awd_lstm \
        --data_version 1

    After running the script with the above command, the file
    'logs/training_runs.txt' now contains a nicely formatted version of the
    calling command with a separate line for each argument name/value pair.

    We can also use variables that are passed to our function. All function
    args and kwargs will be passed to the string formatter so your variable
    names must match:

    @log_cmd('logs/train_run_v{version_number}.{ext}')
    def train(version_number, ext, epochs, arch='lstm'):
        # Train model
    """
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            fn_locals = bound_args(func, args, kwargs, True)
            res = 'python'
            for arg in sys.argv:
                pre = ' \\\n\t' if arg.startswith('-') else ' '
                res += pre+arg
            save(res+'\n\n', Path(path.format(**fn_locals)), mode)
            return func(*args, **kwargs)
        return wrapped
    return decorator


def wrapmethods(*decorators, methods=(), internals=False):
    """Class wrapper that applies 1 or more decorators to every non-magic
    method (properties are also excluded). For example, we often want @debug
    to be applied to many different methods.

    Parameters
    ----------
    decorators: callable
        1 or more decorators to apply to methods within a class. By default,
        methods with 1 or 2 leading underscores are excluded.
    methods: Iterable[str]
        Names of methods to wrap if you don't want to wrap all of them.
        Internal methods can be wrapped but magic methods and properties
        cannot.
    internals: bool
        If True, apply decorators to methods named with leading single
        underscores. This will be ignored if `methods` is specified.
    """

    def wrapper(cls):
        special_methods = (staticmethod, classmethod)
        to_wrap = dict.fromkeys(methods, True) if methods \
            else hdir(cls, False, internals=internals)
        for attr, is_method in to_wrap.items():
            f = cls.__dict__[attr]
            if not is_method or isinstance(f, property):
                continue

            # Classmethod and staticmethod decorators need to be applied last.
            final_wrapper = identity
            if isinstance(f, special_methods):
                final_wrapper = type(f)
                f = f.__func__

            for d in decorators:
                f = d(f)
            setattr(cls, attr, final_wrapper(f))
        return cls
    return wrapper


def add_docstring(func):
    """Add the docstring from another function/class to the decorated
    function/class.

    Examples
    --------
    @add_docstring(nn.Conv2d)
    class ReflectionPaddedConv2d(nn.Module):
        ...
    """
    def decorator(new_func):
        new_func.__doc__ = f'{new_func.__doc__}\n\n{func.__doc__}'
        @wraps(new_func)
        def wrapper(*args, **kwargs):
            return new_func(*args, **kwargs)
        return wrapper
    return decorator


def timer(func):
    """Provide conservative time estimate for a function to run. Behavior may
    not be interpretable for recursive functions.

    Parameters
    -----------
    func: function
        The function to time.

    Examples
    ---------
    import time

    @timer
    def count_to(x):
        for i in range(x):
            time.sleep(0.5)

    >>> count_to(10)
    [TIMER]: count_to executed in approximately 5.0365 seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f'\n[TIMER]: {func.__name__} executed in approximately '
              f'{duration:.3f} seconds.\n')
        return output
    return wrapper


@contextmanager
def block_timer():
    """Context manager to time a block of code. This works similarly to @timer
    but can be used on code outside of functions.

    Examples
    --------
    with block_timer() as bt:
        # Code inside the context manager will be timed.
        arr = [str(i) for i in range(25_000_000)]
        first = None
        while first != '100':
            arr.pop(0)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        print(f'[TIMER]: Block executed in {duration:.3f} seconds.')


def count_calls(func):
    """Count the number of times a function has been called. The function can
    access this value inside itself through the attribute 'calls'. Note that
    counting is defined such that during the first call, func.calls already=1
    (i.e. it can be considered the n'th call, not that n calls have previously
    taken place not counting the current one).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        calls = getattr(wrapper, 'calls', 0)
        wrapper.calls = calls + 1
        return func(*args, **kwargs)
    return wrapper


def copy_func(func):
    """Copy a function. Regular copy and deepcopy functionality do not work
    on functions the way they do on most objects. If we want to create a new
    function based on another without altering the old one (as in
    `rename_params`), this should be used.

    Parameters
    ----------
    func: function
        Function to duplicate.

    Returns
    -------
    function: Copy of input `func`.

    Examples
    --------
    def foo(a, b=3, *args, c=5, **kwargs):
        return a, b, c, args, kwargs

    foo2 = copy_func(foo)

    >>> foo2.__code__ == foo.__code__
    True

    >>> foo2 == foo
    False
    """
    new_func = types.FunctionType(func.__code__, func.__globals__,
                                  func.__name__, func.__defaults__,
                                  func.__closure__)
    defaults = getattr(func, '__kwdefaults__') or {}
    new_func.__kwdefaults__ = defaults.copy()
    return update_wrapper(new_func, func)


def rename_params(func, **old2new):
    """Rename one or more parameters. Docstrings and default arguments are
    updated accordingly. This is useful when working with code that uses
    `hasarg`. For example, my Incendio library uses parameter names
    to pass the correct arguments to different metrics.

    Parameters
    ----------
    func: function
        The old function to change.
    old2new: str
        One or more parameter names to change and their corresponding new
        names. See Example below for a more concrete example.

    Returns
    -------
    function: Same as input `func` but with updated parameter names.

    Examples
    --------
    def foo(a, b, *args, c=3, **kwargs):
        pass

    foo_metric = rename_params(func, a=y_true, b=y_pred)

    `foo_metric` will work exactly like `foo` but its first two parameters will
    now be named "y_true" and "y_pred", respectively. """
    new_func = copy_func(func)
    sig = inspect.signature(new_func)
    kw_defaults = func.__kwdefaults__ or {}
    names, params = map(list, zip(*sig.parameters.items()))
    for old, new in old2new.items():
        idx = names.index(old)
        default = kw_defaults.get(old) or params[idx].default
        params[idx] = inspect.Parameter(new, params[idx].kind, default=default)
    new_func.__signature__ = sig.replace(parameters=params)
    return new_func


def immutify_defaults(func):
    """Decorator to make a function's defaults arguments effectively immutable.
    We accomplish this by storing the initially provided defaults and assigning
    them back to the function's signature after each call. If you use a
    variable as a default argument, this does not mean that the variable's
    value will remain unchanged - it just ensures the initially provided value
    will be used for each call.
    """
    # If `__hash__` is not None, object is immutable already.
    # Python sets __defaults__ and __kwdefaults__ to None when they're empty.
    _defaults = tuple(o if getattr(o, '__hash__') else deepcopy(o)
                      for o in getattr(func, '__defaults__') or ()) or None
    _kwdefaults = {k: v if getattr(v, '__hash__') else deepcopy(v) for k, v
                   in (getattr(func, '__kwdefaults__') or {}).items()} or None

    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        wrapper.__defaults__ = func.__defaults__ = deepcopy(_defaults)
        wrapper.__kwdefaults__ = func.__kwdefaults__ = deepcopy(_kwdefaults)
        return res
    return wrapper

